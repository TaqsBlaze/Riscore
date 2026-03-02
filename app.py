"""
Real-Time Predictive Risk Assessment & Alert System
Microfinance Organization Prototype - v2

Async Architecture:
  * All Flask route handlers are `async def` (Flask 2.0+ native support)
  * Database I/O  -> SQLAlchemy AsyncSession + aiomysql driver (non-blocking)
  * ML inference  -> ThreadPoolExecutor via run_in_executor (CPU-bound, off event loop)
  * Alert dispatch -> asyncio.create_task  (fire-and-forget, response doesn't wait)
  * Stats queries  -> asyncio.gather       (all 8 DB calls run in parallel)
  * Startup        -> asyncio.run(init_db / seed_data) before Flask starts
"""

import os, asyncio, random, datetime, pickle, re, uuid, logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from flask import Flask, jsonify, request, render_template, url_for
from flask_cors import CORS

# SQLAlchemy async
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, func, select, update, desc, asc
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

# --------------------------------------------------------------
#  Logging
# --------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("risksense")

# --------------------------------------------------------------
#  Config
# --------------------------------------------------------------
load_dotenv()

BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# aiomysql is the async counterpart of PyMySQL - same protocol, same credentials.
# Switch dialect prefix from  mysql+pymysql://  ->  mysql+aiomysql://
DB_URL = (
    "mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"
    "?charset=utf8mb4"
).format(
    user     = os.getenv("MYSQL_USER",     "root"),
    password = os.getenv("MYSQL_PASSWORD", ""),
    host     = os.getenv("MYSQL_HOST",     "localhost"),
    port     = os.getenv("MYSQL_PORT",     "3306"),
    database = os.getenv("MYSQL_DATABASE", "microfinance_db"),
)

# --------------------------------------------------------------
#  Async SQLAlchemy Engine & Session Factory
# --------------------------------------------------------------
engine = create_async_engine(
    DB_URL,
    poolclass = NullPool,  # avoid cross-event-loop connection reuse in Flask async
    echo      = False,     # set True to log generated SQL
)

# expire_on_commit=False prevents "detached instance" errors in async
AsyncSessionFactory = async_sessionmaker(
    engine,
    class_           = AsyncSession,
    expire_on_commit = False,
)


# --------------------------------------------------------------
#  ORM Models
# --------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Borrower(Base):
    __tablename__ = "borrowers"

    id                      = Column(String(20),  primary_key=True)
    full_name               = Column(String(120), nullable=False)
    first_name              = Column(String(60))
    last_name               = Column(String(60))
    salary                  = Column(Float,  default=0.0)
    employment_sector       = Column(String(80))
    job_title               = Column(String(80))
    total_prev_loans        = Column(Float,  default=0.0)
    active_loans            = Column(Float,  default=0.0)
    outstanding_balance     = Column(Float,  default=0.0)
    avg_loan_amount         = Column(Float,  default=0.0)
    common_loan_reason      = Column(String(80),  default="Unknown")
    return_rate             = Column(Float,  default=100.0)
    days_past_due           = Column(Float,  default=0.0)
    mfi_diversity_score     = Column(Float,  default=1.0)
    risk_score              = Column(Float,  default=0.0)
    risk_label              = Column(String(20),  default="Low")
    risk_probability_high   = Column(Float,  default=0.0)
    risk_probability_medium = Column(Float,  default=0.0)
    risk_probability_low    = Column(Float,  default=0.0)
    data_source             = Column(String(40),  default="salary_inference")
    created_at              = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self):
        return {
            c.name: (
                getattr(self, c.name).isoformat()
                if isinstance(getattr(self, c.name), datetime.datetime)
                else getattr(self, c.name)
            )
            for c in self.__table__.columns
        }


class Transaction(Base):
    __tablename__ = "transactions"

    id               = Column(Integer,    primary_key=True, autoincrement=True)
    borrower_id      = Column(String(20))
    type             = Column(String(40))
    amount           = Column(Float,  default=0.0)
    description      = Column(Text)
    is_anomaly       = Column(Boolean, default=False)
    anomaly_score    = Column(Float,  default=0.0)
    risk_score_after = Column(Float,  default=0.0)
    risk_label_after = Column(String(20),  default="Low")
    timestamp        = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self):
        return {
            c.name: (
                getattr(self, c.name).isoformat()
                if isinstance(getattr(self, c.name), datetime.datetime)
                else getattr(self, c.name)
            )
            for c in self.__table__.columns
        }


class Alert(Base):
    __tablename__ = "alerts"

    id            = Column(Integer,    primary_key=True, autoincrement=True)
    borrower_id   = Column(String(20))
    borrower_name = Column(String(120))
    alert_type    = Column(String(60))
    message       = Column(Text)
    severity      = Column(String(20))
    channel       = Column(String(60),  default="Dashboard")
    is_read       = Column(Boolean,     default=False)
    timestamp     = Column(DateTime,    default=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self):
        return {
            c.name: (
                getattr(self, c.name).isoformat()
                if isinstance(getattr(self, c.name), datetime.datetime)
                else getattr(self, c.name)
            )
            for c in self.__table__.columns
        }


# --------------------------------------------------------------
#  DB Init  (async - called once at startup via asyncio.run)
# --------------------------------------------------------------
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("MySQL tables ready")

async def ensure_db_ready():
    """
    Initialize tables only when needed and seed demo data only if empty.
    """
    try:
        async with AsyncSessionFactory() as session:
            result = await session.execute(select(func.count(Borrower.id)))
            count = result.scalar() or 0
        if count > 0:
            log.info("Database already initialized with data; skipping init/seed")
            return
        log.info("Database initialized but empty; seeding demo data")
        await seed_data()
        return
    except Exception as exc:
        log.warning("Database check failed; attempting init/seed", exc_info=True)
        try:
            await init_db()
            await seed_data()
        except Exception:
            log.error("Database init/seed failed; continuing without DB", exc_info=True)
            return


# --------------------------------------------------------------
#  Flask App
# --------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
CORS(app, resources={r"/*": {"origins": ["*", "http://localhost:5500"]}})


# --------------------------------------------------------------
#  Load Trained Model Artefacts  (sync, done once at import time)
# --------------------------------------------------------------
def load_artefacts():
    mp = os.path.join(MODEL_DIR, "risk_model.pkl")
    if not os.path.exists(mp):
        raise FileNotFoundError("Model not found - run `python train_model.py` first.")
    with open(os.path.join(MODEL_DIR, "risk_model.pkl"),    "rb") as f: model    = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: le       = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"),  "rb") as f: feat_col = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metadata.pkl"),      "rb") as f: meta     = pickle.load(f)
    return model, le, feat_col, meta


log.info("Loading trained model artefacts...")
RISK_MODEL, LABEL_ENC, FEATURE_COLS, META = load_artefacts()
log.info(f"Model loaded | Accuracy: {META['accuracy']:.1%} | Classes: {META['label_classes']}")

MFI_DF = pd.read_csv(DATA_PATH)
MFI_DF["_name_key"] = MFI_DF["Full Name"].str.lower().str.strip()
log.info(f"MFI consortium data: {len(MFI_DF)} records")

# --------------------------------------------------------------
#  Thread pool for CPU-bound ML inference
#
#  sklearn's predict_proba holds the GIL and cannot yield.
#  Offloading to a ThreadPoolExecutor means the event loop stays
#  free to serve other requests while scoring is in progress.
# --------------------------------------------------------------
ML_EXECUTOR = ThreadPoolExecutor(
    max_workers        = 4,
    thread_name_prefix = "ml-worker",
)

FREQUENT_APPLICATION_WINDOW_DAYS = 30
FREQUENT_APPLICATION_THRESHOLD = 3

def _parse_float(value, field_name, *, min_value=None):
    try:
        if value is None or value == "":
            raise ValueError
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid number.")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}.")
    return parsed

# --------------------------------------------------------------
#  Feature Engineering  (pure CPU, runs inside executor)
# --------------------------------------------------------------
ALL_SECTORS = [c.replace("Employment Sector_", "")  for c in META["cat_feature_names"]
               if c.startswith("Employment Sector_")]
ALL_REASONS = [c.replace("Common Loan Reason_", "") for c in META["cat_feature_names"]
               if c.startswith("Common Loan Reason_")]


def _build_features(salary, sector, reason, total_loans, active_loans,
                    outstanding, avg_loan, return_rate, days_due, mfi_score):
    """Synchronous CPU work - always called inside ThreadPoolExecutor."""
    salary = max(float(salary), 1)
    row = {
        "Current Monthly Salary (USD)":    salary,
        "Total Previous Loans":            float(total_loans),
        "Active Loans":                    float(active_loans),
        "Total Outstanding Balance (USD)": float(outstanding),
        "Avg Loan Amount (USD)":           float(avg_loan),
        "Historical Return Rate (%)":      float(return_rate),
        "Days Past Due (Max)":             float(days_due),
        "MFI Diversity Score":             float(mfi_score),
        "Debt_to_Income":                  float(outstanding) / salary,
        "Loan_to_Income":                  float(avg_loan)    / salary,
        "Active_Loan_Density":             float(active_loans) / max(float(total_loans), 1),
        "Return_Rate_Norm":                float(return_rate) / 100.0,
        "Is_Overdue":                      int(float(days_due) > 0),
        "Overdue_Severity": (
            0 if float(days_due) == 0 else
            1 if float(days_due) <= 30 else
            2 if float(days_due) <= 60 else
            3 if float(days_due) <= 90 else 4
        ),
    }
    for s in ALL_SECTORS:
        row[f"Employment Sector_{s}"] = int(sector == s)
    for r in ALL_REASONS:
        row[f"Common Loan Reason_{r}"] = int(reason == r)
    return pd.DataFrame([row])[FEATURE_COLS]


def _score_sync(salary, sector, reason, total_loans, active_loans,
                outstanding, avg_loan, return_rate, days_due, mfi_score):
    """Synchronous ML scoring - must run in ThreadPoolExecutor, never on event loop."""
    X         = _build_features(salary, sector, reason, total_loans, active_loans,
                                 outstanding, avg_loan, return_rate, days_due, mfi_score)
    proba     = RISK_MODEL.predict_proba(X)[0]
    classes   = LABEL_ENC.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    label     = classes[int(np.argmax(proba))]
    score     = round(
        prob_dict.get("High",   0) * 100 +
        prob_dict.get("Medium", 0) * 45  +
        prob_dict.get("Low",    0) * 10,
        1,
    )
    return score, label, prob_dict


async def score_borrower_async(salary, sector, reason, total_loans, active_loans,
                                outstanding, avg_loan, return_rate, days_due, mfi_score):
    """
    Async wrapper: offloads CPU-bound ML inference to the thread pool.
    The event loop is free to handle other requests while scoring runs.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        ML_EXECUTOR,
        _score_sync,
        salary, sector, reason, total_loans, active_loans,
        outstanding, avg_loan, return_rate, days_due, mfi_score,
    )

def evaluate_application_anomalies(
    *,
    salary: float,
    total_loans: float,
    active_loans: float,
    outstanding: float,
    return_rate: float,
    days_due: float,
    is_existing_borrower: bool,
    recent_application_count: int,
):
    anomalies = []

    def add(code: str, description: str, score: float):
        anomalies.append({
            "code": code,
            "description": description,
            "score": float(score),
        })

    debt_to_income = float(outstanding) / max(float(salary), 1.0)

    # User applying for a new loan while still carrying active outstanding debt.
    if float(active_loans) > 0 and float(outstanding) > 0:
        add(
            "OUTSTANDING_ACTIVE_LOAN",
            "User is applying while an outstanding active loan exists.",
            20.0,
        )

    if int(recent_application_count) >= FREQUENT_APPLICATION_THRESHOLD:
        add(
            "FREQUENT_LOAN_APPLICATIONS",
            f"High application frequency: {recent_application_count} assessments in {FREQUENT_APPLICATION_WINDOW_DAYS} days.",
            18.0,
        )

    if (not is_existing_borrower) and float(total_loans) <= 0 and float(active_loans) <= 0:
        add(
            "NEW_USER_NO_HISTORY",
            "New user has no previous borrowing records.",
            12.0,
        )

    # Additional anomalies.
    if debt_to_income >= 1.5:
        add(
            "HIGH_DEBT_TO_INCOME",
            f"Debt-to-income ratio is high ({debt_to_income:.2f}).",
            16.0,
        )

    if float(return_rate) < 70:
        add(
            "LOW_REPAYMENT_RATE",
            f"Historical return rate is low ({float(return_rate):.1f}%).",
            14.0,
        )

    if float(days_due) > 60:
        add(
            "SEVERE_PAST_DUE",
            f"Severe delinquency history detected ({float(days_due):.0f} days past due).",
            15.0,
        )

    anomaly_score = round(min(100.0, sum(item["score"] for item in anomalies)), 1)
    return {
        "is_anomaly": len(anomalies) > 0,
        "anomaly_score": anomaly_score,
        "anomalies": anomalies,
    }


# --------------------------------------------------------------
#  MFI Name Lookup  (in-memory pandas - fast, no I/O)
# --------------------------------------------------------------
def lookup_mfi(first_name: str, last_name: str):
    full  = f"{first_name} {last_name}".lower().strip()
    match = MFI_DF[MFI_DF["_name_key"] == full]
    if not match.empty:
        return match.iloc[0].to_dict(), "exact"
    match = MFI_DF[MFI_DF["_name_key"].str.endswith(last_name.lower().strip())]
    if not match.empty:
        return match.iloc[0].to_dict(), "partial"
    return None, None


def infer_from_salary(salary: float) -> dict:
    pcts = META["salary_percentiles"]
    rank = sum(salary > v for v in pcts.values()) / len(pcts)
    return {
        "Employment Sector":               "Unknown",
        "Job Title":                       "Unknown",
        "Total Previous Loans":            max(1, int(3 - rank)),
        "Active Loans":                    max(0, int(2 - rank * 2)),
        "Total Outstanding Balance (USD)": round(salary * max(0.2, 1.4 - rank * 1.2), 0),
        "Avg Loan Amount (USD)":           round(salary * max(0.3, 0.9 - rank * 0.5), 0),
        "Common Loan Reason":              "Emergency",
        "Historical Return Rate (%)":      round(min(100, 60 + rank * 42), 1),
        "Days Past Due (Max)":             max(0, int((1 - rank) * 90)),
        "MFI Diversity Score":             max(1, int(3 - rank * 2)),
    }


# --------------------------------------------------------------
#  Alert Dispatch  (fire-and-forget via asyncio.create_task)
# --------------------------------------------------------------
async def _dispatch_alert_channels(name: str, message: str, severity: str, channel: str):
    """
    Simulates multi-channel notification (SMS, Email, Dashboard).
    Each channel is awaited concurrently with asyncio.gather.
    In production replace asyncio.sleep with real aiohttp / SMTP calls.
    """
    async def _send(ch: str):
        await asyncio.sleep(0)  # yield; replace with real async I/O
        log.info(f"[{ch}] {severity} -> {name}: {message[:80]}")

    await asyncio.gather(*[_send(ch.strip()) for ch in channel.split(",")])


async def create_alert(
    session: AsyncSession,
    borrower_id: str, name: str,
    alert_type: str, message: str,
    severity: str, channel: str = "Dashboard",
):
    """
    Writes the alert row to MySQL, then schedules channel dispatch as a
    background task - the HTTP response is NOT held up waiting for it.
    """
    session.add(Alert(
        borrower_id   = borrower_id,
        borrower_name = name,
        alert_type    = alert_type,
        message       = message,
        severity      = severity,
        channel       = channel,
        timestamp     = datetime.datetime.now(datetime.timezone.utc),
    ))
    # Fire-and-forget: returns immediately, task runs in the background
    asyncio.create_task(
        _dispatch_alert_channels(name, message, severity, channel)
    )


# --------------------------------------------------------------
#  Seed Demo Data  (fully async - parallel scoring + inserts)
# --------------------------------------------------------------
async def seed_data():
    async with AsyncSessionFactory() as session:
        result = await session.execute(select(func.count(Borrower.id)))
        if result.scalar():
            return  # already seeded

    sample = MFI_DF.sample(10, random_state=99)
    now    = datetime.datetime.now(datetime.timezone.utc)

    async def _insert_one(i: int, row: pd.Series):
        async with AsyncSessionFactory() as session:
            parts = row["Full Name"].split()
            first = parts[0];  last = " ".join(parts[1:])
            sal   = float(row["Current Monthly Salary (USD)"])
            sec   = row["Employment Sector"];   rea = row["Common Loan Reason"]
            tr    = float(row["Total Previous Loans"]);  al = float(row["Active Loans"])
            ob    = float(row["Total Outstanding Balance (USD)"])
            am    = float(row["Avg Loan Amount (USD)"])
            rr    = float(row["Historical Return Rate (%)"])
            dp    = float(row["Days Past Due (Max)"])
            ms    = float(row["MFI Diversity Score"])

            # Scoring runs in the thread pool even during seeding
            score, label, probs = await score_borrower_async(
                sal, sec, rea, tr, al, ob, am, rr, dp, ms
            )
            bid = f"S{i+1:03d}"
            session.add(Borrower(
                id=bid, full_name=row["Full Name"], first_name=first, last_name=last,
                salary=sal, employment_sector=sec, job_title=row["Job Title"],
                total_prev_loans=tr, active_loans=al, outstanding_balance=ob,
                avg_loan_amount=am, common_loan_reason=rea, return_rate=rr,
                days_past_due=dp, mfi_diversity_score=ms,
                risk_score=score, risk_label=label,
                risk_probability_high=probs.get("High",   0),
                risk_probability_medium=probs.get("Medium", 0),
                risk_probability_low=probs.get("Low",    0),
                data_source="mfi_exact", created_at=now,
            ))
            for day in range(8, 0, -1):
                ts = now - datetime.timedelta(days=day)
                v  = max(0.0, min(100.0, score + random.uniform(-10, 10)))
                session.add(Transaction(
                    borrower_id=bid, type="history", amount=0.0,
                    description="Historical record", is_anomaly=False,
                    anomaly_score=0.0, risk_score_after=round(v, 1),
                    risk_label_after=label, timestamp=ts,
                ))
            await session.commit()

    # All 10 borrowers scored + inserted concurrently (separate sessions)
    await asyncio.gather(*[
        _insert_one(i, row)
        for i, (_, row) in enumerate(sample.iterrows())
    ])
    log.info("Demo portfolio seeded")


# --------------------------------------------------------------
#  Routes
# --------------------------------------------------------------
@app.route("/")
async def index():
    return render_template("index.html", model_accuracy=round(META["accuracy"] * 100, 1))

@app.route("/dashboard")
async def dashboard():
    return render_template("dashboard.html")


@app.route("/api/borrowers")
async def get_borrowers():
    try:
        async with AsyncSessionFactory() as session:
            result    = await session.execute(
                select(Borrower).order_by(desc(Borrower.risk_score))
            )
            borrowers = result.scalars().all()
            return jsonify([b.to_dict() for b in borrowers])
    except Exception:
        log.exception("get_borrowers() DB error")
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/borrowers/<bid>")
async def get_borrower(bid):
    try:
        async with AsyncSessionFactory() as session:
            result   = await session.execute(select(Borrower).where(Borrower.id == bid))
            borrower = result.scalar_one_or_none()
            if not borrower:
                return jsonify({"error": "Not found"}), 404
            return jsonify(borrower.to_dict())
    except Exception:
        log.exception("get_borrower() DB error")
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/borrowers/<bid>/history")
async def risk_history(bid):
    try:
        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(
                    Transaction.timestamp,
                    Transaction.risk_score_after.label("risk_score"),
                    Transaction.type,
                )
                .where(Transaction.borrower_id == bid)
                .order_by(asc(Transaction.timestamp))
                .limit(30)
            )
            rows = result.all()
            return jsonify([
                {
                    "timestamp":  r.timestamp.isoformat(),
                    "risk_score": r.risk_score,
                    "type":       r.type,
                }
                for r in rows
            ])
    except Exception:
        log.exception("risk_history() DB error")
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/assess", methods=["POST"])
async def assess():
    data           = request.get_json(silent=True) or {}
    first_name     = data.get("first_name", "").strip()
    last_name      = data.get("last_name",  "").strip()
    payslip_salary = data.get("payslip_salary")
    match_type     = None

    if not data:
        return jsonify({"error": "Invalid JSON body."}), 400
    if payslip_salary:
        try:
            salary = _parse_float(payslip_salary, "payslip_salary", min_value=0.01)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        try:
            salary = _parse_float(data.get("salary"), "salary", min_value=0.01)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    if not first_name or not last_name:
        return jsonify({"error": "First name and last name are required."}), 400

    # -- In-memory MFI lookup (instant, no I/O) ---------------
    mfi_row, match_type = lookup_mfi(first_name, last_name)
    if mfi_row:
        sec = mfi_row.get("Employment Sector", "Unknown")
        rea = mfi_row.get("Common Loan Reason", "Emergency")
        tr  = float(mfi_row.get("Total Previous Loans", 0))
        al  = float(mfi_row.get("Active Loans", 0))
        ob  = float(mfi_row.get("Total Outstanding Balance (USD)", 0))
        am  = float(mfi_row.get("Avg Loan Amount (USD)", 0))
        rr  = float(mfi_row.get("Historical Return Rate (%)", 100))
        dp  = float(mfi_row.get("Days Past Due (Max)", 0))
        ms  = float(mfi_row.get("MFI Diversity Score", 1))
        job = mfi_row.get("Job Title", "Unknown")
        data_source = f"mfi_{match_type}"
    else:
        inf = infer_from_salary(salary)
        sec, rea = inf["Employment Sector"], inf["Common Loan Reason"]
        tr,  al  = inf["Total Previous Loans"], inf["Active Loans"]
        ob,  am  = inf["Total Outstanding Balance (USD)"], inf["Avg Loan Amount (USD)"]
        rr,  dp  = inf["Historical Return Rate (%)"], inf["Days Past Due (Max)"]
        ms       = inf["MFI Diversity Score"]
        job      = "Unknown"
        data_source = "salary_inference"
        match_type = "salary_inference"

    # -- ML scoring (CPU-bound -> thread pool, non-blocking) ---
    score, label, probs = await score_borrower_async(
        salary, sec, rea, tr, al, ob, am, rr, dp, ms
    )

    full_name = f"{first_name} {last_name}"
    now       = datetime.datetime.now(datetime.timezone.utc)
    frequent_window_start = now - datetime.timedelta(days=FREQUENT_APPLICATION_WINDOW_DAYS)

    # -- Persist to MySQL (async DB I/O) ----------------------
    try:
        async with AsyncSessionFactory() as session:
            result   = await session.execute(
                select(Borrower).where(
                    func.lower(Borrower.full_name) == full_name.lower()
                )
            )
            existing = result.scalar_one_or_none()
            recent_assessment_count = 0

            if existing:
                bid = existing.id
                result_recent = await session.execute(
                    select(func.count(Transaction.id)).where(
                        Transaction.borrower_id == bid,
                        Transaction.type == "assessment",
                        Transaction.timestamp >= frequent_window_start,
                    )
                )
                recent_assessment_count = int(result_recent.scalar() or 0)
                await session.execute(
                    update(Borrower)
                    .where(Borrower.id == bid)
                    .values(
                        salary                  = salary,
                        risk_score              = score,
                        risk_label              = label,
                        risk_probability_high   = probs.get("High",   0),
                        risk_probability_medium = probs.get("Medium", 0),
                        risk_probability_low    = probs.get("Low",    0),
                        data_source             = data_source,
                    )
                )
            else:
                bid = "B" + uuid.uuid4().hex[:6].upper()
                session.add(Borrower(
                    id=bid, full_name=full_name, first_name=first_name, last_name=last_name,
                    salary=salary, employment_sector=sec, job_title=job,
                    total_prev_loans=tr, active_loans=al, outstanding_balance=ob,
                    avg_loan_amount=am, common_loan_reason=rea, return_rate=rr,
                    days_past_due=dp, mfi_diversity_score=ms,
                    risk_score=score, risk_label=label,
                    risk_probability_high=probs.get("High",   0),
                    risk_probability_medium=probs.get("Medium", 0),
                    risk_probability_low=probs.get("Low",    0),
                    data_source=data_source, created_at=now,
                ))

            anomaly_eval = evaluate_application_anomalies(
                salary=salary,
                total_loans=tr,
                active_loans=al,
                outstanding=ob,
                return_rate=rr,
                days_due=dp,
                is_existing_borrower=bool(existing),
                recent_application_count=recent_assessment_count + 1,
            )

            anomaly_codes = ", ".join(a["code"] for a in anomaly_eval["anomalies"]) if anomaly_eval["anomalies"] else "none"
            session.add(Transaction(
                borrower_id=bid, type="assessment", amount=salary,
                description=f"Assessment [{data_source}] anomalies: {anomaly_codes}",
                is_anomaly=anomaly_eval["is_anomaly"], anomaly_score=anomaly_eval["anomaly_score"],
                risk_score_after=score, risk_label_after=label, timestamp=now,
            ))

            # Alert row is written here; channel dispatch is fire-and-forget
            if label == "High":
                msg = (f"{full_name} assessed as HIGH RISK "
                       f"(score {score:.1f}/100). Immediate review recommended.")
                await create_alert(session, bid, full_name, "HIGH_RISK",
                                   msg, "CRITICAL", "SMS,Email,Dashboard")
            elif label == "Medium":
                msg = (f"{full_name} scored MEDIUM RISK "
                       f"({score:.1f}/100). Enhanced monitoring advised.")
                await create_alert(session, bid, full_name, "MEDIUM_RISK",
                                   msg, "HIGH", "Dashboard")

            if anomaly_eval["is_anomaly"]:
                anomaly_msg = (
                    f"{full_name} triggered {len(anomaly_eval['anomalies'])} anomaly checks "
                    f"(score {anomaly_eval['anomaly_score']:.1f}/100): {anomaly_codes}."
                )
                severity = "CRITICAL" if anomaly_eval["anomaly_score"] >= 40 else "HIGH"
                await create_alert(session, bid, full_name, "ANOMALY_DETECTED",
                                   anomaly_msg, severity, "Dashboard")

            await session.commit()

    except Exception as exc:
        log.exception("assess() DB error")
        return jsonify({"error": "Database unavailable"}), 503

    return jsonify({
        "success":       True,
        "borrower_id":   bid,
        "full_name":     full_name,
        "salary":        salary,
        "risk_score":    score,
        "risk_label":    label,
        "probabilities": {k: round(v * 100, 1) for k, v in probs.items()},
        "data_source":   data_source,
        "match_type":    match_type,
        "anomaly_detection": anomaly_eval,
        "mfi_details": {
            "employment_sector": sec,
            "job_title":         job,
            "total_prev_loans":  tr,
            "active_loans":      al,
            "outstanding":       ob,
            "return_rate":       rr,
            "days_past_due":     dp,
        },
    })


@app.route("/api/parse-payslip", methods=["POST"])
async def parse_payslip():
    data     = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400
    text     = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"success": False, "error": "Text is required."}), 400
    patterns = [
        r"(?:net\s*(?:pay|salary)|take[\s-]?home|gross\s*(?:pay|salary))[^\d]*(\d[\d,\.]+)",
        r"(?:salary|wage|pay)[^\d]{0,20}(\d[\d,\.]+)",
        r"\$\s*(\d[\d,\.]+)",
        r"(\d[\d,\.]+)\s*(?:USD|usd)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if 50 < val < 100000:
                    return jsonify({"success": True, "salary": val})
            except ValueError:
                pass
    return jsonify({"success": False, "error": "Could not extract salary."}), 422


@app.route("/api/alerts")
async def get_alerts():
    try:
        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(Alert).order_by(desc(Alert.timestamp)).limit(50)
            )
            alerts = result.scalars().all()
            return jsonify([a.to_dict() for a in alerts])
    except Exception:
        log.exception("get_alerts() DB error")
        return jsonify([]), 200


@app.route("/api/transactions")
async def get_transactions():
    try:
        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(Transaction, Borrower)
                .join(Borrower, Borrower.id == Transaction.borrower_id, isouter=True)
                .order_by(desc(Transaction.timestamp))
                .limit(50)
            )
            rows = result.all()
            payload = []
            for tx, borrower in rows:
                client = borrower.full_name if borrower else tx.borrower_id
                payload.append({
                    "id": tx.id,
                    "client": client,
                    "amount": float(tx.amount or 0.0),
                    "risk": max(0.0, min(1.0, float(tx.risk_score_after or 0.0) / 100.0)),
                    "status": "Active",
                    "date": tx.timestamp.isoformat(),
                })
            return jsonify(payload)
    except Exception as exc:
        log.exception("get_transactions() error")
        return jsonify([]), 200


@app.route("/api/alerts/mark-read", methods=["POST"])
async def mark_read():
    try:
        async with AsyncSessionFactory() as session:
            await session.execute(
                update(Alert).where(Alert.is_read == False).values(is_read=True)
            )
            await session.commit()
            return jsonify({"success": True})
    except Exception:
        log.exception("mark_read() DB error")
        return jsonify({"error": "Database unavailable"}), 503


@app.route("/api/stats")
async def get_stats():
    """
    All 8 aggregation queries run CONCURRENTLY via asyncio.gather.
    Each coroutine opens its own session so they execute in parallel
    across separate pooled connections - total time ~ slowest single query
    instead of sum of all queries.
    """
    async def _count_all():
        async with AsyncSessionFactory() as s:
            r = await s.execute(select(func.count(Borrower.id)))
            return r.scalar() or 0

    async def _count_label(label: str):
        async with AsyncSessionFactory() as s:
            r = await s.execute(
                select(func.count(Borrower.id)).where(Borrower.risk_label == label)
            )
            return r.scalar() or 0

    async def _count_unread():
        async with AsyncSessionFactory() as s:
            r = await s.execute(
                select(func.count(Alert.id)).where(Alert.is_read == False)
            )
            return r.scalar() or 0

    async def _count_assessments():
        async with AsyncSessionFactory() as s:
            r = await s.execute(
                select(func.count(Transaction.id)).where(Transaction.type == "assessment")
            )
            return r.scalar() or 0

    async def _avg_risk():
        async with AsyncSessionFactory() as s:
            r = await s.execute(select(func.avg(Borrower.risk_score)))
            return float(r.scalar() or 0.0)

    async def _total_salary():
        async with AsyncSessionFactory() as s:
            r = await s.execute(select(func.sum(Borrower.salary)))
            return float(r.scalar() or 0.0)

    # Fire all 8 queries simultaneously
    try:
        (total, high, medium, low,
         unread, assessments,
         avg_r, tot_sal) = await asyncio.gather(
            _count_all(),
            _count_label("High"),
            _count_label("Medium"),
            _count_label("Low"),
            _count_unread(),
            _count_assessments(),
            _avg_risk(),
            _total_salary(),
        )
    except Exception:
        log.exception("get_stats() DB error")
        return jsonify({"error": "Database unavailable"}), 503

    return jsonify({
        "total_borrowers":        total,
        "high":                   high,
        "medium":                 medium,
        "low":                    low,
        "unread_alerts":          unread,
        "total_assessments":      assessments,
        "avg_risk_score":         round(avg_r,   1),
        "total_salary_portfolio": round(tot_sal, 2),
        "model_accuracy":         round(META["accuracy"] * 100, 1),
    })


@app.route("/api/portfolio-risk-trend")
async def risk_trend():
    try:
        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(
                    func.date(Transaction.timestamp).label("day"),
                    func.avg(Transaction.risk_score_after).label("avg_risk"),
                )
                .group_by(func.date(Transaction.timestamp))
                .order_by(asc(func.date(Transaction.timestamp)))
                .limit(14)
            )
            rows = result.all()
            return jsonify([
                {"day": str(r.day), "avg_risk": round(float(r.avg_risk), 2)}
                for r in rows
            ])
    except Exception:
        log.exception("risk_trend() DB error")
        return jsonify({"error": "Database unavailable"}), 503


# --------------------------------------------------------------
#  Startup
# --------------------------------------------------------------
if __name__ == "__main__":
    log.info("Connecting to MySQL (async)...")
    asyncio.run(ensure_db_ready())

    log.info("Starting FinanceGuard v2 -> http://127.0.0.1:5000")
    # Flask 2.0+ runs async routes natively via Werkzeug.
    # For production, use Hypercorn (ASGI) for full async concurrency:
    #   pip install hypercorn
    #   hypercorn app:app --bind 0.0.0.0:5000 --workers 4
    app.run(debug=False, host="0.0.0.0", port=5000)
