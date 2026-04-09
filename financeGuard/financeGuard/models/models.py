from financeGuard import db
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, func, select, update, desc, asc
)
from zoneinfo import ZoneInfo


time_zone = "Africa/Harare"


def now_local() -> datetime:
    return datetime.now(ZoneInfo(time_zone))


class Borrower(db.Model):
    __tablename__ = "borrowers"

    id                      = Column(String(20),  primary_key=True)
    full_name               = Column(String(120), nullable=False)
    first_name              = Column(String(60))
    last_name               = Column(String(60))
    salary                  = Column(Float,  default=0.0)
    loan_amount             = Column(Float,  default=0.0)
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
    created_at              = Column(DateTime, default=now_local)

    def to_dict(self):
        return {
            c.name: (
                getattr(self, c.name).isoformat()
                if isinstance(getattr(self, c.name), datetime)
                else getattr(self, c.name)
            )
            for c in self.__table__.columns
        }


class Transaction(db.Model):
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
    status           = Column(String(32),  default="processing")
    tracking_number  = Column(String(36), unique=True, index=True)
    deposit_channel  = Column(String(32))
    deposit_details  = Column(Text)
    deposit_updated_at = Column(DateTime)
    timestamp        = Column(DateTime, default=now_local)

    def to_dict(self):
        return {
            c.name: (
                getattr(self, c.name).isoformat()
                if isinstance(getattr(self, c.name), datetime)
                else getattr(self, c.name)
            )
            for c in self.__table__.columns
        }


class Alert(db.Model):
    __tablename__ = "alerts"

    id            = Column(Integer,    primary_key=True, autoincrement=True)
    borrower_id   = Column(String(20))
    borrower_name = Column(String(120))
    alert_type    = Column(String(60))
    message       = Column(Text)
    severity      = Column(String(20))
    channel       = Column(String(60),  default="Dashboard")
    is_read       = Column(Boolean,     default=False)
    timestamp     = Column(DateTime,    default=now_local)

    def to_dict(self):
        return {
            c.name: (
                getattr(self, c.name).isoformat()
                if isinstance(getattr(self, c.name), datetime)
                else getattr(self, c.name)
            )
            for c in self.__table__.columns
        }


class User(db.Model):
    __tablename__ = "users"

    id            = Column(String(36), primary_key=True)
    full_name     = Column(String(120), nullable=False)
    email         = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(200), nullable=False)
    created_at    = Column(DateTime, default=now_local)

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class BlacklistedUser(db.Model):
    __tablename__ = "blacklisted_user"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id  = Column(String(20))
    full_name    = Column(String(120))
    reason       = Column(Text)
    credit_score = Column(Float, default=0.0)
    added_at     = Column(DateTime, default=now_local)

    def to_dict(self):
        return {
            "id": self.id,
            "borrower_id": self.borrower_id,
            "full_name": self.full_name,
            "reason": self.reason,
            "credit_score": self.credit_score,
            "added_at": self.added_at.isoformat(),
        }
