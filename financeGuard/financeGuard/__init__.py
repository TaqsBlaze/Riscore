from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
import os

try:
    from flask_mail import Mail
except ModuleNotFoundError:  # optional dependency for local dev
    Mail = None


app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://192.168.1.105",
            "http://localhost:5000",
            "http://127.0.0.1:5000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key")  # Change this in production

sync_database_url = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root@localhost/microfinance_db",
)
async_database_url = os.getenv(
    "ASYNC_DATABASE_URL",
    (
        sync_database_url.replace("mysql+pymysql://", "mysql+aiomysql://")
        if sync_database_url.startswith("mysql+pymysql://")
        else sync_database_url
    ),
)

app.config["SQLALCHEMY_DATABASE_URI"] = sync_database_url
app.config["ASYNC_DATABASE_URI"] = async_database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = "financeGuard/static/uploads"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = ('pdf', 'doc', 'docx')
app.config['TIMEZONE'] = 'Africa/Harare'
app.config['MAIL_SERVER'] = 'mail.my-domain.co.zw'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'noreply@my-domain.co.zw'
app.config['MAIL_PASSWORD'] = '********'
app.config['MAIL_DEFAULT_SENDER'] = ('Alert', 'noreply@my-domain.co.zw')
app.config['ADMIN_ALERT_EMAIL'] = os.getenv("ADMIN_ALERT_EMAIL", "")
app.config['MIMETYPE'] = {'.mjs':'application/javascript'}
app.config['MODEL_PATH'] = "financeGuard/model/risk_model.pkl"
app.config['LABEL_ENCODER_PATH'] = "financeGuard/model/label_encoder.pkl"
app.config['FEATURE_COLS_PATH'] = "financeGuard/model/feature_cols.pkl"
app.config['MODEL_DIR'] = "financeGuard/model"
app.config['DATA_DIR'] = "financeGuard/static/data"

mail = Mail(app) if Mail else None

app.app_context()
db = SQLAlchemy(app)
migrate = Migrate(app, db, render_as_batch = True)

from financeGuard.api import endpoints
