# FinanceGuard Diagrams

This document captures the current system structure based on the active codebase.

## 1) Context Level Diagram

```mermaid
flowchart LR
    U[Borrower / Applicant]
    R[Risk Officer / Admin]
    W[FinanceGuard Web App]
    A[Flask API]
    M[ML Risk Model]
    DB[(MySQL Database)]
    D[(MFI Data CSV)]
    S[(Uploaded Files / Payslip / National ID)]
    N[Notifications / Alerts]

    U -->|Submits application| W
    U -->|Tracks status| W
    R -->|Reviews dashboard| W
    W -->|HTTP requests| A
    A -->|Scores application| M
    A -->|Reads reference data| D
    A -->|Stores borrowers, transactions, alerts| DB
    A -->|Processes uploads| S
    A -->|Sends alerts| N
    W <-->|JSON / HTML responses| A
```

## 2) Activity Diagram

```mermaid
flowchart TD
    Start([Start])
    Open[Applicant opens application page]
    Fill[Enter full name, amount, reason]
    UploadPayslip[Upload payslip]
    OCRPayslip[Extract salary and payslip name]
    UploadID[Upload National ID]
    OCROID[Extract ID name with OCR]
    MatchName{Do names match?}
    Submit[Submit application payload]
    Validate[Validate required fields]
    LoadAssets[Load model artefacts and MFI CSV]
    Lookup[Lookup borrower in MFI data]
    Score[Run risk scoring model]
    Anomaly[Evaluate anomalies]
    BuildFeedback[Build area_feedback]
    Decide["Rundecide_application()"]
    Persist[Save Borrower + Transaction + Alert rows]
    Render[Return response to UI]
    Split{Decision status?}
    Approved[Approved path]
    Rejected[Rejected path]
    PassArea[Populate area_feedback.passed]
    FailArea[Populate area_feedback.failed]
    End([End])

    Start --> Open --> Fill --> UploadPayslip --> OCRPayslip --> UploadID --> OCROID --> MatchName
    MatchName -- No --> Rejected
    MatchName -- Yes --> Submit --> Validate --> LoadAssets --> Lookup --> Score --> Anomaly --> BuildFeedback --> Decide --> Persist --> Render --> Split
    Split -- approved --> Approved --> PassArea --> End
    Split -- rejected --> Rejected --> FailArea --> End

    Validate -->|Missing fields| Rejected
    Rejected --> FailArea
    Approved --> PassArea
```

### Pass / Fail categorization used by the current system

- `area_feedback.failed` is populated from anomaly checks plus a `High` risk label.
- `area_feedback.passed` is populated from every defined area that did not fail.
- The UI renders rejected feedback in the "Areas that need attention" section and approved feedback in the "Areas cleared" section.
- The final API response also returns `decision_feedback.failed` and `decision_feedback.passed`.

## 3) Data Flow Diagram

```mermaid
flowchart LR
    Applicant[Applicant]
    Officer[Risk Officer]
    UI[Web UI Pages]
    API[Flask Routes]
    OCR[OCR / Document Parsing]
    ML[Risk Scoring Model]
    DAO[Decision + Feedback Builder]
    DB[(MySQL)]
    CSV[(data.csv / MFI dataset)]

    Applicant -->|Form data, payslip, National ID| UI
    Officer -->|Dashboard, risk score, tracking| UI
    UI -->|POST /api/assess| API
    UI -->|GET /api/application-status/<tracking_number>| API
    UI -->|GET /api/transactions, /api/alerts, /api/stats| API

    API --> OCR
    OCR -->|Salary, payslip name, ID name| API
    API --> CSV
    API --> ML
    ML -->|risk_score, risk_label, probabilities| API
    API --> DAO
    DAO -->|decision_status, decision_reason, area_feedback| API
    API --> DB
    DB -->|Borrower, Transaction, Alert, BlacklistedUser| API
    API -->|JSON response| UI
```

## 4) ERD Diagram

```mermaid
erDiagram
    BORROWERS {
        string id PK
        string full_name
        string first_name
        string last_name
        float salary
        float loan_amount
        string employment_sector
        string job_title
        float total_prev_loans
        float active_loans
        float outstanding_balance
        float avg_loan_amount
        string common_loan_reason
        float return_rate
        float days_past_due
        float mfi_diversity_score
        float risk_score
        string risk_label
        float risk_probability_high
        float risk_probability_medium
        float risk_probability_low
        string data_source
        datetime created_at
    }

    TRANSACTIONS {
        int id PK
        string borrower_id FK
        string type
        float amount
        text description
        bool is_anomaly
        float anomaly_score
        float risk_score_after
        string risk_label_after
        string status
        string tracking_number
        datetime timestamp
    }

    ALERTS {
        int id PK
        string borrower_id FK
        string borrower_name
        string alert_type
        text message
        string severity
        string channel
        bool is_read
        datetime timestamp
    }

    USERS {
        string id PK
        string full_name
        string email
        string password_hash
        datetime created_at
    }

    BLACKLISTED_USER {
        int id PK
        string borrower_id
        string full_name
        text reason
        float credit_score
        datetime added_at
    }

    BORROWERS ||--o{ TRANSACTIONS : "has"
    BORROWERS ||--o{ ALERTS : "generates"
    BORROWERS ||--o{ BLACKLISTED_USER : "may be copied into"
```

## 5) Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Applicant
    participant UI as Application Form
    participant API as /api/assess
    participant OCR as OCR Parser
    participant ML as Risk Model
    participant DB as MySQL
    participant DASH as Risk Score Page / Tracking Page

    Applicant->>UI: Enter name, amount, reason
    Applicant->>UI: Upload payslip and National ID
    UI->>OCR: Extract salary, payslip name, ID name
    OCR-->>UI: salary + extracted names
    UI->>API: POST /api/assess with form data
    API->>API: Validate fields and normalize names
    API->>DB: Check existing borrower
    API->>ML: score_borrower_async(...)
    ML-->>API: risk_score, risk_label, probabilities
    API->>API: evaluate_application_anomalies(...)
    API->>API: build area_feedback (passed / failed)
    API->>API: decide_application(...)
    API->>DB: Insert Borrower and Transaction
    API->>DB: Insert Alert if needed
    API-->>UI: JSON response with decision_feedback
    UI->>DASH: Store last assessment in localStorage
    Applicant->>DASH: View risk score or tracking status
    DASH->>API: GET /api/application-status/<tracking_number>
    API->>DB: Load Transaction + Borrower
    DB-->>API: status, reason, risk values
    API-->>DASH: Status response
```

