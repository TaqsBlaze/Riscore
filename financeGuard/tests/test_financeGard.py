import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'financeGard'))

from financeGuard.api.endpoints import (
    evaluate_application_anomalies,
    _parse_float,
    FREQUENT_APPLICATION_THRESHOLD,
)

def test_high_requested_amount_triggers_anomaly():
    anomalies = evaluate_application_anomalies(
        salary=1000,
        total_loans=2,
        active_loans=1,
        outstanding=500,
        return_rate=90,
        days_due=0,
        is_existing_borrower=False,
        recent_application_count=1,
        loan_amount=4000,
    )
    codes = {entry['code'] for entry in anomalies['anomalies']}
    assert 'HIGH_REQUESTED_AMOUNT' in codes
    assert anomalies['is_anomaly']

def test_low_requested_amount_skips_amount_anomaly():
    anomalies = evaluate_application_anomalies(
        salary=1200,
        total_loans=0,
        active_loans=0,
        outstanding=0,
        return_rate=100,
        days_due=0,
        is_existing_borrower=False,
        recent_application_count=1,
        loan_amount=300,
    )
    codes = {entry['code'] for entry in anomalies['anomalies']}
    assert 'HIGH_REQUESTED_AMOUNT' not in codes

def test_parse_float_requires_positive_amount():
    assert _parse_float('2500', 'amount', min_value=0.01) == 2500.0
    with pytest.raises(ValueError):
        _parse_float('-1', 'amount', min_value=0.01)

def test_outstanding_active_loan_anomaly_triggers():
    anomalies = evaluate_application_anomalies(
        salary=1500,
        total_loans=1,
        active_loans=1,
        outstanding=1200,
        return_rate=95,
        days_due=0,
        is_existing_borrower=True,
        recent_application_count=1,
        loan_amount=1000,
    )
    codes = {entry['code'] for entry in anomalies['anomalies']}
    assert 'OUTSTANDING_ACTIVE_LOAN' in codes
    assert anomalies['is_anomaly']

def test_frequent_applications_anomaly_triggered():
    anomalies = evaluate_application_anomalies(
        salary=2000,
        total_loans=2,
        active_loans=0,
        outstanding=0,
        return_rate=100,
        days_due=0,
        is_existing_borrower=True,
        recent_application_count=FREQUENT_APPLICATION_THRESHOLD,
        loan_amount=500,
    )
    codes = {entry['code'] for entry in anomalies['anomalies']}
    assert 'FREQUENT_LOAN_APPLICATIONS' in codes

def test_high_debt_to_income_anomaly_triggered():
    anomalies = evaluate_application_anomalies(
        salary=800,
        total_loans=0,
        active_loans=0,
        outstanding=2000,
        return_rate=90,
        days_due=0,
        is_existing_borrower=False,
        recent_application_count=1,
        loan_amount=400,
    )
    codes = {entry['code'] for entry in anomalies['anomalies']}
    assert 'HIGH_DEBT_TO_INCOME' in codes
