import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'financeGard'))

def test_basic_functionality():
    '''Test basic functionality'''
    assert True  # Replace with actual tests

class TestFinancegard:
    '''Test class for financeGard'''
    
    def test_app_creation(self):
        '''Test application creation'''
        # Add your app-specific tests here
        pass
    
    def test_health_endpoint(self):
        '''Test health endpoint'''
        # Add health endpoint test
        pass
