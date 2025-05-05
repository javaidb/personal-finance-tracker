from datetime import datetime
import pandas as pd

class Transaction:
    """Transaction model for representing financial transactions."""
    
    def __init__(self, date_time: datetime, amount: float, details: str, 
                 classification: str = None, account_name: str = None, balance: float = None):
        self.date_time = date_time
        self.amount = amount
        self.details = details
        self.classification = classification
        self.account_name = account_name
        self.balance = balance

    @classmethod
    def from_dataframe_row(cls, row):
        """Create a Transaction instance from a pandas DataFrame row."""
        return cls(
            date_time=pd.to_datetime(row['DateTime']),
            amount=float(row['Amount']),
            details=str(row['Details']),
            classification=row.get('Classification'),
            account_name=row.get('Account_Name'),
            balance=float(row['Balance']) if 'Balance' in row and pd.notna(row['Balance']) else None
        )

    def to_dict(self):
        """Convert transaction to dictionary."""
        return {
            'DateTime': self.date_time.strftime('%Y-%m-%d'),
            'Amount': self.amount,
            'Details': self.details,
            'Classification': self.classification,
            'Account_Name': self.account_name,
            'Balance': self.balance
        } 