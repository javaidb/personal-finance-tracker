from src.modules.pdf_interpreter import PDFReader
import pandas as pd
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't wrap wide columns
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Format floats to 2 decimal places

def inspect_dataframe():
    # Initialize reader and get DataFrame
    reader = PDFReader()
    df = reader.process_raw_df()
    
    # Sort by DateTime
    df = df.sort_values('DateTime')
    
    # Create running balance columns for each account type
    print("\n=== Complete Transaction History with Running Balances ===")
    
    # Show summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total number of transactions: {len(df)}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

    # Display the last 50 transactions with both account_balance and running_balance
    print("\n=== Last 50 Transactions ===")
    print(df[['DateTime', 'Account Type', 'Account Name', 'account_balance', 'running_balance', 'Amount']].tail(50))

if __name__ == "__main__":
    inspect_dataframe() 