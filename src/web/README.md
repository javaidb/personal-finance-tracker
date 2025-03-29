# Personal Finance Tracker Web Application

This web application provides a browser-based interface for the Personal Finance Tracker. It allows you to upload and process bank statements, view transactions, categorize spending, and visualize financial data.

## Features

- Upload and process PDF bank statements
- Interactive dashboard with financial data visualization
- Transaction listing and categorization
- Balance charts and spending trends
- Category-based spending analysis

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- PDF bank statements from Scotiabank

### Installation

1. Navigate to the project root directory
2. Activate your virtual environment:
   ```
   .venv_finance\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Web Application

#### Option 1: Run the batch file
Simply double-click the `run_webapp.bat` file in the project root directory.

#### Option 2: Run manually
1. Activate your virtual environment
2. Navigate to the web directory:
   ```
   cd src\web
   ```
3. Run the Flask application:
   ```
   python app.py
   ```
4. Open a web browser and go to: `http://127.0.0.1:5000/`

## Usage

1. Upload your bank statements from the home page
2. Wait for processing to complete
3. View your financial dashboard
4. Explore transactions and spending categories
5. Analyze balance trends over time

## Extending

- Add new chart types in dashboard.html
- Extend API endpoints in app.py
- Create new visualization pages 