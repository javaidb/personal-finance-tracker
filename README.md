# Personal Finance Tracker

<img src="https://github.com/user-attachments/assets/b9d234cc-f426-4833-b7fd-93ae682f0176" alt="Scotiabank Logo" width="200">

This repository provides tools for analyzing and visualizing Scotiabank financial data. It extracts data from bank statement PDFs, categorizes transactions, and produces insights on financial health and spending habits. The project now offers both a web interface and Jupyter notebook functionality.

***Note**: Parameters are tuned for bank statement PDFs generated as of December 2024, any changes to the statement's file standards from Scotiabank after this point may/may not be reflected here.*

## Features

- PDF bank statement processing and data extraction
- Transaction categorization and analysis
- Financial data visualization
- Interactive web dashboard
- Balance charts and spending trends
- Category-based spending analysis
- Jupyter notebook interface for custom analysis

## Repository Structure

This project is organized into several directories:
```
    personal-finance-tracker/
    ├── bank_statements/                   
    │   └── Chequing                       # Folder holding accounts/PDFs belong to chequing accounts
    │   │   └── ...                        # <Generic account name>
    │   │   │   └── ...                    # <Generic statement PDFs>
    │   └── Savings                        # Folder holding accounts/PDFs belong to savings accounts
    │   └── Credit                         # Folder holding accounts/PDFs belong to credit accounts
    ├── cached_data/                       
    │   └── databank.json                  # Bank of personalized associations to categorize by
    ├── src/                               
    │   └── modules/                       
    │   │   └── pdf_interpreter.py         # Module for interpreting bank statement PDFs
    │   │   └── helper_fns.py              # Module for helper fns to use throughout build
    │   └── notebooks/                     
    │   │   └── finance_visualizer.py      # Module to visualize finance insights
    │   └── web/                           
    │       └── app.py                     # Flask web application
    │       └── static/                    # Static web assets
    │       └── templates/                 # HTML templates
    ├── README.md                          # Project overview and documentation
    └── requirements.txt                   # List of dependencies required to run the project
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation
1. Clone this repository to your local machine
2. Create and activate a virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make a copy of config within *src/config* and rename to *config.py*
   - Add general estimates of rent paid for property (See config_template for instructions/formatting)

### Setup
- Within bank_statements/..., there are 3 folders called 'Chequing', 'Savings' and 'Credit'
- Download bank statements from ScotiaOnline and place them in the appropriate folders
  - E.g. Statements from 'Student Banking' would go in *bank_statements/Chequing/Student Banking/...*
  - See README per folder for more information

## Usage Options

### Option 1: Web Interface

1. Run the web application:
   ```
   # Either double-click the run_webapp.bat file
   # Or run manually:
   cd src\web
   python app.py
   ```
2. Open a web browser and go to: `http://127.0.0.1:5000/`
3. Upload bank statements from the home page
4. View your financial dashboard and analyze your spending

### Option 2: Jupyter Notebook

Navigate to the finance-tracker notebook and hit 'run all' to process data and generate visualizations.

## Contributors and Dependencies
<p align="center">
  <img src="https://github.com/user-attachments/assets/b9d234cc-f426-4833-b7fd-93ae682f0176" alt="Scotiabank Logo" width="200" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/fd9c0f77-868c-4fdb-a53d-ba6344781b4e" width="120" />
</p>

## Author

- [@javaidb](https://www.github.com/javaidb)
