@echo off
echo.
echo ====================================================
echo    Personal Finance Tracker Web Application
echo ====================================================
echo.
echo Place your bank statements in the following directory structure:
echo bank_statements/[bank_name]/[account_type]/[account_name]/statements.pdf
echo.
echo Examples:
echo bank_statements/scotiabank/chequing/main_account/statement.pdf
echo bank_statements/scotiabank/credit/visa_card/statement.pdf
echo.
echo Creating basic required directories...
mkdir bank_statements 2>nul
mkdir cached_data 2>nul
mkdir logs 2>nul

echo.
echo Checking for existing bank folders...
dir /b /ad "bank_statements" >nul 2>&1
if errorlevel 1 (
    echo No bank folders found. Creating sample structure...
    mkdir bank_statements\sample_bank 2>nul
    mkdir bank_statements\sample_bank\chequing 2>nul
    mkdir bank_statements\sample_bank\credit 2>nul
    mkdir bank_statements\sample_bank\savings 2>nul
    echo Created sample structure: bank_statements\sample_bank\
    echo Please rename 'sample_bank' to your actual bank name.
) else (
    echo Found existing bank folders in bank_statements\
)

echo.
echo Creating sample databank.json file if it doesn't exist...
if not exist "cached_data\databank.json" (
  echo { "categories": { "Groceries": { "patterns": [], "totalMatches": 0 }, "Dining": { "patterns": [], "totalMatches": 0 }, "Transport": { "patterns": [], "totalMatches": 0 }, "Bills": { "patterns": [], "totalMatches": 0 }, "Shopping": { "patterns": [], "totalMatches": 0 }, "Entertainment": { "patterns": [], "totalMatches": 0 }, "Income": { "patterns": [], "totalMatches": 0 }, "Transfer": { "patterns": [], "totalMatches": 0 }, "uncategorized": { "patterns": [], "totalMatches": 0 } } } > cached_data\databank.json
)

echo.
echo Activating virtual environment...
if exist .venv\Scripts\activate (
  call .venv\Scripts\activate
) else (
  echo Virtual environment not found. Creating one...
  python -m venv .venv
  call .venv\Scripts\activate
)

:: Set port number
set PORT=8000

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Starting the web application...
echo.
echo Web app will be available at: http://127.0.0.1:%PORT%/
echo.

REM Start the web application and open the browser
start http://localhost:%PORT%/
python src/run.py

REM When Flask ends, deactivate the virtual environment
call .venv\Scripts\deactivate 