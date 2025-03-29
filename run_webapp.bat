@echo off
echo ======================================================
echo Personal Finance Tracker Web Application
echo ======================================================
echo.
echo This web app will process bank statements from your folders.
echo Bank statements should be placed in:
echo bank_statements/[account_type]/[account_name]/statements.pdf
echo.
echo Example:
echo bank_statements/Chequing/MyAccount/January2023.pdf
echo bank_statements/Credit/VisaCard/Feb2023.pdf
echo.
echo Activating virtual environment...
call .venv_finance\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Starting web application...
cd src\web
python app.py

pause 