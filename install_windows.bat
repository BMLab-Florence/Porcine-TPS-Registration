@echo off
setlocal
cd /d "%~dp0"
echo ===========================================================
echo  Porcine Tissue TPS Registration - initial installation
echo ===========================================================
echo.
echo Checking for Python 3.12 (64-bit recommended)...
py -3.12 --version
if errorlevel 1 (
  echo.
  echo Python 3.12 was not found.
  echo Install Python 3.12 from https://www.python.org/downloads/
  echo or use the manual instructions in README.md for Python 3.10-3.13.
  pause
  exit /b 1
)
echo.
echo Creating the isolated environment...
py -3.12 -m venv .venv
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto :fail
echo.
echo SUCCESS: installation and core tests completed.
echo Double-click start_windows.bat to open the application.
pause
exit /b 0
:fail
echo.
echo Installation or tests failed. Read the error above and consult README.md.
pause
exit /b 1
