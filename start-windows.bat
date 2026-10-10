@echo off
REM Feelable - one-click start for Windows. Double-click this file.
cd /d "%~dp0"
title Feelable

where py >nul 2>nul
if %errorlevel%==0 (set PY=py) else (set PY=python)
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python is not installed. Get it from https://www.python.org/downloads/
  echo During setup TICK "Add python.exe to PATH", then double-click this file again.
  pause
  exit /b 1
)

if not exist "backend\.deps_ok" (
  echo Installing Python packages - first time only, a few minutes...
  %PY% -m pip install -r backend\requirements.txt
  if errorlevel 1 ( echo Python package install failed - see the message above. & pause & exit /b 1 )
  echo ok> "backend\.deps_ok"
)

if not exist "web\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Node.js is not installed. Get the LTS version from https://nodejs.org/ then double-click this file again.
    pause
    exit /b 1
  )
  echo Building the web app - first time only...
  pushd web
  call npm install
  call npm run build
  popd
  if not exist "web\dist\index.html" ( echo Web build failed - see the message above. & pause & exit /b 1 )
)

echo.
echo  Feelable is starting...  Open  http://localhost:8000  in Chrome.
echo  Keep this window open while you use it. Close it to stop.
echo.
start "" http://localhost:8000
cd backend
%PY% -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
