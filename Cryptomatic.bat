@echo off
:start
echo Starting Cryptomatic...
python app.py
echo Application exited with code %ERRORLEVEL%
if %ERRORLEVEL% == 42 (
    echo Restarting after update...
    goto start
)
pause