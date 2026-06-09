@echo off
cd /d "C:\code\krylov"
set MPLBACKEND=Agg

:: Print Start Header
echo. >> "C:\code\krylov\thesis_log.txt"
echo ============================================================ >> "C:\code\krylov\thesis_log.txt"
echo [START] Schatten Norm Script Triggered at: %date% %time% >> "C:\code\krylov\thesis_log.txt"
echo ============================================================ >> "C:\code\krylov\thesis_log.txt"

:: Run the Python script (unbuffered)
"C:\code\krylov\.venv\Scripts\python.exe" -u "C:\code\krylov\schatten_norms.py" >> "C:\code\krylov\thesis_log.txt" 2>&1

:: Print End Header
echo ============================================================ >> "C:\code\krylov\thesis_log.txt"
echo [END] Schatten Norm Script Finished at: %date% %time% >> "C:\code\krylov\thesis_log.txt"
echo ============================================================ >> "C:\code\krylov\thesis_log.txt"
echo. >> "C:\code\krylov\thesis_log.txt"