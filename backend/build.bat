@REM @echo off
REM Step 1: Run pyinstaller with app.spec
@REM pyinstaller app_splash.spec -y
@REM cd ..\thekode
@REM npm run build

@REM cd ..\backend
@REM .\.venv\Scripts\activate

pyinstaller app.spec -y --clean

REM Step 2: Create directory structure
mkdir dist\app\_internal\coreforecast\lib\Release\

REM Step 3: Copy libcoreforecast.dll to the specified directory
@REM copy backend/venv/Lib/site-packages/coreforecast/lib/Release/libcoreforecast.dll dist\app\_internal\coreforecast\lib\Release\
copy .\libcoreforecast.dll dist\app\_internal\coreforecast\lib\Release\
copy .\server_details.json dist\app\server_details.json
copy .\algos.json dist\app\algos.json
copy .\seasonality.json dist\app\seasonality.json
xcopy ..\thekode\out dist\app\public\out /E /I
