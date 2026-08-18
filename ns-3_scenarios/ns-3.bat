@echo off
set SCRIPT_NAME=%~1

if "%SCRIPT_NAME%"=="" (
    echo Vui long nhap ten file kich ban! Vi du: run_ns3.bat cttc-nr-scenario-handover-storm
    exit /b
)

echo [1/3] Dang day file %SCRIPT_NAME%.cc sang WSL...
:: Thay "Ubuntu" bằng tên bản WSL của bạn nếu khác (kiểm tra bằng lệnh: wsl -l -v)
wsl -d ns-3 cp %SCRIPT_NAME%.cc /home/thentt/ns-3-dev/scratch/

echo [2/3] Dang bien dich va chay mo phong tren WSL...
wsl -d ns-3 -e bash -c "cd /home/thentt/ns-3-dev && ./ns3 run scratch/%SCRIPT_NAME%"

echo [3/3] Dang lay ket qua CSV ve Windows...
:: Copy tất cả các file .csv được tạo ra ở thư mục gốc ns-3 về thư mục hiện tại trên Windows
wsl -d ns-3 -e bash -c "cp /home/thentt/ns-3-dev/*.csv $(pwd)"

echo Hoan thanh!
