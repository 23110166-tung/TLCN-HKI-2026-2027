@echo off
chcp 65001 >nul
title HCMUTE DocAI - Hệ thống Thẩm định Hồ sơ Học vụ Sinh viên
echo ===============================================================================
echo        TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH (HCMUTE)
echo                 TIỂU LUẬN CHUYÊN NGÀNH TRÍ TUỆ NHÂN TẠO
echo        HCMUTE DocAI: Nhận dạng, Trích xuất và Thẩm định Biểu mẫu Học vụ
echo ===============================================================================
echo.

:: 1. Kiểm tra môi trường Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LỖI] Chưa phát hiện Python trên hệ thống!
    echo Vui lòng cài đặt Python (phiên bản 3.10 hoặc 3.11) và tích chọn 'Add Python to PATH'.
    pause
    exit /b
)

:: 2. Kiểm tra hoặc tạo môi trường ảo (Virtual Environment)
if not exist "venv" (
    echo [1/3] Đang khởi tạo môi trường ảo venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [LỖI] Không thể tạo môi trường ảo venv.
        pause
        exit /b
    )
    echo [2/3] Đang kích hoạt venv và cài đặt thư viện từ requirements.txt...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo [1/2] Đang kích hoạt môi trường ảo venv...
    call venv\Scripts\activate.bat
)

:: 3. Khởi chạy Streamlit App
echo.
echo [3/3] Đang khởi động ứng dụng Web Streamlit trên localhost:8501...
echo Trình duyệt mặc định sẽ tự động mở sau vài giây...
echo.
streamlit run app.py --server.port=8501 --server.address=localhost

pause

