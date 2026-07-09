# Sử dụng bản slim để image nhẹ, phù hợp chạy trên laptop hoặc cluster
FROM python:3.10-slim

# Thiết lập thư mục làm việc chính
WORKDIR /app

# 1. Copy file requirements từ thư mục deploy vào trước để cache layer
COPY services/agent/requirements.txt .

# 2. Cài đặt các thư viện (Sử dụng --index-url để lấy bản CPU cho nhẹ nếu không cần GPU)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Copy thư mục core (chứa env.py và model.py)
# Điều này cực kỳ quan trọng để lệnh 'from core.model import DRQN' hoạt động
COPY core/ ./core/

# 4. Copy model đã train vào thư mục models bên trong container
COPY models/ ./models/

# 5. Copy file main.py (API Service) vào root của WORKDIR
COPY services/agent/main.py .

# 6. Thiết lập biến môi trường để Python nhận diện thư mục /app là một package
ENV PYTHONPATH=/app

# Mở port cho FastAPI
EXPOSE 8000

# Chạy ứng dụng uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# force rebuild
