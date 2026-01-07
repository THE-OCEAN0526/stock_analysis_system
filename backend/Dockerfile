# 使用輕量級 Python 3.12
FROM python:3.12-slim

# 設定容器內的工作目錄
WORKDIR /app

# 先複製套件清單並安裝，利用 Docker layer cache 節省時間
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有後端代碼到容器
COPY . .

# 暴露 FastAPI 的 Port
EXPOSE 8000

# 啟動命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
