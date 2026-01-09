# 1. 使用輕量級 Python 3.12 映像檔
FROM python:3.12-slim

# 2. 設定容器內的工作目錄為 /app
WORKDIR /app

# 3. 設定環境變數，確保 Python 輸出直接印在終端機，方便查看 Docker 日誌
# 並將當前目錄加入 PYTHONPATH 確保模組引用正常
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 4. 安裝系統依賴 (如果你的套件需要編譯環境，例如 numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 先複製套件清單並安裝，利用 Docker layer cache 節省時間
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 複製後端所有程式碼 (包含 controllers, services, models 資料夾)
# 配合 .dockerignore 檔案會自動排除 __pycache__ 與 .env
COPY . .

# 7. 暴露 FastAPI 的預設 Port
EXPOSE 8000

# 8. 啟動命令
# 使用 uvicorn 啟動 main.py 中的 app 實例
# --reload 用於開發環境，若正式部署建議移除
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
