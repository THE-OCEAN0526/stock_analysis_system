from fastapi import FastAPI
from controllers import stock_controller

app = FastAPI(
    title="專業股票分析系統 API",
    description="基於 MVC 架構重構的 API 系統",
    version="2.0.0"
)

# 掛載路由，並統一加上前綴 /api/v1
app.include_router(stock_controller.router, prefix="/api/v1", tags=["Stock Analysis"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
