from fastapi import APIRouter, Query, HTTPException
from services.stock_service import StockService
import logging

# 初始化路由
router = APIRouter()

# 初始化 Service 實例 (OOP 方式)
stock_service = StockService()

@router.get("/stock/{ticker}", summary="獲取完整股票分析數據")
async def get_stock_analysis(
    ticker: str,
    period: str = Query("1y", description="時間範圍 (例如: 1d, 1mo, 1y, max)"),
    interval: str = Query("1d", description="資料密度 (例如: 1m, 15m, 1d)"),
    short_p: int = Query(10, description="短期均線週期"),
    long_p: int = Query(50, description="長期均線週期"),
    predict_modes: list[str] = Query([])
):
    print(f"🚀 [DEBUG] 收到請求，目標股票: {ticker}")
    """
    股票分析 Controller：
    1. 接收前端傳來的 ticker 與參數
    2. 呼叫 StockService 進行數據抓取與計算
    3. 回傳標準化 JSON 格式
    """
    try:
        # 呼叫 Service 層處理核心邏輯
        result = stock_service.get_full_analysis(
            ticker=ticker,
            period=period,
            interval=interval,
            short_p=short_p,
            long_p=long_p,
            predict_modes=predict_modes
        )
        
        # 檢查 Service 執行的結果狀態
        if result.get("status") == "error":
            # 如果是業務邏輯上的錯誤，拋出 400 或 404
            raise HTTPException(status_code=400, detail=result.get("message"))
            
        return result

    except HTTPException as he:
        # 重新拋出已定義的 HTTP 異常
        raise he
    except Exception as e:
        # 捕捉未預期的系統錯誤
        logging.error(f"Controller Error: {str(e)}")
        raise HTTPException(status_code=500, detail="伺服器內部錯誤，請檢查後端日誌")

@router.get("/health", include_in_schema=False)
async def health_check():
    """系統健康檢查接口"""
    return {"status": "healthy"}

@router.get("/stocks/list", summary="獲取全台股股票清單")
async def get_all_stocks():
    """
    回傳格式: ["2330.TW - 台積電", "2454.TW - 聯發科", ...]
    """
    try:
        # 這裡可以直接調用 downloader，或者透過 StockService 轉發
        from services.downloader import StockDownloader
        downloader = StockDownloader()
        stocks = downloader.get_taiwan_stock_list()
        return {"status": "success", "data": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
