from pydantic import BaseModel
from typing import Dict, Any

class StockResponse(BaseModel):
    status: str
    ticker: str
    reference_price: float
    today_stats: Dict[str, Any]
    performance: Dict[str, Any]
    data: Dict[str, Any]
