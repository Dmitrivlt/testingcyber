import os, time, hmac, hashlib, math
from dataclasses import dataclass
from typing import Dict, Any, Optional
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ========= ENV =========
load_dotenv()
API_KEY     = os.getenv("BINANCE_API_KEY") or ""
API_SECRET  = os.getenv("BINANCE_API_SECRET") or ""
USE_TESTNET = (os.getenv("USE_TESTNET", "true").lower() == "true")
SYMBOL      = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL    = os.getenv("INTERVAL", "1m")
LEVERAGE    = int(os.getenv("LEVERAGE", "2"))
POS_PCT     = float(os.getenv("POSITION_SIZE_PCT", "1.0"))

BASE_URL = "https://testnet.binancefuture.com" if USE_TESTNET else "https://fapi.binance.com"

# ========= TA =========
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

# ========= Binance Futures HTTP (без SDK) =========
class BinanceFuturesHTTP:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.session = requests.Session()
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url
        self.recv_window = 5000
        self.time_offset_ms = 0
        self.sync_time()

    # ----- подпись -----
    def _ts(self) -> int:
        return int(time.time() * 1000) + self.time_offset_ms

    def _sign(self, params: Dict[str, Any]) -> str:
        q = urlencode(params, doseq=True)
        sig = hmac.new(self.api_secret, q.encode(), hashlib.sha256).hexdigest()
        return q + "&signature=" + sig

    # ----- base requests -----
    def get(self, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False):
        params = params or {}
        headers = {}
        url = self.base_url + path
        if signed:
            params["timestamp"] = self._ts()
            params["recvWindow"] = self.recv_window
            headers["X-MBX-APIKEY"] = self.api_key
            return self.session.get(url + "?" + self._sign(params), headers=headers, timeout=15).json()
        return self.session.get(url, params=params, timeout=15).json()

    def post(self, path: str, params: Dict[str, Any], signed: bool = True):
        headers = {"X-MBX-APIKEY": self.api_key}
        url = self.base_url + path
        params["timestamp"] = self._ts()
        params["recvWindow"] = self.recv_window
        return self.session.post(url, headers=headers, data=self._sign(params), timeout=15).json()

    # ----- time sync -----
    def sync_time(self):
        try:
            r = self.session.get(self.base_url + "/fapi/v1/time", timeout=10).json()
            server = int(r["serverTime"])
            self.time_offset_ms = server - int(time.time() * 1000)
            print("Time synced. offset(ms)=", self.time_offset_ms)
        except Exception as e:
            print("Time sync failed:", e)
            self.time_offset_ms = 0

    # ----- endpoints -----
    def exchange_info(self, symbol: Optional[str] = None):
        p = {"symbol": symbol} if symbol else {}
        return self.get("/fapi/v1/exchangeInfo", p)

    def klines(self, symbol: str, interval: str, limit: int = 210):
        return self.get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def mark_price(self, symbol: str):
        return self.get("/fapi/v1/premiumIndex", {"symbol": symbol})

    def position_risk(self, symbol: str):
        return self.get("/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)

    def balance(self):
        return self.get("/fapi/v2/balance", signed=True)

    def change_position_mode(self, dual_side: bool):
        # false = One-way
        return self.post("/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if dual_side else "false"})

    def change_margin_type(self, symbol: str, margin_type: str):
        # "ISOLATED" / "CROSSED"
        return self.post("/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})

    def change_leverage(self, symbol: str, leverage: int):
        return self.post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def new_order(self, symbol: str, side: str, type_: str, quantity: str, reduce_only: Optional[bool] = None):
        p = {"symbol": symbol, "side": side, "type": type_, "quantity": quantity}
        if reduce_only is not None:
            p["reduceOnly"] = "true" if reduce_only else "false"
        return self.post("/fapi/v1/order", p)

# ========= helpers =========
def parse_filters(ex_info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    sym = next(s for s in ex_info["symbols"] if s["symbol"] == symbol)
    f = {ff["filterType"]: ff for ff in sym.get("filters", [])}
    lot = f.get("MARKET_LOT_SIZE") or f.get("LOT_SIZE", {})
    tick = f.get("PRICE_FILTER", {})
    notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL", {})
    res = {
        "minQty": float(lot.get("minQty", "0")),
        "maxQty": float(lot.get("maxQty", "1e50")),
        "stepSize": float(lot.get("stepSize", "1")),
        "tickSize": float(tick.get("tickSize", "0")),
        "minNotional": float(notional.get("minNotional", "0")) if "minNotional" in notional else None,
        "maxNotional": float(notional.get("maxNotional", "1e50")) if "maxNotional" in notional else None,
    }
    return res

def round_step(x: float, step: float) -> float:
    if step == 0:
        return x
    return math.floor(x / step) * step

# ========= Strategy Runner =========
@dataclass
class EMACrossoverInvertedLive:
    len50: int = 50
    len100: int = 100
    len200: int = 200

    def __post_init__(self):
        self.client = BinanceFuturesHTTP(API_KEY, API_SECRET, BASE_URL)

        # режим one-way и изолированная маржа (не обязательно; может вернуть ошибку, если позиция открыта)
        try: print("Position mode:", self.client.change_position_mode(False))
        except Exception as e: print("Position mode note:", e)

        try: print("Margin type:", self.client.change_margin_type(SYMBOL, "ISOLATED"))
        except Exception as e: print("Margin type note:", e)

        try: print("Leverage:", self.client.change_leverage(SYMBOL, LEVERAGE))
        except Exception as e: print("Leverage note:", e)

        try:
            ex = self.client.exchange_info(SYMBOL)
            self.filters = parse_filters(ex, SYMBOL)
            print("Filters:", self.filters)
        except Exception as e:
            print("Exchange info note:", e)
            self.filters = {"minQty": 0.0, "maxQty": 1e50, "stepSize": 1.0, "tickSize": 0.0,
                            "minNotional": None, "maxNotional": None}

        try:
            mp = self.client.mark_price(SYMBOL)
            print("Mark price:", mp.get("markPrice"))
        except Exception as e:
            print("Mark price note:", e)

    # ---- data ----
    def get_klines(self, limit=210) -> pd.DataFrame:
        k = self.client.klines(SYMBOL, INTERVAL, limit=limit)
        df = pd.DataFrame(k, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        df["close"] = df["close"].astype(float)
        return df

    def get_position_qty(self) -> float:
        r = self.client.position_risk(SYMBOL)
        if isinstance(r, list) and r:
            return float(r[0]["positionAmt"])
        return 0.0

    def account_equity(self) -> float:
        bal = self.client.balance()
        usdt = next((b for b in bal if b["asset"] == "USDT"), None)
        return float(usdt["balance"]) if usdt else 0.0

    def qty_from_equity(self, price: float, equity_usdt: float) -> float:
        notional = POS_PCT * equity_usdt
        raw_qty = notional / price
        qty = max(self.filters["minQty"], round_step(raw_qty, self.filters["stepSize"]))
        return qty

    def ensure_notional_ok(self, price: float, qty: float) -> bool:
        if qty <= 0:
            return False
        notional = price * qty
        mn = self.filters.get("minNotional")
        mx = self.filters.get("maxNotional")
        if mn is not None and notional < mn:
            print(f"Notional too small: {notional:.6f} < minNotional {mn}")
            return False
        if mx is not None and notional > mx:
            print(f"Notional too big: {notional:.6f} > maxNotional {mx}")
            return False
        return True

    def market_close_all(self, side_to_close: str, qty: float):
        if qty <= 0: return
        try:
            r = self.client.new_order(SYMBOL, side_to_close, "MARKET", str(qty), reduce_only=True)
            print("Close:", r)
        except Exception as e:
            print("reduceOnly close warning:", e)

    def market_open(self, side: str, qty: float):
        r = self.client.new_order(SYMBOL, side, "MARKET", str(qty))
        print("Open:", r)

    def run_once_on_close(self):
        df = self.get_klines()
        close = df["close"].astype(float)

        ema50  = ema(close, self.len50)
        ema100 = ema(close, self.len100)  # визуальный фон
        ema200 = ema(close, self.len200)

        long_sig  = crossunder(ema50, ema200)  # Лонг при crossunder(50,200)
        short_sig = crossover(ema50, ema200)   # Шорт при crossover(50,200)

        go_long  = bool(long_sig.iloc[-1])
        go_short = bool(short_sig.iloc[-1])

        # --- Снимок позиции всегда (для наглядности) ---
        try:
            pr = self.client.position_risk(SYMBOL)
            if isinstance(pr, list) and pr:
                p = pr[0]
                pos_qty = float(p.get("positionAmt", 0))
                entry   = float(p.get("entryPrice", 0))
                upnl    = float(p.get("unRealizedProfit", 0))
                side_txt = "LONG" if pos_qty > 0 else "SHORT" if pos_qty < 0 else "FLAT"
                print(f"Position snapshot: side={side_txt}, qty={pos_qty}, entry={entry}, uPnL={upnl}")
        except Exception as e:
            print("Position snapshot note:", e)

        if not (go_long or go_short):
            print("Нет сигнала на закрытии свечи.")
            return

        price  = float(self.client.mark_price(SYMBOL)["markPrice"])
        equity = self.account_equity()
        qty    = self.qty_from_equity(price, equity)

        if not self.ensure_notional_ok(price, qty):
            return

        pos_qty = self.get_position_qty()
        print(f"Signal: {'LONG' if go_long else 'SHORT'} | equity={equity:.4f} | price={price:.6f} | qty={qty} | pos_qty={pos_qty}")

        if go_long:
            if pos_qty < 0: self.market_close_all("BUY", abs(pos_qty))
            self.market_open("BUY", qty)

        if go_short:
            if pos_qty > 0: self.market_close_all("SELL", abs(pos_qty))
            self.market_open("SELL", qty)

# ========= main loop =========
if __name__ == "__main__":
    bot = EMACrossoverInvertedLive(len50=50, len100=100, len200=200)
    last_close_time = None
    while True:
        try:
            kl = bot.get_klines(limit=2)
            close_time = int(kl["close_time"].iloc[-1])
            if close_time != last_close_time:
                bot.run_once_on_close()
                last_close_time = close_time
        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print("Loop error:", e)
        time.sleep(10)
