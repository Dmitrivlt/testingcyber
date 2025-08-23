import os, time, math
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# pip install binance-connector
from binance.um_futures import UMFutures

# ================== Config ==================
load_dotenv()
API_KEY     = os.getenv("BINANCE_API_KEY")
API_SECRET  = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
SYMBOL      = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL    = os.getenv("INTERVAL", "1m")
LEVERAGE    = int(os.getenv("LEVERAGE", "2"))
POS_PCT     = float(os.getenv("POSITION_SIZE_PCT", "1.0"))

# Testnet base_url (mainnet = None)
BASE_URL = "https://testnet.binancefuture.com" if USE_TESTNET else None

# ================== TA helpers ==================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    # текущая свеча: a>b, предыдущая: a<=b
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    # текущая свеча: a<b, предыдущая: a>=b
    return (a < b) & (a.shift(1) >= b.shift(1))

# ================== Filters helpers ==================
def parse_filters(exchange_info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    sym = next(s for s in exchange_info["symbols"] if s["symbol"] == symbol)
    filters = {f["filterType"]: f for f in sym.get("filters", [])}
    lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
    tick = filters.get("PRICE_FILTER", {})
    # На USD-M фьючах minNotional может отсутствовать; тогда просто не проверяем
    notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL", {})
    return {
        "minQty": float(lot.get("minQty", "0")),
        "maxQty": float(lot.get("maxQty", "1e50")),
        "stepSize": float(lot.get("stepSize", "1")),
        "tickSize": float(tick.get("tickSize", "0")),
        "minNotional": float(notional.get("minNotional", "0")) if "minNotional" in notional else None,
        "maxNotional": float(notional.get("maxNotional", "1e50")) if "maxNotional" in notional else None,
    }

def round_step(x: float, step: float) -> float:
    if step == 0:
        return x
    return math.floor(x / step) * step

# ================== Strategy Runner ==================
@dataclass
class EMACrossoverInvertedLive:
    len50: int = 50
    len100: int = 100
    len200: int = 200

    def __post_init__(self):
        self.client = UMFutures(key=API_KEY, secret=API_SECRET, base_url=BASE_URL)

        # Режим позиции: one-way (не хедж). Если уже выставлен — будет warning, это нормально.
        try:
            self.client.change_position_mode(dualSidePosition="false")
        except Exception as e:
            print("Position mode note:", e)

        # По желанию можно изолированную маржу; если позиция открыта — может не дать сменить.
        try:
            self.client.change_margin_type(symbol=SYMBOL, marginType="ISOLATED")
        except Exception as e:
            print("Margin type note:", e)

        # Плечо
        try:
            self.client.change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        except Exception as e:
            print("Leverage note:", e)

        # Биржевые фильтры
        try:
            self.filters = parse_filters(self.client.exchange_info(), SYMBOL)
            print("Filters:", self.filters)
        except Exception as e:
            print("Exchange info note:", e)
            self.filters = {"minQty": 0.0, "maxQty": 1e50, "stepSize": 1.0, "tickSize": 0.0,
                            "minNotional": None, "maxNotional": None}

        # Быстрая проверка цены
        try:
            mp = self.client.mark_price(symbol=SYMBOL)
            print("Mark price:", mp.get("markPrice"))
        except Exception as e:
            print("Mark price note:", e)

    # ---------- Data ----------
    def get_price(self) -> float:
        mp = self.client.mark_price(symbol=SYMBOL)
        return float(mp["markPrice"])

    def get_klines(self, limit=210) -> pd.DataFrame:
        k = self.client.klines(symbol=SYMBOL, interval=INTERVAL, limit=limit)
        df = pd.DataFrame(k, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])
        df["close"] = df["close"].astype(float)
        return df

    # ---------- Account/position ----------
    def get_position_qty(self) -> float:
        # Возьмём position_information (qty>0=long, <0=short)
        info = self.client.position_information(symbol=SYMBOL)
        if isinstance(info, list) and info:
            return float(info[0]["positionAmt"])
        return 0.0

    def account_equity(self) -> float:
        # Баланс USDT на фьючерсах
        bal = self.client.balance()
        usdt = next((b for b in bal if b["asset"] == "USDT"), None)
        if not usdt:
            return 0.0
        return float(usdt["balance"])

    # ---------- Sizing/checks ----------
    def qty_from_equity(self, price: float, equity_usdt: float) -> float:
        # Номинал позиции = POS_PCT * equity; qty = notional / price
        notional = POS_PCT * equity_usdt
        raw_qty = notional / price
        # округляем под stepSize и проверяем минимум
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

    # ---------- Trading ----------
    def market_close_all(self, side_to_close: str, qty: float):
        if qty <= 0:
            return
        try:
            self.client.new_order(
                symbol=SYMBOL, side=side_to_close, type="MARKET",
                quantity=str(qty), reduceOnly="true"
            )
        except Exception as e:
            print("reduceOnly close warning:", e)

    def market_open(self, side: str, qty: float):
        self.client.new_order(symbol=SYMBOL, side=side, type="MARKET", quantity=str(qty))

    # ---------- Logic on bar close ----------
    def run_once_on_close(self):
        df = self.get_klines()
        close = df["close"].astype(float)

        ema50  = ema(close, self.len50)
        ema100 = ema(close, self.len100)  # для визуального/отладочного фона (на входы не влияет)
        ema200 = ema(close, self.len200)

        long_sig  = crossunder(ema50, ema200)  # Лонг на crossunder(50,200)
        short_sig = crossover(ema50, ema200)   # Шорт на crossover(50,200)

        go_long  = bool(long_sig.iloc[-1])
        go_short = bool(short_sig.iloc[-1])

        if not (go_long or go_short):
            print("Нет сигнала на закрытии свечи.")
            return

        price  = self.get_price()
        equity = self.account_equity()
        qty    = self.qty_from_equity(price, equity)

        if not self.ensure_notional_ok(price, qty):
            return

        pos_qty = self.get_position_qty()
        print(f"Signal: {'LONG' if go_long else 'SHORT'} | equity={equity:.4f} | price={price:.6f} | qty={qty} | pos_qty={pos_qty}")

        # Переворот: сначала закрыть встречную сторону, затем открыть нужную
        if go_long:
            if pos_qty < 0:
                self.market_close_all("BUY", abs(pos_qty))
            self.market_open("BUY", qty)

        if go_short:
            if pos_qty > 0:
                self.market_close_all("SELL", abs(pos_qty))
            self.market_open("SELL", qty)


if __name__ == "__main__":
    # Перед стартом: проверь .env (ключи, USE_TESTNET, SYMBOL, INTERVAL, плечо, процент позиции)
    bot = EMACrossoverInvertedLive(len50=50, len100=100, len200=200)
    last_close_time = None

    # Простейший цикл «на закрытии свечи»
    while True:
        try:
            kl = bot.get_klines(limit=2)
            close_time = int(kl["close_time"].iloc[-1])  # миллисекунды конца текущей свечи
            if close_time != last_close_time:
                bot.run_once_on_close()
                last_close_time = close_time
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print("Loop error:", e)
        time.sleep(10)
