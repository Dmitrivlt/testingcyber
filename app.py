import os, time, json, csv, threading, traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, List

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from dotenv import load_dotenv
import websocket  # websocket-client

# === импорт из твоего bot.py ===
from bot import (
    BinanceFuturesHTTP, ema, crossover, crossunder, parse_filters, round_step,
    API_KEY, API_SECRET
)

# ---------- ENV ----------
load_dotenv()
def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(default)).lower()
    return v in ("1", "true", "yes", "y", "on")

USE_TESTNET      = env_bool("USE_TESTNET", False)
DEFAULT_SYMBOL   = os.getenv("SYMBOL", "CYBERUSDT")
DEFAULT_INTERVAL = os.getenv("INTERVAL", "1m")
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", "2"))
# По умолчанию используем 90% депозита как МАРЖУ
DEFAULT_POS_PCT  = float(os.getenv("POSITION_SIZE_PCT", "0.90"))
TRADES_CSV       = os.getenv("TRADES_CSV", "trades.csv")

# ---------- CONFIG / ENGINE ----------
@dataclass
class Config:
    symbol: str = DEFAULT_SYMBOL
    interval: str = DEFAULT_INTERVAL
    leverage: int = DEFAULT_LEVERAGE
    pos_pct: float = DEFAULT_POS_PCT
    use_testnet: bool = USE_TESTNET
    # EMA
    len50: int = 4
    len100: int = 100
    len200: int = 6
    # Strategy mode
    # standard  → LONG=crossunder(ema50, ema200), SHORT=crossover(ema50, ema200)
    # inverted  → LONG=crossover(ema50, ema200), SHORT=crossunder(ema50, ema200)
    logic_mode: str = "inverted"
    # Risk
    sl_pct: float = 0.0            # стоп-лосс % (0=выкл)
    tp_pct: float = 0.0            # тейк-профит %
    trail_pct: float = 0.0         # трейлинг %
    daily_loss_limit: float = 0.0  # дневной лимит убытка (USDT, 0=выкл)

class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_rest = "https://testnet.binancefuture.com" if cfg.use_testnet else "https://fapi.binance.com"
        self.base_ws   = "wss://stream.binancefuture.com/stream?streams=" if cfg.use_testnet else "wss://fstream.binance.com/stream?streams="
        self.client = BinanceFuturesHTTP(API_KEY, API_SECRET, self.base_rest)
        self.filters = {"minQty": 0.0, "maxQty": 1e50, "stepSize": 1.0, "tickSize": 0.0,
                        "minNotional": None, "maxNotional": None}
        self.last_signal = None
        self.last_trade  = None
        self.last_note   = None
        self.state: Dict[str, Any] = {
            "running": False,
            "last_error": None,
            "snapshot": {},
            "ema": {},
            "signals": {},
            "filters": {},
            "blocked": False,
        }
        self.lock = threading.RLock()

        # RM / trailing
        self.trading_blocked = False
        self.trail_max = None
        self.trail_min = None
        self.start_equity = None

        # WS & loops
        self._stop_event = threading.Event()
        self.thread_rm: Optional[threading.Thread] = None
        self.thread_ws: Optional[threading.Thread] = None
        self.ws: Optional[websocket.WebSocketApp] = None

        # position change tracking (для логирования выходов по брекетам)
        self._prev_pos_qty: float = 0.0

        # CSV init
        self._ensure_csv_header()

        self._init_exchange()

    # ---------- CSV ----------
    def _ensure_csv_header(self):
        if not os.path.exists(TRADES_CSV):
            with open(TRADES_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts_utc","symbol","action","qty","price","reason","upnl","equity"])

    def _log_trade(self, action: str, qty: float, price: float, reason: str, upnl: float = 0.0, equity: float = 0.0):
        with open(TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([datetime.utcnow().isoformat()+"Z", self.cfg.symbol, action, qty, price, reason, upnl, equity])

    # ---------- helpers ----------
    def _get_mark(self) -> float:
        mp = self.client.mark_price(self.cfg.symbol)
        return float(mp.get("markPrice", 0) or 0)

    def _get_pos(self) -> Dict[str, Any]:
        r = self.client.position_risk(self.cfg.symbol)
        return r[0] if isinstance(r, list) and r else {}

    def _account_equity(self) -> float:
        bal = self.client.balance()
        usdt = next((b for b in bal if b["asset"] == "USDT"), None)
        return float(usdt["balance"]) if usdt else 0.0

    def _qty_from_equity(self, price: float, equity_usdt: float) -> float:
        """
        ВСЕГДА используем pos_pct депозита как МАРЖУ (например, 0.90 = 90%),
        а номинал позиции = маржа * плечо. Так доля маржи не зависит от leverage.
        """
        lev = max(1.0, float(self.cfg.leverage))
        margin_target = max(0.0, float(self.cfg.pos_pct)) * equity_usdt  # 90% депозита по умолчанию
        notional = margin_target * lev
        raw_qty = notional / price if price > 0 else 0.0
        return max(self.filters["minQty"], round_step(raw_qty, self.filters["stepSize"]))

    def _ensure_notional_ok(self, price: float, qty: float) -> bool:
        if qty <= 0: return False
        notional = price * qty
        mn = self.filters.get("minNotional"); mx = self.filters.get("maxNotional")
        if mn is not None and notional < mn:
            self.state["last_error"] = f"Notional too small: {notional} < {mn}"
            return False
        if mx is not None and notional > mx:
            self.state["last_error"] = f"Notional too big: {notional} > {mx}"
            return False
        return True

    def _fmt_price(self, p: float) -> float:
        # стоп-цены должны соответствовать tickSize
        step = self.filters.get("tickSize", 0.0) or 0.0
        if step <= 0: return p
        # округлим вниз к сетке (для SELL/long SL) и вверх (для BUY/short SL) там, где нужно — упростим: вниз
        digits = max(0, str(step)[::-1].find('.'))
        return round((p // step) * step, digits)

    def _open_brackets(self, side: str, entry: float):
        """
        Фьючерсы не поддерживают "настоящее OCO". Эмулируем брекеты:
        ставим два условных MARKET ордера с closePosition=true (SL и TP).
        Когда один триггерится и позиция закрывается, второй уже нечему исполнять.
        """
        try:
            if self.cfg.sl_pct > 0:
                if side == "LONG":
                    stop = self._fmt_price(entry * (1.0 - self.cfg.sl_pct/100.0))
                    self.client.post("/fapi/v1/order", {
                        "symbol": self.cfg.symbol, "side": "SELL", "type": "STOP_MARKET",
                        "stopPrice": f"{stop}", "closePosition": "true", "reduceOnly": "true",
                        "workingType": "MARK_PRICE"
                    })
                else:
                    stop = self._fmt_price(entry * (1.0 + self.cfg.sl_pct/100.0))
                    self.client.post("/fapi/v1/order", {
                        "symbol": self.cfg.symbol, "side": "BUY", "type": "STOP_MARKET",
                        "stopPrice": f"{stop}", "closePosition": "true", "reduceOnly": "true",
                        "workingType": "MARK_PRICE"
                    })
            if self.cfg.tp_pct > 0:
                if side == "LONG":
                    tp = self._fmt_price(entry * (1.0 + self.cfg.tp_pct/100.0))
                    self.client.post("/fapi/v1/order", {
                        "symbol": self.cfg.symbol, "side": "SELL", "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": f"{tp}", "closePosition": "true", "reduceOnly": "true",
                        "workingType": "MARK_PRICE"
                    })
                else:
                    tp = self._fmt_price(entry * (1.0 - self.cfg.tp_pct/100.0))
                    self.client.post("/fapi/v1/order", {
                        "symbol": self.cfg.symbol, "side": "BUY", "type": "TAKE_PROFIT_MARKET",
                        "stopPrice": f"{tp}", "closePosition": "true", "reduceOnly": "true",
                        "workingType": "MARK_PRICE"
                    })
        except Exception as e:
            self.state["last_error"] = f"Bracket place error: {e}"

    def _cancel_all_orders(self):
        try:
            self.client.post("/fapi/v1/allOpenOrders", {"symbol": self.cfg.symbol})
        except Exception:
            pass

    def _flatten(self):
        pos = self._get_pos()
        pos_qty = float(pos.get("positionAmt", 0) or 0)
        if pos_qty > 0:
            self.client.post("/fapi/v1/order", {"symbol": self.cfg.symbol, "side": "SELL", "type": "MARKET",
                                                "quantity": f"{abs(pos_qty)}", "reduceOnly": "true"})
        elif pos_qty < 0:
            self.client.post("/fapi/v1/order", {"symbol": self.cfg.symbol, "side": "BUY", "type": "MARKET",
                                                "quantity": f"{abs(pos_qty)}", "reduceOnly": "true"})
        self._cancel_all_orders()

    # ---------- init exchange ----------
    def _init_exchange(self):
        try:
            # только плечо (режим/маржу не трогаем — чтобы без ворнингов)
            try:
                self.client.change_leverage(self.cfg.symbol, self.cfg.leverage)
            except Exception as e:
                self.state["last_error"] = f"Leverage note: {e}"

            ex = self.client.exchange_info(self.cfg.symbol)
            self.filters = parse_filters(ex, self.cfg.symbol)
            mark = self._get_mark()
            pos = self._get_pos()
            upnl = float(pos.get("unRealizedProfit", 0) or 0)
            eq  = self._account_equity()
            self.start_equity = eq + upnl
            self._prev_pos_qty = float(pos.get("positionAmt", 0) or 0)

            with self.lock:
                self.state["filters"] = self.filters
                self.state["snapshot"] = {
                    "symbol": self.cfg.symbol,
                    "interval": self.cfg.interval,
                    "leverage": self.cfg.leverage,
                    "pos_pct": self.cfg.pos_pct,
                    "use_testnet": self.cfg.use_testnet,
                    "time_offset_ms": self.client.time_offset_ms,
                    "mark_price": mark,
                    "position_qty": float(pos.get("positionAmt", 0) or 0),
                    "entry_price": float(pos.get("entryPrice", 0) or 0),
                    "upnl": upnl,
                    "equity": eq + upnl,
                    "upnl_pct": 0.0,
                    "side": "LONG" if self._prev_pos_qty>0 else "SHORT" if self._prev_pos_qty<0 else "FLAT",
                    "rr_pct": 0.0,
                    "server_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
        except Exception as e:
            self.state["last_error"] = f"Init exchange note: {e}"

    # ---------- risk/tick loop ----------
    def _tick_update_and_risk(self):
        try:
            mark = self._get_mark()
            pos  = self._get_pos()
            entry = float(pos.get("entryPrice", 0) or 0)
            qty   = float(pos.get("positionAmt", 0) or 0)
            upnl  = float(pos.get("unRealizedProfit", 0) or 0)
            eq    = self._account_equity()
            side  = "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT"
            notional = abs(qty) * mark
            upnl_pct = (upnl / notional * 100.0) if notional > 0 else 0.0

            rr_pct = 0.0
            if entry > 0 and qty != 0:
                if qty > 0:
                    rr_pct = (mark / entry - 1.0) * 100.0
                else:
                    rr_pct = (entry / mark - 1.0) * 100.0

            # trailing baselines
            if qty > 0:
                self.trail_max = mark if self.trail_max is None else max(self.trail_max, mark)
                self.trail_min = None
            elif qty < 0:
                self.trail_min = mark if self.trail_min is None else min(self.trail_min, mark)
                self.trail_max = None
            else:
                self.trail_max = self.trail_min = None

            # лог выхода по брекету (позиция была !=0 и стала 0)
            if self._prev_pos_qty != 0 and qty == 0:
                self._log_trade("EXIT", abs(self._prev_pos_qty), mark, "BRACKET_FILLED", upnl, eq + upnl)
                self._cancel_all_orders()

            self._prev_pos_qty = qty

            with self.lock:
                self.state["snapshot"].update({
                    "mark_price": mark, "position_qty": qty, "entry_price": entry,
                    "upnl": upnl, "equity": eq + upnl, "upnl_pct": upnl_pct,
                    "side": side, "rr_pct": rr_pct,
                    "server_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                })
                self.state["blocked"] = self.trading_blocked

            # Daily loss limit
            if self.cfg.daily_loss_limit > 0 and self.start_equity is not None:
                if (self.start_equity - (eq + upnl)) >= self.cfg.daily_loss_limit:
                    if not self.trading_blocked:
                        self.trading_blocked = True
                        try:
                            self._flatten()
                            self.last_note = "Daily loss limit hit: closed position."
                            self._log_trade("EXIT", abs(qty), mark, "DAILY_LOSS_LIMIT", upnl, eq + upnl)
                        except Exception as e:
                            self.state["last_error"] = f"Flatten on DLL error: {e}"

            # SL/TP/Trailing как “мягкая” защита (если брекеты выключены)
            # Если cfg.sl_pct/tp_pct > 0 — брекеты уже стоят, этот блок можно не использовать.
            if qty != 0 and (self.cfg.sl_pct==0 and self.cfg.tp_pct==0) and not self.trading_blocked and entry > 0:
                if qty > 0:
                    if self.cfg.trail_pct > 0 and self.trail_max is not None:
                        if mark <= self.trail_max * (1.0 - self.cfg.trail_pct / 100.0):
                            self._flatten(); self.last_note = "Trailing stop hit"
                            self._log_trade("EXIT", abs(qty), mark, "TRAILING", upnl, eq + upnl)
                else:
                    if self.cfg.trail_pct > 0 and self.trail_min is not None:
                        if mark >= self.trail_min * (1.0 + self.cfg.trail_pct / 100.0):
                            self._flatten(); self.last_note = "Trailing stop hit"
                            self._log_trade("EXIT", abs(qty), mark, "TRAILING", upnl, eq + upnl)

        except Exception as e:
            self.state["last_error"] = f"Tick/RM error: {e}"

    def _risk_loop(self):
        while not self._stop_event.is_set():
            self._tick_update_and_risk()
            time.sleep(1.5)

    # ---------- strategy on kline close (from WS) ----------
    def _on_bar_close(self):
        try:
            df = pd.DataFrame(self.client.klines(self.cfg.symbol, self.cfg.interval, limit=210), columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","trades","tbbav","tbqav","ignore"
            ])
            df["close"] = df["close"].astype(float)
            close = df["close"].astype(float)
            ema50  = ema(close, self.cfg.len50)
            ema100 = ema(close, self.cfg.len100)
            ema200 = ema(close, self.cfg.len200)

            if self.cfg.logic_mode == "inverted":
                go_long  = bool(crossover(ema50, ema200).iloc[-1])
                go_short = bool(crossunder(ema50, ema200).iloc[-1])
            else:
                go_long  = bool(crossunder(ema50, ema200).iloc[-1])
                go_short = bool(crossover(ema50, ema200).iloc[-1])

            with self.lock:
                self.state["ema"] = {"ema50": float(ema50.iloc[-1]), "ema100": float(ema100.iloc[-1]), "ema200": float(ema200.iloc[-1])}
                self.state["signals"] = {"long": go_long, "short": go_short}
                self.last_note = "Нет сигнала на закрытии свечи." if not (go_long or go_short) else None

            if self.trading_blocked or not (go_long or go_short):
                return

            mark = self._get_mark()
            pos  = self._get_pos()
            qtyp = float(pos.get("positionAmt", 0) or 0)
            eq   = self._account_equity()
            qty  = self._qty_from_equity(mark, eq)
            if not self._ensure_notional_ok(mark, qty):
                return

            if go_long:
                if qtyp < 0:
                    self.client.post("/fapi/v1/order", {"symbol": self.cfg.symbol, "side": "BUY", "type": "MARKET",
                                                        "quantity": f"{abs(qtyp)}", "reduceOnly": "true"})
                self.client.post("/fapi/v1/order", {"symbol": self.cfg.symbol, "side": "BUY", "type": "MARKET",
                                                    "quantity": f"{qty}"})
                self.last_signal = "LONG"; self.last_trade = f"{datetime.utcnow().isoformat()}Z LONG qty={qty}"
                self.trail_max = mark; self.trail_min = None
                self._log_trade("ENTER_LONG", qty, mark, "SIGNAL")
                # брекеты
                if self.cfg.sl_pct>0 or self.cfg.tp_pct>0: self._open_brackets("LONG", mark)

            if go_short:
                if qtyp > 0:
                    self.client.post("/fapi/v1/order", {"symbol": self.cfg.symbol, "side": "SELL", "type": "MARKET",
                                                        "quantity": f"{abs(qtyp)}", "reduceOnly": "true"})
                self.client.post("/fapi/v1/order", {"symbol": self.cfg.symbol, "side": "SELL", "type": "MARKET",
                                                    "quantity": f"{qty}"})
                self.last_signal = "SHORT"; self.last_trade = f"{datetime.utcnow().isoformat()}Z SHORT qty={qty}"
                self.trail_min = mark; self.trail_max = None
                self._log_trade("ENTER_SHORT", qty, mark, "SIGNAL")
                if self.cfg.sl_pct>0 or self.cfg.tp_pct>0: self._open_brackets("SHORT", mark)

        except Exception as e:
            self.state["last_error"] = f"on_bar_close error: {e}"

    # ---------- WebSocket ----------
    def _ws_stream_name(self) -> str:
        return f"{self.cfg.symbol.lower()}@kline_{self.cfg.interval}"

    def _ws_loop(self):
        def on_message(ws, message):
            try:
                data = json.loads(message)
                k = data.get("data", {}).get("k", {})
                if not k: return
                is_closed = bool(k.get("x"))
                if is_closed:
                    self._on_bar_close()
            except Exception as e:
                self.state["last_error"] = f"WS message error: {e}"

        def on_error(ws, error):
            self.state["last_error"] = f"WS error: {error}"

        def on_close(ws, code, msg):
            # авто-переподключение
            if not self._stop_event.is_set():
                time.sleep(2)
                self._start_ws()

        def on_open(ws):
            pass

        url = self.base_ws + self._ws_stream_name()
        self.ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
        self.ws.run_forever(ping_interval=20, ping_timeout=10)

    def _start_ws(self):
        if self.thread_ws and self.thread_ws.is_alive(): return
        self.thread_ws = threading.Thread(target=self._ws_loop, daemon=True)
        self.thread_ws.start()

    # ---------- controls ----------
    def start(self):
        if self.state.get("running", False): return
        self._stop_event.clear()
        self.trading_blocked = False
        self.state["running"] = True
        # baseline equity ресетим при старте
        pos = self._get_pos()
        upnl = float(pos.get("unRealizedProfit", 0) or 0)
        self.start_equity = self._account_equity() + upnl
        self._prev_pos_qty = float(pos.get("positionAmt", 0) or 0)
        # запуски
        self.thread_rm = threading.Thread(target=self._risk_loop, daemon=True); self.thread_rm.start()
        self._start_ws()

    def stop(self):
        self._stop_event.set()
        if self.ws:
            try: self.ws.close()
            except Exception: pass
        if self.thread_rm: self.thread_rm.join(timeout=2)
        if self.thread_ws: self.thread_ws.join(timeout=2)
        self.state["running"] = False

    def flatten(self):
        try:
            self._flatten()
            mark = self._get_mark()
            pos = self._get_pos()
            upnl = float(pos.get("unRealizedProfit", 0) or 0)
            eq   = self._account_equity()
            self.last_note = "Position flattened."
            self._log_trade("EXIT", 0.0, mark, "MANUAL_FLATTEN", upnl, eq + upnl)
        except Exception as e:
            self.state["last_error"] = f"Flatten error: {e}"

    def reset_baseline(self):
        pos = self._get_pos()
        upnl = float(pos.get("unRealizedProfit", 0) or 0)
        self.start_equity = self._account_equity() + upnl
        self.trading_blocked = False
        self.last_note = "Baseline equity reset."

    def reconfigure(self, **kw):
        was_running = self.state.get("running", False)
        if was_running: self.stop()

        symbol   = kw.get("symbol")
        interval = kw.get("interval")
        leverage = kw.get("leverage")
        pos_pct  = kw.get("pos_pct")
        use_test = kw.get("use_testnet")
        logic    = kw.get("logic_mode")
        sl_pct   = kw.get("sl_pct")
        tp_pct   = kw.get("tp_pct")
        trail    = kw.get("trail_pct")
        dll      = kw.get("daily_loss_limit")

        if symbol:   self.cfg.symbol = symbol.upper().strip()
        if interval: self.cfg.interval = interval.strip()
        if leverage is not None: self.cfg.leverage = int(leverage)
        if pos_pct  is not None: self.cfg.pos_pct  = float(pos_pct)
        if use_test is not None: self.cfg.use_testnet = bool(use_test)
        if logic:    self.cfg.logic_mode = logic if logic in ("inverted","standard") else self.cfg.logic_mode
        if sl_pct  is not None: self.cfg.sl_pct  = max(0.0, float(sl_pct))
        if tp_pct  is not None: self.cfg.tp_pct  = max(0.0, float(tp_pct))
        if trail   is not None: self.cfg.trail_pct = max(0.0, float(trail))
        if dll     is not None: self.cfg.daily_loss_limit = max(0.0, float(dll))

        # recreate clients/urls/filters
        self.base_rest = "https://testnet.binancefuture.com" if self.cfg.use_testnet else "https://fapi.binance.com"
        self.base_ws   = "wss://stream.binancefuture.com/stream?streams=" if self.cfg.use_testnet else "wss://fstream.binance.com/stream?streams="
        self.client = BinanceFuturesHTTP(API_KEY, API_SECRET, self.base_rest)
        try: self.client.change_leverage(self.cfg.symbol, self.cfg.leverage)
        except Exception: pass
        ex = self.client.exchange_info(self.cfg.symbol)
        self.filters = parse_filters(ex, self.cfg.symbol)

        # пересобрать baseline и трейлинг
        self.trail_max = self.trail_min = None
        self.trading_blocked = False
        self.start_equity = None
        self._prev_pos_qty = 0.0

        if was_running: self.start()

# ---------- FastAPI ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
engine = Engine(Config())

# ---------- HTML UI ----------
INDEX_HTML = """
<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>EMA Crossover Bot — Dashboard</title>
<style>
  body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:20px}
  .grid{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:12px}
  .card{padding:12px;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  .title{font-size:20px;font-weight:700;margin-bottom:8px}
  .muted{color:#6b7280}
  .btn{padding:10px 16px;border-radius:10px;border:1px solid #e5e7eb;cursor:pointer;background:#111827;color:#fff}
  .btn2{padding:10px 16px;border-radius:10px;border:1px solid #e5e7eb;cursor:pointer;background:#f3f4f6}
  .danger{background:#991b1b;color:#fff}
  .row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
  input,select{padding:8px 10px;border-radius:8px;border:1px solid #d1d5db}
  code{background:#f9fafb;padding:2px 6px;border-radius:6px}
  table{border-collapse:collapse;width:100%}
  th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;font-size:12px}
</style>
<script>
async function fetchState(){
  const r = await fetch('/api/state'); const s = await r.json();
  const e=(id,v)=>document.getElementById(id).textContent=v;
  e('running', s.running?'RUNNING':'STOPPED');
  e('blocked', s.blocked?'YES':'NO');
  e('symbol', s.snapshot.symbol||'-'); e('interval', s.snapshot.interval||'-');
  e('leverage', s.snapshot.leverage||'-'); e('pos_pct', s.snapshot.pos_pct||'-');
  e('net', s.snapshot.use_testnet?'TESTNET':'MAINNET');
  e('offset', s.snapshot.time_offset_ms??'-');
  e('mark', (s.snapshot.mark_price??0).toFixed(6));
  e('qty', s.snapshot.position_qty??0);
  e('side', s.snapshot.side||'FLAT');
  e('entry', (s.snapshot.entry_price??0).toFixed(6));
  e('upnl', (s.snapshot.upnl??0).toFixed(6));
  e('upnl_pct', (s.snapshot.upnl_pct??0).toFixed(2)+'%');
  e('equity', (s.snapshot.equity??0).toFixed(6));
  e('rr', (s.snapshot.rr_pct??0).toFixed(2)+'%');
  e('ema50', (s.ema.ema50??0).toFixed(6));
  e('ema200', (s.ema.ema200??0).toFixed(6));
  e('sig', `long=${s.signals.long?1:0} | short=${s.signals.short?1:0}`);
  e('filters', JSON.stringify(s.filters));
  e('last_trade', s.last_trade || '-');
  e('last_note', s.last_note || '-');
  e('last_error', s.last_error || '-');
  e('server_time', s.snapshot.server_time || '-');

  // trades table (last 30)
  const rt = await fetch('/api/trades'); const tj = await rt.json();
  const rows = (tj.slice(-30)).map(x=>`<tr><td>${x.ts_utc}</td><td>${x.symbol}</td><td>${x.action}</td><td>${x.qty}</td><td>${x.price}</td><td>${x.reason}</td><td>${x.upnl}</td><td>${x.equity}</td></tr>`).join('');
  document.getElementById('trades_body').innerHTML = rows || '<tr><td colspan="8" class="muted">нет записей</td></tr>';
}
async function applyConfig(ev){
  ev.preventDefault();
  const payload={
    use_testnet: document.getElementById('f_testnet').checked,
    symbol: document.getElementById('f_symbol').value.trim(),
    interval: document.getElementById('f_interval').value.trim(),
    leverage: parseInt(document.getElementById('f_leverage').value,10),
    pos_pct: parseFloat(document.getElementById('f_pospct').value),
    logic_mode: document.getElementById('f_logic').value,
    sl_pct: parseFloat(document.getElementById('f_sl').value||0),
    tp_pct: parseFloat(document.getElementById('f_tp').value||0),
    trail_pct: parseFloat(document.getElementById('f_trail').value||0),
    daily_loss_limit: parseFloat(document.getElementById('f_dll').value||0),
  };
  const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const j=await r.json(); alert(j.ok?'Config applied':'Error: '+(j.error||'unknown')); fetchState();
}
async function startBot(){ await fetch('/api/start',{method:'POST'}); fetchState(); }
async function stopBot(){ await fetch('/api/stop',{method:'POST'}); fetchState(); }
async function flatten(){ await fetch('/api/flatten',{method:'POST'}); fetchState(); }
async function resetBaseline(){ await fetch('/api/reset_baseline',{method:'POST'}); fetchState(); }
setInterval(fetchState,3000); window.addEventListener('load',fetchState);
</script>
</head>
<body>
  <div class="row">
    <div class="title">EMA Crossover Bot — Dashboard</div>
    <div class="row"><span class="muted">Status:</span> <b id="running">-</b></div>
    <div class="row"><span class="muted">Trading blocked:</span> <b id="blocked">-</b></div>
    <button class="btn" onclick="startBot()">Start</button>
    <button class="btn2" onclick="stopBot()">Stop</button>
    <button class="btn danger" onclick="flatten()">Flatten (Close Position)</button>
    <button class="btn2" onclick="resetBaseline()">Reset baseline</button>
  </div>

  <div class="grid">
    <div class="card">
      <div class="title">Config</div>
      <div class="row"><span class="muted">Network:</span> <b id="net">-</b></div>
      <div class="row"><span class="muted">Symbol:</span> <b id="symbol">-</b></div>
      <div class="row"><span class="muted">TF:</span> <b id="interval">-</b></div>
      <div class="row"><span class="muted">Leverage:</span> <b id="leverage">-</b></div>
      <div class="row"><span class="muted">% Equity:</span> <b id="pos_pct">-</b></div>
      <div class="row"><span class="muted">Time offset(ms):</span> <b id="offset">-</b></div>
    </div>

    <div class="card">
      <div class="title">Market / EMA</div>
      <div class="row"><span class="muted">Mark:</span> <b id="mark">-</b></div>
      <div class="row"><span class="muted">EMA50:</span> <b id="ema50">-</b></div>
      <div class="row"><span class="muted">EMA200:</span> <b id="ema200">-</b></div>
      <div class="row"><span class="muted">Signals:</span> <b id="sig">-</b></div>
    </div>

    <div class="card">
      <div class="title">Position</div>
      <div class="row"><span class="muted">Side:</span> <b id="side">-</b></div>
      <div class="row"><span class="muted">Qty:</span> <b id="qty">-</b></div>
      <div class="row"><span class="muted">Entry:</span> <b id="entry">-</b></div>
      <div class="row"><span class="muted">uPnL:</span> <b id="upnl">-</b></div>
      <div class="row"><span class="muted">uPnL%:</span> <b id="upnl_pct">-</b></div>
      <div class="row"><span class="muted">R/R%:</span> <b id="rr">-</b></div>
      <div class="row"><span class="muted">Equity:</span> <b id="equity">-</b></div>
    </div>

    <div class="card">
      <div class="title">Diagnostics</div>
      <div class="row"><span class="muted">Filters:</span> <b id="filters">-</b></div>
      <div class="row"><span class="muted">Last trade:</span> <b id="last_trade">-</b></div>
      <div class="row"><span class="muted">Note:</span> <b id="last_note">-</b></div>
      <div class="row"><span class="muted">Error:</span> <b id="last_error">-</b></div>
      <div class="row"><span class="muted">Server time:</span> <b id="server_time">-</b></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <div class="title">Change settings</div>
    <form onsubmit="applyConfig(event)">
      <div class="row">
        <label>Use testnet</label><input id="f_testnet" type="checkbox"/>
        <label>Symbol</label><input id="f_symbol" value="CYBERUSDT"/>
        <label>Interval</label>
        <select id="f_interval">
          <option>1m</option><option>3m</option><option>5m</option><option>15m</option>
          <option>30m</option><option>1h</option><option>2h</option><option>4h</option>
        </select>
        <label>Logic</label>
        <select id="f_logic"><option value="inverted" selected>inverted</option><option value="standard">standard</option></select>
      </div>
      <div class="row">
        <label>Leverage</label><input id="f_leverage" type="number" min="1" max="125" value="2"/>
        <!-- По умолчанию показываем 0.90 -->
        <label>% Equity</label><input id="f_pospct" type="number" step="0.01" min="0.01" max="1.00" value="0.90"/>
        <label>SL %</label><input id="f_sl" type="number" step="0.01" min="0" value="0"/>
        <label>TP %</label><input id="f_tp" type="number" step="0.01" min="0" value="0"/>
        <label>Trail %</label><input id="f_trail" type="number" step="0.01" min="0" value="0"/>
        <label>Daily loss limit (USDT)</label><input id="f_dll" type="number" step="0.01" min="0" value="0"/>
      </div>
      <div class="row">
        <button class="btn2" type="submit">Apply</button>
        <a class="btn2" href="/trades.csv" download>Download trades.csv</a>
      </div>
      <div class="muted">Брекеты (SL/TP) ставятся как STOP_MARKET/TAKE_PROFIT_MARKET с closePosition=true (OCO-подобно).</div>
    </form>
  </div>

  <div class="card" style="margin-top:16px;">
    <div class="title">Trades (last 30)</div>
    <table><thead><tr>
      <th>ts_utc</th><th>symbol</th><th>action</th><th>qty</th><th>price</th><th>reason</th><th>upnl</th><th>equity</th>
    </tr></thead><tbody id="trades_body"><tr><td colspan="8" class="muted">loading…</td></tr></tbody></table>
  </div>
</body></html>
"""

# ---------- API ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
_engine = Engine(Config())

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/api/state")
def api_state():
    with _engine.lock:
        d = {
            "running": _engine.state.get("running", False),
            "blocked": _engine.state.get("blocked", False),
            "last_error": _engine.state.get("last_error"),
            "last_signal": _engine.last_signal,
            "last_trade": _engine.last_trade,
            "last_note": _engine.last_note,
            "snapshot": _engine.state.get("snapshot", {}),
            "ema": _engine.state.get("ema", {}),
            "signals": _engine.state.get("signals", {}),
            "filters": _engine.state.get("filters", {}),
        }
    return JSONResponse(d)

@app.get("/api/trades")
def api_trades():
    rows: List[Dict[str, Any]] = []
    if os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "r") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
    return rows

@app.get("/trades.csv")
def download_csv():
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_utc","symbol","action","qty","price","reason","upnl","equity"])
    return FileResponse(TRADES_CSV, filename="trades.csv", media_type="text/csv")

@app.post("/api/start")
def api_start():
    _engine.start()
    return {"ok": True, "running": True}

@app.post("/api/stop")
def api_stop():
    _engine.stop()
    return {"ok": True, "running": False}

@app.post("/api/flatten")
def api_flatten():
    _engine.flatten()
    return {"ok": True}

@app.post("/api/reset_baseline")
def api_reset_baseline():
    _engine.reset_baseline()
    return {"ok": True}

@app.post("/api/config")
async def api_config(req: Request):
    body = await req.json()
    try:
        _engine.reconfigure(**body)
        return {"ok": True, "cfg": asdict(_engine.cfg)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

# ASGI app export
app = app
