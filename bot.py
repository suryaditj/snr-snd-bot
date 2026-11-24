"""
SNR/SND Multi-Timeframe Trading Bot — single-file edition.
Markets: XAUUSD, BTCUSD  |  Methods: SCALPING (H1+M15), ENTRY DAY (H4+H1+M5)
Hosted on Railway, controlled from Telegram inline keyboards.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# ============================================================ logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("snrbot")

# ============================================================ time utils
UTC = timezone.utc
WIB = ZoneInfo("Asia/Jakarta")
_INTERVALS = {
    "M5": timedelta(minutes=5), "M15": timedelta(minutes=15),
    "H1": timedelta(hours=1), "H4": timedelta(hours=4),
}


def now_utc() -> datetime:
    return datetime.now(UTC)


def fmt_wib(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(WIB).strftime("%d %b %Y, %H:%M WIB")


def is_closed(open_time: datetime, interval: str) -> bool:
    if open_time.tzinfo is None:
        open_time = open_time.replace(tzinfo=UTC)
    return now_utc() >= open_time + _INTERVALS[interval]


# ============================================================ models
class Direction(str, Enum):
    BULLISH = "BULLISH"; BEARISH = "BEARISH"; NEUTRAL = "NEUTRAL"


class Action(str, Enum):
    BUY = "BUY"; SELL = "SELL"


class Status(str, Enum):
    ANALYZING = "ANALYZING"
    WAITING_ZONE = "WAITING ZONE"
    WAITING_CONFIRMATION = "WAITING CONFIRMATION"
    CONFIRMED_BUY = "CONFIRMED BUY"
    CONFIRMED_SELL = "CONFIRMED SELL"
    NO_TRADE = "NO TRADE"
    INVALIDATED = "INVALIDATED"


class ZoneKind(str, Enum):
    DEMAND = "DEMAND"; SUPPLY = "SUPPLY"; SUPPORT = "SUPPORT"; RESISTANCE = "RESISTANCE"


class ZoneState(str, Enum):
    FRESH = "FRESH"; TESTED = "TESTED"; INVALID = "INVALID"


class Pattern(str, Enum):
    RBR = "RBR"; DBD = "DBD"; RBD = "RBD"; DBR = "DBR"; AREA = "AREA"


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float; high: float; low: float; close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Swing:
    index: int; price: float; is_high: bool; confirmed_at: int


@dataclass(frozen=True)
class Break:
    direction: Direction; broken_index: int; candle_index: int; displacement_atr: float


@dataclass
class Zone:
    kind: ZoneKind
    pattern: Pattern
    timeframe: str
    proximal: float
    distal: float
    created_at: datetime
    state: ZoneState = ZoneState.FRESH
    touches: int = 0
    quality: float = 0.0
    supports: Action | None = None

    @property
    def low(self) -> float:
        return min(self.proximal, self.distal)

    @property
    def high(self) -> float:
        return max(self.proximal, self.distal)


@dataclass
class Structure:
    direction: Direction
    swings: list[Swing] = field(default_factory=list)
    bos: Break | None = None
    choch: Break | None = None


@dataclass
class Signal:
    action: Action
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    zone_text: str
    confirmation_tf: str
    reasons: list[str]


@dataclass
class Analysis:
    market: str
    method: str
    direction_tf: str
    direction: Direction = Direction.NEUTRAL
    confirmation_tf: str = ""
    confirmation_dir: Direction | None = None
    zones: list[Zone] = field(default_factory=list)
    status: Status = Status.ANALYZING
    reasons: list[str] = field(default_factory=list)
    signal: Signal | None = None
    timestamp: datetime = field(default_factory=now_utc)
    last_close: datetime | None = None
    error: str | None = None


# ============================================================ config
class ConfigError(Exception):
    pass


def load_config() -> dict[str, Any]:
    path = Path(__file__).parent / "config.json"
    if not path.exists():
        raise ConfigError(f"config.json tidak ditemukan di {path}")
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    required = {"timezone", "markets", "methods", "structure", "zones",
                "snr", "confirmation", "risk", "api", "bot"}
    missing = required - set(cfg)
    if missing:
        raise ConfigError(f"Config missing keys: {missing}")
    for mkt, p in cfg["markets"].items():
        for k in ("symbol", "display", "precision", "min_rr"):
            if k not in p:
                raise ConfigError(f"Market {mkt} missing {k}")
    return cfg


# ============================================================ data layer
class APIError(Exception):
    pass


class TwelveData:
    def __init__(self, api_key: str, cfg: dict[str, Any]):
        if not api_key:
            raise APIError("TWELVEDATA_API_KEY kosong")
        self._key = api_key
        c = cfg["api"]
        self._base = c["base_url"].rstrip("/")
        self._timeout = c["timeout_seconds"]
        self._retries = c["max_retries"]
        self._backoff = c["retry_backoff_seconds"]
        self._output = c["candles_per_request"]

    async def fetch(self, symbol: str, interval: str) -> list[Candle]:
        params = {
            "symbol": symbol, "interval": interval,
            "outputsize": self._output, "apikey": self._key,
            "timezone": "UTC", "format": "JSON",
        }
        url = f"{self._base}/time_series"
        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    r = await client.get(url, params=params)
                if r.status_code == 429:
                    raise APIError("Rate limit Twelve Data (429)")
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and data.get("status") == "error":
                    raise APIError(data.get("message", "API error"))
                return self._parse(data, interval)
            except (httpx.TimeoutException, httpx.HTTPError, APIError) as e:
                last_exc = e
                if attempt < self._retries:
                    await asyncio.sleep(self._backoff * attempt)
                    logger.warning("Retry %d/%d: %s", attempt, self._retries, e)
        raise APIError(f"Gagal fetch setelah {self._retries}x: {last_exc}")

    @staticmethod
    def _parse(data: dict[str, Any], interval: str) -> list[Candle]:
        values = data.get("values") or []
        if not values:
            raise APIError(f"Data kosong untuk {interval}")
        out: list[Candle] = []
        for row in values:
            try:
                ts = row["datetime"]
                if " " in ts:
                    fmt = "%Y-%m-%d %H:%M:%S"
                elif "T" in ts:
                    fmt = "%Y-%m-%dT%H:%M:%S"
                else:
                    fmt = "%Y-%m-%d"
                ot = datetime.strptime(ts, fmt).replace(tzinfo=UTC)
                out.append(Candle(
                    open_time=ot,
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row.get("volume") or 0),
                ))
            except (KeyError, ValueError):
                continue
        out.sort(key=lambda c: c.open_time)
        return out


class Cache:
    def __init__(self, cfg: dict[str, Any]):
        self._ttl = {k: int(v) for k, v in cfg["api"]["cache_ttl_seconds"].items()}
        self._store: dict[tuple[str, str], tuple[list[Candle], datetime]] = {}

    def get(self, sym: str, interval: str) -> list[Candle] | None:
        entry = self._store.get((sym, interval))
        if not entry:
            return None
        candles, expires = entry
        if now_utc() >= expires:
            return None
        return candles

    def set(self, sym: str, interval: str, candles: list[Candle]) -> None:
        ttl = self._ttl.get(interval, 300)
        self._store[(sym, interval)] = (candles, now_utc() + timedelta(seconds=ttl))


def clean_candles(candles: list[Candle], interval: str, min_required: int) -> list[Candle]:
    seen: set = set()
    unique: list[Candle] = []
    for c in candles:
        if c.open_time in seen:
            continue
        seen.add(c.open_time)
        if c.high < c.low or c.high < max(c.open, c.close) or c.low > min(c.open, c.close):
            continue
        unique.append(c)
    unique.sort(key=lambda c: c.open_time)
    closed = [c for c in unique if is_closed(c.open_time, interval)]
    if len(closed) < min_required:
        raise APIError(f"Candle close tidak cukup ({len(closed)}<{min_required}) untuk {interval}")
    return closed


# ============================================================ structure analysis
def atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    tail = trs[-period:]
    return sum(tail) / len(tail)


def detect_swings(candles: list[Candle], lookback: int, confirm: int) -> list[Swing]:
    out: list[Swing] = []
    n = len(candles)
    last_ok = n - confirm
    for i in range(lookback, n - lookback):
        if i > last_ok:
            break
        left = candles[i - lookback:i]
        right = candles[i + 1:i + 1 + lookback]
        if not left or not right:
            continue
        hi = candles[i].high
        lo = candles[i].low
        if hi > max(c.high for c in left) and hi > max(c.high for c in right):
            out.append(Swing(i, hi, True, i + lookback))
        if lo < min(c.low for c in left) and lo < min(c.low for c in right):
            out.append(Swing(i, lo, False, i + lookback))
    out.sort(key=lambda s: s.index)
    return out


def detect_bos(candles: list[Candle], swings: list[Swing], a: float, threshold: float):
    if not swings or a <= 0:
        return None
    hi = next((s for s in reversed(swings) if s.is_high), None)
    lo = next((s for s in reversed(swings) if not s.is_high), None)
    for i in range(len(candles) - 1, -1, -1):
        body = abs(candles[i].close - candles[i].open)
        if body < a * threshold:
            continue
        if hi and candles[i].close > hi.price and i > hi.confirmed_at:
            return Break(Direction.BULLISH, hi.index, i, body / a)
        if lo and candles[i].close < lo.price and i > lo.confirmed_at:
            return Break(Direction.BEARISH, lo.index, i, body / a)
    return None


def swing_direction(swings: list[Swing]) -> Direction:
    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if not s.is_high]
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
            return Direction.BULLISH
        if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
            return Direction.BEARISH
    return Direction.NEUTRAL


def detect_choch(candles, swings, a, threshold, bos):
    if not swings or a <= 0:
        return None
    trend = bos.direction if bos else swing_direction(swings)
    if trend == Direction.NEUTRAL:
        return None
    hi = next((s for s in reversed(swings) if s.is_high), None)
    lo = next((s for s in reversed(swings) if not s.is_high), None)
    for i in range(len(candles) - 1, -1, -1):
        body = abs(candles[i].close - candles[i].open)
        if body < a * threshold:
            continue
        if trend == Direction.BULLISH and lo and i > lo.confirmed_at and candles[i].close < lo.price:
            return Break(Direction.BEARISH, lo.index, i, body / a)
        if trend == Direction.BEARISH and hi and i > hi.confirmed_at and candles[i].close > hi.price:
            return Break(Direction.BULLISH, hi.index, i, body / a)
    return None


def analyze_structure(candles: list[Candle], cfg: dict[str, Any]) -> Structure:
    s = cfg["structure"]
    a = atr(candles)
    swings = detect_swings(candles, int(s["swing_lookback"]), int(s["swing_confirmation_bars"]))
    bos = detect_bos(candles, swings, a, float(s["bos_min_displacement_atr"]))
    choch = detect_choch(candles, swings, a, float(s["bos_min_displacement_atr"]), bos)
    if choch:
        direction = choch.direction
    elif bos:
        direction = bos.direction
    else:
        direction = swing_direction(swings)
    return Structure(direction, swings, bos, choch)


# ============================================================ zone detection
def _candle_dir(c: Candle) -> int:
    if c.close > c.open:
        return 1
    if c.close < c.open:
        return -1
    return 0


def _scan_base(candles, start, a, cfg):
    z = cfg["zones"]
    body_lim = a * float(z["base_max_body_atr_ratio"])
    range_lim = a * float(z["base_max_range_atr_ratio"])
    max_candles = int(z["base_max_candles"])
    end = start
    count = 0
    while end < len(candles) and count < max_candles:
        c = candles[end]
        if abs(c.close - c.open) > body_lim or (c.high - c.low) > range_lim:
            break
        end += 1
        count += 1
    return (end - 1) if count > 0 else None


def detect_snd(candles: list[Candle], cfg: dict[str, Any], tf: str) -> list[Zone]:
    a = atr(candles)
    if a <= 0 or len(candles) < 10:
        return []
    z_cfg = cfg["zones"]
    disp_mult = float(z_cfg["displacement_atr_multiplier"])
    max_width = float(z_cfg["max_zone_width_atr_ratio"])
    zones: list[Zone] = []
    i = 1
    n = len(candles)
    while i < n - 2:
        end = _scan_base(candles, i, a, cfg)
        if end is None:
            i += 1
            continue
        if end + 1 >= n:
            break
        pre = candles[i - 1]
        post = candles[end + 1]
        if abs(post.close - post.open) < a * disp_mult:
            i = end + 1
            continue
        block = candles[i:end + 1]
        prox = max(c.high for c in block)
        dist = min(c.low for c in block)
        width = abs(prox - dist)
        if width <= 0 or width > a * max_width:
            i = end + 1
            continue
        pre_d, post_d = _candle_dir(pre), _candle_dir(post)
        pattern = None
        if pre_d == 1 and post_d == 1:
            pattern = Pattern.RBR
        elif pre_d == -1 and post_d == -1:
            pattern = Pattern.DBD
        elif pre_d == 1 and post_d == -1:
            pattern = Pattern.RBD
        elif pre_d == -1 and post_d == 1:
            pattern = Pattern.DBR
        if pattern is None:
            i = end + 1
            continue
        kind = ZoneKind.DEMAND if pattern in (Pattern.RBR, Pattern.DBR) else ZoneKind.SUPPLY
        supports = Action.BUY if kind == ZoneKind.DEMAND else Action.SELL
        disp = abs(post.close - post.open) / a
        quality = min(100.0, min(40.0, disp * 20) + max(0, 30 - (end - i) * 5)
                      + max(0, 20 - (width / a) * 10) + 10)
        zones.append(Zone(
            kind=kind, pattern=pattern, timeframe=tf,
            proximal=prox, distal=dist, created_at=post.open_time,
            quality=round(quality, 2), supports=supports,
        ))
        i = end + 2
    return update_zone_states(zones, candles)


def detect_snr(candles: list[Candle], swings: list[Swing], cfg: dict[str, Any], tf: str) -> list[Zone]:
    a = atr(candles)
    if a <= 0 or not swings:
        return []
    tol = a * float(cfg["snr"]["level_tolerance_atr_ratio"])
    min_touch = int(cfg["snr"]["min_touches"])
    max_width = a * float(cfg["zones"]["max_zone_width_atr_ratio"])
    clusters: list[list[Swing]] = []
    for s in swings:
        placed = False
        for cl in clusters:
            if abs(s.price - cl[-1].price) <= tol:
                cl.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    out: list[Zone] = []
    last_price = candles[-1].close
    for cl in clusters:
        if len(cl) < min_touch:
            continue
        prices = [s.price for s in cl]
        prox, dist = max(prices), min(prices)
        if prox - dist > max_width:
            continue
        if last_price < prox:
            kind, supports = ZoneKind.RESISTANCE, Action.SELL
        else:
            kind, supports = ZoneKind.SUPPORT, Action.BUY
        out.append(Zone(
            kind=kind, pattern=Pattern.AREA, timeframe=tf,
            proximal=prox, distal=dist, created_at=candles[-1].open_time,
            quality=min(100.0, 40 + len(cl) * 10), supports=supports,
        ))
    return update_zone_states(out, candles)


def update_zone_states(zones: list[Zone], candles: list[Candle]) -> list[Zone]:
    result: list[Zone] = []
    for z in zones:
        state = ZoneState.FRESH
        touches = 0
        invalid = False
        for c in candles:
            if c.open_time < z.created_at:
                continue
            if z.kind in (ZoneKind.DEMAND, ZoneKind.SUPPORT):
                if c.close < z.distal:
                    invalid = True
                    break
                if c.low <= z.proximal and c.high >= z.distal:
                    touches += 1
                    state = ZoneState.TESTED
            else:
                if c.close > z.proximal:
                    invalid = True
                    break
                if c.high >= z.distal and c.low <= z.proximal:
                    touches += 1
                    state = ZoneState.TESTED
        if invalid:
            continue
        z.state = state
        z.touches = touches
        result.append(z)
    return result


def pick_zone(zones: list[Zone], direction: Direction, cfg: dict[str, Any]) -> Zone | None:
    if direction == Direction.NEUTRAL:
        return None
    target = Action.BUY if direction == Direction.BULLISH else Action.SELL
    pool = [z for z in zones if z.supports == target]
    pool.sort(key=lambda z: z.quality, reverse=True)
    limit = int(cfg["zones"]["max_active_zones_per_direction"])
    return pool[0] if pool else None


def price_in_zone(price: float, z: Zone) -> bool:
    return z.low <= price <= z.high


# ============================================================ confirmation
def confirm_entry(candles: list[Candle], zone: Zone, direction: Direction, cfg: dict[str, Any]) -> tuple[bool, list[str], int]:
    if direction == Direction.NEUTRAL or len(candles) < 3:
        return False, ["data tidak cukup"], 0
    a = atr(candles)
    if a <= 0:
        return False, ["ATR nol"], 0
    action = Action.BUY if direction == Direction.BULLISH else Action.SELL
    last, prev = candles[-1], candles[-2]
    if not (last.low <= zone.proximal and last.high >= zone.distal):
        return False, ["harga belum di zona"], 0

    weights = cfg["confirmation"]["weights"]
    min_score = int(cfg["confirmation"]["min_score"])
    hits: list[str] = []
    score = 0
    strong = False

    def add(name: str, text: str) -> None:
        nonlocal score, strong
        w = int(weights.get(name, 0))
        if w <= 0:
            return
        score += w
        hits.append(text)
        if w >= 2:
            strong = True

    # rejection candle
    body = abs(last.close - last.open)
    rng = last.high - last.low
    if body > 0 and rng > 0:
        if action == Action.BUY:
            lower_wick = min(last.open, last.close) - last.low
            if lower_wick >= body * 1.5 and last.close > last.open:
                add("rejection_candle", "Rejection candle bullish")
        else:
            upper_wick = last.high - max(last.open, last.close)
            if upper_wick >= body * 1.5 and last.close < last.open:
                add("rejection_candle", "Rejection candle bearish")

    # pin bar
    if body > 0 and rng > 0 and body <= rng * 0.35:
        if action == Action.BUY:
            lower_wick = min(last.open, last.close) - last.low
            if lower_wick >= rng * 0.6:
                add("pin_bar", "Pin bar bullish")
        else:
            upper_wick = last.high - max(last.open, last.close)
            if upper_wick >= rng * 0.6:
                add("pin_bar", "Pin bar bearish")

    # engulfing
    if action == Action.BUY:
        if (prev.close < prev.open and last.close > last.open
                and last.close >= prev.open and last.open <= prev.close):
            add("engulfing", "Bullish engulfing")
    else:
        if (prev.close > prev.open and last.close < last.open
                and last.close <= prev.open and last.open >= prev.close):
            add("engulfing", "Bearish engulfing")

    # strong displacement
    if body >= a * 1.2:
        if (action == Action.BUY and last.close > last.open) or \
           (action == Action.SELL and last.close < last.open):
            add("strong_displacement", "Displacement kuat")

    # minor BOS
    window = candles[-4:-1]
    if len(window) == 3:
        if action == Action.BUY and last.close > max(c.high for c in window):
            add("minor_bos", "Minor BOS bullish")
        if action == Action.SELL and last.close < min(c.low for c in window):
            add("minor_bos", "Minor BOS bearish")

    # retest breakout
    if prev.low <= zone.proximal and prev.high >= zone.distal:
        if action == Action.BUY and last.close > zone.proximal and last.close > last.open:
            add("retest_breakout", "Retest bullish breakout")
        if action == Action.SELL and last.close < zone.distal and last.close < last.open:
            add("retest_breakout", "Retest bearish breakout")

    # momentum
    last3 = candles[-3:]
    avg_body = sum(abs(c.close - c.open) for c in candles[-20:]) / max(1, len(candles[-20:]))
    if avg_body > 0:
        if action == Action.BUY and all(c.close > c.open and abs(c.close - c.open) >= avg_body for c in last3):
            add("momentum_candle", "Momentum bullish")
        if action == Action.SELL and all(c.close < c.open and abs(c.close - c.open) >= avg_body for c in last3):
            add("momentum_candle", "Momentum bearish")

    ok = score >= min_score and strong
    return ok, hits, score


# ============================================================ risk
def compute_signal(direction: Direction, zone: Zone, entry: float, a: float,
                   swings: list[Swing], cfg: dict[str, Any], market_cfg: dict[str, Any],
                   confirmation_tf: str) -> Signal | None:
    if direction == Direction.NEUTRAL or a <= 0:
        return None
    buffer = a * float(cfg["risk"]["sl_buffer_atr_ratio"])
    tp1_rr = float(cfg["risk"]["tp1_rr_default"])
    tp2_rr = float(cfg["risk"]["tp2_rr_default"])
    min_rr = float(market_cfg["min_rr"])
    precision = int(market_cfg["precision"])

    if direction == Direction.BULLISH:
        action = Action.BUY
        cands = [zone.distal]
        lows = [s.price for s in swings if not s.is_high]
        if lows:
            cands.append(lows[-1])
        sl = min(cands) - buffer
        if sl >= entry:
            return None
        risk = entry - sl
        tp1 = entry + risk * tp1_rr
        tp2 = entry + risk * tp2_rr
    else:
        action = Action.SELL
        cands = [zone.proximal]
        highs = [s.price for s in swings if s.is_high]
        if highs:
            cands.append(highs[-1])
        sl = max(cands) + buffer
        if sl <= entry:
            return None
        risk = sl - entry
        tp1 = entry - risk * tp1_rr
        tp2 = entry - risk * tp2_rr

    if risk <= 0:
        return None
    rr = abs(tp1 - entry) / risk
    if rr < min_rr:
        return None

    def r(p: float) -> float:
        return round(p, precision)

    return Signal(
        action=action, entry=r(entry), sl=r(sl), tp1=r(tp1), tp2=r(tp2),
        rr=round(rr, 2),
        zone_text=f"{zone.kind.value} {zone.timeframe} {r(zone.distal)}–{r(zone.proximal)}",
        confirmation_tf=confirmation_tf,
        reasons=[f"SL invalidasi zona + buffer ATR", f"TP1 {tp1_rr}R, TP2 {tp2_rr}R"],
    )


# ============================================================ orchestrator
class Orchestrator:
    def __init__(self, cfg: dict[str, Any], client: TwelveData, cache: Cache):
        self._cfg = cfg
        self._client = client
        self._cache = cache

    async def _candles(self, symbol: str, interval: str) -> list[Candle]:
        cached = self._cache.get(symbol, interval)
        if cached:
            return cached
        raw = await self._client.fetch(symbol, interval)
        min_req = int(self._cfg["bot"]["min_candles_required"])
        clean = clean_candles(raw, interval, min_req)
        self._cache.set(symbol, interval, clean)
        return clean

    async def analyze(self, market: str, method: str) -> Analysis:
        cfg = self._cfg
        result = Analysis(market=market, method=method, direction_tf="")
        try:
            if market not in cfg["markets"]:
                return self._fail(result, f"Market tidak dikenal: {market}")
            if method not in cfg["methods"]:
                return self._fail(result, f"Metode tidak dikenal: {method}")

            m_cfg = cfg["markets"][market]
            method_cfg = cfg["methods"][method]
            symbol = m_cfg["symbol"]
            dir_tf = method_cfg["direction_tf"]
            zone_tf = method_cfg["zone_tf"]
            conf_tf = method_cfg["confirmation_tf"]
            result.direction_tf = dir_tf
            result.confirmation_tf = conf_tf

            dir_candles = await self._candles(symbol, dir_tf)
            zone_candles = dir_candles if zone_tf == dir_tf else await self._candles(symbol, zone_tf)
            conf_candles = (
                zone_candles if conf_tf == zone_tf
                else dir_candles if conf_tf == dir_tf
                else await self._candles(symbol, conf_tf)
            )
            result.last_close = conf_candles[-1].open_time

            structure_dir = analyze_structure(dir_candles, cfg)
            result.direction = structure_dir.direction
            result.confirmation_dir = analyze_structure(conf_candles, cfg).direction

            if structure_dir.direction == Direction.NEUTRAL:
                return self._set(result, Status.NO_TRADE,
                                 f"Arah {dir_tf} NEUTRAL — tidak ada BOS/CHoCH valid")

            snd = detect_snd(zone_candles, cfg, zone_tf)
            snr = detect_snr(zone_candles, structure_dir.swings, cfg, zone_tf)
            all_zones = snd + snr
            result.zones = all_zones

            zone = pick_zone(all_zones, structure_dir.direction, cfg)
            if zone is None:
                return self._set(result, Status.NO_TRADE,
                                 f"Tidak ada zona valid searah {structure_dir.direction.value}")

            price = zone_candles[-1].close
            if not price_in_zone(price, zone):
                return self._set(result, Status.WAITING_ZONE,
                                 f"Harga belum memasuki zona {zone.kind.value} {zone.timeframe}")

            ok, hits, score = confirm_entry(conf_candles, zone, structure_dir.direction, cfg)
            if not ok:
                reasons = ["Harga di zona, menunggu konfirmasi"]
                if hits:
                    reasons.extend(hits)
                return self._set(result, Status.WAITING_CONFIRMATION, *reasons)

            a = atr(zone_candles)
            sig = compute_signal(structure_dir.direction, zone, conf_candles[-1].close,
                                 a, structure_dir.swings, cfg, m_cfg, conf_tf)
            if sig is None:
                return self._set(result, Status.NO_TRADE,
                                 "Risk/Reward tidak memenuhi minimum")

            result.signal = sig
            status = Status.CONFIRMED_BUY if sig.action == Action.BUY else Status.CONFIRMED_SELL
            return self._set(result, status,
                             f"Arah {dir_tf} {structure_dir.direction.value}",
                             f"Zona {zone.kind.value} {zone.timeframe}",
                             f"Konfirmasi {conf_tf}: " + ", ".join(hits),
                             f"RR {sig.rr}")

        except APIError as e:
            return self._fail(result, f"API error: {e}")
        except Exception as e:
            logger.exception("Analyzer error")
            return self._fail(result, f"Kesalahan: {e}")

    @staticmethod
    def _set(r: Analysis, status: Status, *reasons: str) -> Analysis:
        r.status = status
        r.reasons = list(reasons)
        return r

    @staticmethod
    def _fail(r: Analysis, msg: str) -> Analysis:
        r.status = Status.INVALIDATED
        r.reasons = [msg]
        r.error = msg
        return r


# ============================================================ formatting
def format_analysis(a: Analysis, cfg: dict[str, Any]) -> str:
    m_disp = cfg["markets"][a.market]["display"] if a.market in cfg["markets"] else a.market
    m_label = cfg["methods"][a.method]["label"] if a.method in cfg["methods"] else a.method.upper()
    m_prec = int(cfg["markets"][a.market]["precision"]) if a.market in cfg["markets"] else 2

    def fmt_price(p: float) -> str:
        return f"{p:.{m_prec}f}"

    lines = [f"📊 {m_disp} — {m_label}", f"🕐 {fmt_wib(a.timestamp)}"]
    if a.last_close:
        lines.append(f"🕯️ Candle close terakhir: {fmt_wib(a.last_close)}")
    lines.append("")
    lines.append(f"Arah Market: {a.direction.value}")
    lines.append("Timeframe:")
    lines.append(f"- {a.direction_tf}: {a.direction.value}")
    if a.confirmation_dir:
        lines.append(f"- {a.confirmation_tf}: {a.confirmation_dir.value}")
    lines.append("")

    if a.zones:
        lines.append("Zona Aktif:")
        for z in a.zones[:3]:
            lines.append(f"- {z.kind.value} {z.timeframe}: {fmt_price(z.distal)} – {fmt_price(z.proximal)}")
            q = "HIGH" if z.quality >= 75 else "MEDIUM" if z.quality >= 50 else "LOW"
            lines.append(f"  Status: {z.state.value} | Quality: {q}")
        lines.append("")

    lines.append(f"Status Analisa:\n{a.status.value}")
    if a.reasons:
        lines.append("")
        lines.append("Alasan:")
        for r in a.reasons:
            lines.append(f"- {r}")

    lines.append("")
    if a.signal:
        s = a.signal
        lines.append(f"Entry: {fmt_price(s.entry)}")
        lines.append(f"Stop Loss: {fmt_price(s.sl)}")
        lines.append(f"Take Profit 1: {fmt_price(s.tp1)}")
        lines.append(f"Take Profit 2: {fmt_price(s.tp2)}")
        lines.append(f"Risk/Reward: {s.rr}")
    else:
        lines.append("Entry: —")
        lines.append("Stop Loss: —")
        lines.append("Take Profit: —")

    return "\n".join(lines)


# ============================================================ keyboards
def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("XAUUSD", callback_data="mkt:XAUUSD")],
        [InlineKeyboardButton("BTCUSD", callback_data="mkt:BTCUSD")],
    ])


def kb_method() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("SCALPING H1 + M15", callback_data="mth:scalping")],
        [InlineKeyboardButton("ENTRY DAY H4 + H1 + M5", callback_data="mth:entry_day")],
        [InlineKeyboardButton("🏠 Menu Utama", callback_data="back:main")],
    ])


def kb_result() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh Analisa", callback_data="rf")],
        [InlineKeyboardButton("🔁 Ganti Metode", callback_data="back:method")],
        [InlineKeyboardButton("📊 Ganti Market", callback_data="back:market")],
        [InlineKeyboardButton("🏠 Menu Utama", callback_data="back:main")],
    ])


# ============================================================ handlers
_ALLOWED_IDS: set[int] = set()
_COOLDOWN: dict[int, float] = {}
_LOCKS: dict[tuple[int, str, str], asyncio.Lock] = {}


def _load_allowed() -> set[int]:
    raw = os.getenv("ALLOWED_USER_IDS", "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


def _allowed(user_id: int) -> bool:
    return user_id in _ALLOWED_IDS


def _cooldown_ok(chat_id: int, cfg: dict[str, Any]) -> tuple[bool, int]:
    window = int(cfg["bot"]["refresh_cooldown_seconds"])
    now = time.time()
    remain = int(window - (now - _COOLDOWN.get(chat_id, 0)))
    if remain > 0:
        return False, remain
    _COOLDOWN[chat_id] = now
    return True, 0


def _lock(chat_id: int, market: str, method: str) -> asyncio.Lock:
    key = (chat_id, market, method)
    if key not in _LOCKS:
        _LOCKS[key] = asyncio.Lock()
    return _LOCKS[key]


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not _allowed(user.id):
        if update.effective_message:
            await update.effective_message.reply_text("⛔ Akses ditolak.")
        return
    await update.effective_message.reply_text(
        "Pilih market:", reply_markup=kb_main(),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not _allowed(user.id):
        return
    await update.effective_message.reply_text(
        "Bot SNR/SND multi-timeframe.\n/start untuk membuka menu."
    )


async def on_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    user = update.effective_user
    if not q:
        return
    if not user or not _allowed(user.id):
        await q.answer("Akses ditolak", show_alert=True)
        return
    await q.answer()
    market = q.data.split(":", 1)[1]
    ctx.user_data["market"] = market
    await q.edit_message_text(
        f"Market: {market}\nPilih metode analisa:",
        reply_markup=kb_method(),
    )


async def on_method(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    user = update.effective_user
    if not q:
        return
    if not user or not _allowed(user.id):
        await q.answer("Akses ditolak", show_alert=True)
        return
    await q.answer()
    method = q.data.split(":", 1)[1]
    market = ctx.user_data.get("market")
    if not market:
        await q.edit_message_text("Pilih market dulu:", reply_markup=kb_main())
        return
    ctx.user_data["method"] = method
    ok, remain = _cooldown_ok(q.message.chat_id, ctx.bot_data["config"])
    if not ok:
        await q.answer(f"Tunggu {remain} detik.", show_alert=True)
        return
    await q.edit_message_text("ANALYZING...")
    orch: Orchestrator = ctx.bot_data["orch"]
    lk = _lock(q.message.chat_id, market, method)
    if lk.locked():
        await q.edit_message_text("Analisa sedang berjalan, tunggu...")
        return
    async with lk:
        result = await orch.analyze(market, method)
    text = format_analysis(result, ctx.bot_data["config"])
    await ctx.bot.send_message(chat_id=q.message.chat_id, text=text, reply_markup=kb_result())


async def on_refresh(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    user = update.effective_user
    if not q:
        return
    if not user or not _allowed(user.id):
        await q.answer("Akses ditolak", show_alert=True)
        return
    market = ctx.user_data.get("market")
    method = ctx.user_data.get("method")
    if not market or not method:
        await q.answer("Pilih market & metode dulu.", show_alert=True)
        return
    ok, remain = _cooldown_ok(q.message.chat_id, ctx.bot_data["config"])
    if not ok:
        await q.answer(f"Tunggu {remain} detik.", show_alert=True)
        return
    await q.answer("Refresh...")
    orch: Orchestrator = ctx.bot_data["orch"]
    lk = _lock(q.message.chat_id, market, method)
    if lk.locked():
        return
    async with lk:
        result = await orch.analyze(market, method)
    text = format_analysis(result, ctx.bot_data["config"])
    try:
        await q.edit_message_text(text=text, reply_markup=kb_result())
    except Exception:
        await ctx.bot.send_message(chat_id=q.message.chat_id, text=text, reply_markup=kb_result())


async def on_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    user = update.effective_user
    if not q:
        return
    if not user or not _allowed(user.id):
        await q.answer("Akses ditolak", show_alert=True)
        return
    await q.answer()
    target = q.data.split(":", 1)[1]
    if target == "main" or target == "market":
        await q.edit_message_text("Pilih market:", reply_markup=kb_main())
    elif target == "method":
        market = ctx.user_data.get("market", "-")
        await q.edit_message_text(f"Market: {market}\nPilih metode analisa:",
                                  reply_markup=kb_method())


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Handler error: %s", ctx.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                f"⚠️ Terjadi kesalahan: {ctx.error}"
            )
        except Exception:
            pass


# ============================================================ main
def main() -> None:
    load_dotenv()
    global _ALLOWED_IDS
    _ALLOWED_IDS = _load_allowed()
    if not _ALLOWED_IDS:
        logger.warning("ALLOWED_USER_IDS kosong — bot akan menolak semua akses.")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN belum di-set")
    if not api_key:
        raise SystemExit("TWELVEDATA_API_KEY belum di-set")

    cfg = load_config()
    client = TwelveData(api_key, cfg)
    cache = Cache(cfg)
    orch = Orchestrator(cfg, client, cache)

    app = Application.builder().token(token).build()
    app.bot_data["config"] = cfg
    app.bot_data["orch"] = orch

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_market, pattern=r"^mkt:"))
    app.add_handler(CallbackQueryHandler(on_method, pattern=r"^mth:"))
    app.add_handler(CallbackQueryHandler(on_refresh, pattern=r"^rf$"))
    app.add_handler(CallbackQueryHandler(on_back, pattern=r"^back:"))
    app.add_error_handler(on_error)

    logger.info("Bot start (polling).")
    app.run_polling(drop_pending_updates=True, allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
