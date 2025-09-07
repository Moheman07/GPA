#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Analyzer — Unified Full Powerful Script (updated with TA-Lib attempt, better news filter,
USD ticker logging, improved divergence detection)
Output: gold_analysis_unified_full.json
"""
import os
import sys
import math
import time
import json
import logging
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

# Optional TA-Lib
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

# For peak detection in divergence logic
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("gold-unified")

# API keys from env
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

def clean_scalar(x):
    try:
        if x is None:
            return None
        if isinstance(x, (np.generic,)):
            x = x.item()
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x
    except Exception:
        return None

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return clean_scalar(obj)

def safe_pct_change(series: pd.Series) -> pd.Series:
    try:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return pd.Series(dtype=float)

# ------------- Fetch helpers -------------
def fetch_yfinance(symbol: str, period: str = "1y", tries: int = 3, pause: float = 1.0) -> pd.DataFrame:
    import yfinance as yf
    last_err = None
    for i in range(tries):
        try:
            log.info(f"Fetching {symbol} (attempt {i+1}/{tries}) ...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, auto_adjust=False)
            if df is None or df.empty:
                last_err = Exception("Empty dataframe")
                time.sleep(pause)
                continue
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    raise ValueError(f"Missing column {col} from {symbol}")
            df = df.dropna(how="all")
            log.info(f"Fetched {len(df)} rows for {symbol}")
            return df
        except Exception as e:
            last_err = e
            log.warning(f"Fetch attempt {i+1} failed for {symbol}: {e}")
            time.sleep(pause)
    log.error(f"Failed to fetch {symbol}: {last_err}")
    return pd.DataFrame()

def fetch_market_universe(period: str = "1y") -> Dict[str, Any]:
    """
    Returns dict with market data plus used_symbols mapping
    Keys: "GC=F", "^DXY", "SPY"
    Also includes 'used_symbols' -> which symbol was used for USD index
    """
    candidates = {
        "GC=F": ["GC=F"],
        "^DXY": ["^DXY", "DX-Y.NYB", "DXY", "DX=F", "USDX"],
        "SPY": ["SPY"]
    }
    out: Dict[str, Any] = {"used_symbols": {}}
    for out_key, tries in candidates.items():
        fetched = pd.DataFrame()
        chosen = None
        for sym in tries:
            df = fetch_yfinance(sym, period=period)
            if df is not None and not df.empty:
                fetched = df
                chosen = sym
                out[out_key] = df
                out["used_symbols"][out_key] = sym
                log.info(f"Using symbol '{sym}' for key '{out_key}'")
                break
            else:
                log.debug(f"No data for symbol {sym} (key {out_key})")
        if fetched.empty:
            log.warning(f"No data found for any of {tries} (key={out_key})")
            out[out_key] = pd.DataFrame()
            out["used_symbols"][out_key] = None
    return out

# ------------- Technicals -------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def technical_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    ind: Dict[str, pd.Series] = {}
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    # trend
    if TALIB_AVAILABLE:
        ind["sma_20"] = talib.SMA(close, timeperiod=20)
        ind["sma_50"] = talib.SMA(close, timeperiod=50)
        ind["sma_200"] = talib.SMA(close, timeperiod=200)
        ind["ema_12"] = talib.EMA(close, timeperiod=12)
        ind["ema_26"] = talib.EMA(close, timeperiod=26)
    else:
        ind["sma_20"] = close.rolling(20).mean()
        ind["sma_50"] = close.rolling(50).mean()
        ind["sma_200"] = close.rolling(200).mean()
        ind["ema_12"] = ema(close, 12)
        ind["ema_26"] = ema(close, 26)
    # momentum
    if TALIB_AVAILABLE:
        ind["rsi"] = talib.RSI(close, timeperiod=14)
        macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd, macd_sig, macd_hist
        ind["stoch_k"], ind["stoch_d"] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        ind["williams_r"] = talib.WILLR(high, low, close, timeperiod=14)
        ind["cci"] = talib.CCI(high, low, close, timeperiod=14)
        ind["adx"] = talib.ADX(high, low, close, timeperiod=14)
        ind["trix"] = talib.TRIX(close, timeperiod=30)
        ind["ultosc"] = talib.ULTOSC(high, low, close)
    else:
        ind["rsi"] = calc_rsi(close, 14)
        macd, macd_sig, macd_hist = calc_macd(close)
        ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd, macd_sig, macd_hist
    # volatility / bands
    try:
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            ind["bb_upper"], ind["bb_middle"], ind["bb_lower"] = upper, middle, lower
            ind["atr"] = talib.ATR(high, low, close, timeperiod=14)
            ind["sar"] = talib.SAR(high, low)
        else:
            m = close.rolling(20).mean()
            s = close.rolling(20).std()
            ind["bb_upper"] = m + 2 * s
            ind["bb_middle"] = m
            ind["bb_lower"] = m - 2 * s
            tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            ind["atr"] = tr.rolling(14).mean()
    except Exception as e:
        log.warning(f"BB/ATR calc error: {e}")
    # volume-based
    try:
        if TALIB_AVAILABLE:
            ind["obv"] = talib.OBV(close, vol)
            ind["ad"] = talib.AD(high, low, close, vol)
            ind["adosc"] = talib.ADOSC(high, low, close, vol)
            ind["mfi"] = talib.MFI(high, low, close, vol, timeperiod=14)
        else:
            obv = (np.sign(close.diff().fillna(0)) * vol).cumsum()
            ind["obv"] = obv
            ind["ad"] = None
            ind["adosc"] = None
            ind["mfi"] = None
    except Exception as e:
        log.warning(f"Volume indicators error: {e}")
    return ind

# ------------- Patterns & Candles -------------
def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    if not TALIB_AVAILABLE:
        return {"note": "TA-Lib not available - candlestick patterns skipped"}
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    patterns = {
        "doji": talib.CDLDOJI(o, h, l, c),
        "hammer": talib.CDLHAMMER(o, h, l, c),
        "shooting_star": talib.CDLSHOOTINGSTAR(o, h, l, c),
        "engulfing": talib.CDLENGULFING(o, h, l, c),
        "morning_star": talib.CDLMORNINGSTAR(o, h, l, c),
        "evening_star": talib.CDLEVENINGSTAR(o, h, l, c),
    }
    return patterns

def detect_price_patterns(df: pd.DataFrame) -> Dict[str, List[int]]:
    highs = df["High"].rolling(window=5, center=True).max()
    lows = df["Low"].rolling(window=5, center=True).min()
    double_tops, double_bottoms = [], []
    length = len(df)
    for i in range(20, max(20, length - 20)):
        try:
            if highs.iloc[i] == df["High"].iloc[i] and abs(df["High"].iloc[i] - df["High"].iloc[i-20:i].max()) < 1e-6:
                double_tops.append(i)
            if lows.iloc[i] == df["Low"].iloc[i] and abs(df["Low"].iloc[i] - df["Low"].iloc[i-20:i].min()) < 1e-6:
                double_bottoms.append(i)
        except Exception:
            continue
    return {"double_top": double_tops, "double_bottom": double_bottoms, "head_shoulders": [], "triangle": []}

# ------------- Enhancements -------------
def fibonacci_levels(low: float, high: float) -> Dict[str, Dict[str, float]]:
    diff = high - low
    levels = {
        "0.0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.0": high
    }
    extensions = {
        "1.272": high + 0.272 * diff,
        "1.618": high + 0.618 * diff,
        "2.0": high + 1.0 * diff,
        "2.618": high + 1.618 * diff
    }
    return {"levels": levels, "extensions": extensions}

def cluster_levels(levels: List[float], tol: float = 0.01) -> List[float]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters = []
    current = [levels[0]]
    for lev in levels[1:]:
        if abs(lev - current[-1]) / (current[-1] or 1) <= tol:
            current.append(lev)
        else:
            clusters.append(float(np.mean(current)))
            current = [lev]
    if current:
        clusters.append(float(np.mean(current)))
    return clusters

def support_resistance(df: pd.DataFrame) -> Dict[str, Any]:
    result = {}
    try:
        n = len(df)
        supports, resistances = [], []
        for i in range(5, n-5):
            if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["Low"].iloc[i] < df["Low"].iloc[i+1]:
                supports.append(float(df["Low"].iloc[i]))
            if df["High"].iloc[i] > df["High"].iloc[i-1] and df["High"].iloc[i] > df["High"].iloc[i+1]:
                resistances.append(float(df["High"].iloc[i]))
        clustered_support = cluster_levels(supports)
        clustered_resistance = cluster_levels(resistances)
        recent = df.tail(20)
        pivot = (recent["High"].mean() + recent["Low"].mean() + recent["Close"].mean()) / 3.0
        r1 = 2 * pivot - recent["Low"].mean()
        r2 = pivot + (recent["High"].mean() - recent["Low"].mean())
        s1 = 2 * pivot - recent["High"].mean()
        s2 = pivot - (recent["High"].mean() - recent["Low"].mean())
        result = {
            "pivot_points": {"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2},
            "historical_support": clustered_support,
            "historical_resistance": clustered_resistance
        }
    except Exception as e:
        log.warning(f"support_resistance error: {e}")
        result = {"error": str(e)}
    return result

def volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
    try:
        close = df["Close"]
        vol = df["Volume"].astype(float)
        cats = pd.cut(close, bins=bins)
        vp = vol.groupby(cats).sum().sort_values(ascending=False).head(6)
        profile = {str(k): int(v) for k, v in vp.items()}
        price_change = safe_pct_change(close)
        vol_aligned = vol.reindex(price_change.index)
        try:
            corr = float(np.corrcoef(price_change.values, vol_aligned.values)[0,1])
            if math.isnan(corr):
                corr = None
        except Exception:
            corr = None
        up_vol = float(vol_aligned[price_change > 0].mean()) if not price_change.empty else None
        down_vol = float(vol_aligned[price_change < 0].mean()) if not price_change.empty else None
        trend = "مرتفع" if vol.iloc[-1] > vol.rolling(20).mean().iloc[-1] * 1.5 else ("منخفض" if vol.iloc[-1] < vol.rolling(20).mean().iloc[-1] * 0.5 else "عادي")
        return {"current_volume": int(vol.iloc[-1]), "average_volume": float(vol.mean()), "volume_ratio": float(vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] or 1)), "volume_profile": profile, "price_volume_correlation": corr, "uptrend_volume": up_vol, "downtrend_volume": down_vol, "volume_trend": trend}
    except Exception as e:
        log.warning(f"volume_profile error: {e}")
        return {"error": str(e)}

def market_structure(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict[str, Any]:
    try:
        close = df["Close"]
        sma20 = ind.get("sma_20", pd.Series(dtype=float))
        sma50 = ind.get("sma_50", pd.Series(dtype=float))
        sma200 = ind.get("sma_200", pd.Series(dtype=float))
        direction = "متذبذب"
        if not sma20.empty and not sma50.empty and not sma200.empty:
            if close.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]:
                direction = "اتجاه صاعد"
            elif close.iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1]:
                direction = "اتجاه هابط"
        peaks_count = int(((close.shift(1) < close) & (close.shift(-1) < close)).sum())
        troughs_count = int(((close.shift(1) > close) & (close.shift(-1) > close)).sum())
        rel_strength = float((sma20.iloc[-1] / sma50.iloc[-1]) if (not sma20.empty and not sma50.empty and sma50.iloc[-1] != 0) else 1.0)
        conf = "عالية" if direction != "متذبذب" else "متوسطة"
        return {"market_structure": direction, "peaks_count": peaks_count, "troughs_count": troughs_count, "relative_strength": rel_strength, "structure_confidence": conf}
    except Exception as e:
        log.warning(f"market_structure error: {e}")
        return {"error": str(e)}

# ------------- Improved divergence detection -------------
def detect_divergences(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict[str, Any]:
    """
    Use peak/trough detection on price and compare corresponding indicator peaks
    to identify classic bullish/bearish divergences for RSI and MACD.
    Returns counts, recent list (dates & type).
    """
    try:
        close = df["Close"].values
        dates = df.index.to_numpy()
        window_min_prom = 1  # minimal prominence for find_peaks (small)
        divergences = []
        # Use scipy.find_peaks if available, else fallback to simple local-extrema method
        if SCIPY_AVAILABLE:
            # detect peaks and troughs
            peaks_idx, _ = find_peaks(close, prominence=window_min_prom)
            troughs_idx, _ = find_peaks(-close, prominence=window_min_prom)
        else:
            # simple deterministic local maxima/minima:
            peaks_idx = [i for i in range(1, len(close)-1) if close[i] > close[i-1] and close[i] > close[i+1]]
            troughs_idx = [i for i in range(1, len(close)-1) if close[i] < close[i-1] and close[i] < close[i+1]]
        # indicators
        rsi = ind.get("rsi", pd.Series(dtype=float)).values
        macd = ind.get("macd", pd.Series(dtype=float)).values
        def sample_at(idx_list, arr):
            out = []
            for i in idx_list:
                if 0 <= i < len(arr):
                    out.append((i, float(arr[i]) if not math.isnan(arr[i]) else None))
            return out
        # check peaks (bearish divergence: price makes higher high, indicator makes lower high)
        peak_pairs = []
        # consider adjacent peaks pairs
        p_samples = sample_at(peaks_idx, close)
        for j in range(1, len(p_samples)):
            i0, p0 = p_samples[j-1]
            i1, p1 = p_samples[j]
            if p0 is None or p1 is None: continue
            # price higher high
            if p1 > p0:
                # RSI at those indexes
                r0 = rsi[i0] if i0 < len(rsi) else None
                r1 = rsi[i1] if i1 < len(rsi) else None
                if r0 is not None and r1 is not None and r1 < r0:
                    divergences.append({"type":"bearish","indicator":"RSI","price_idx":int(i1),"price_date":str(dates[i1]),"price":float(p1),"indicator_prev":float(r0),"indicator_now":float(r1)})
                # MACD
                m0 = macd[i0] if i0 < len(macd) else None
                m1 = macd[i1] if i1 < len(macd) else None
                if m0 is not None and m1 is not None and m1 < m0:
                    divergences.append({"type":"bearish","indicator":"MACD","price_idx":int(i1),"price_date":str(dates[i1]),"price":float(p1),"indicator_prev":float(m0),"indicator_now":float(m1)})
        # check trough pairs (bullish divergence: price lower low, indicator higher low)
        t_samples = sample_at(troughs_idx, close)
        for j in range(1, len(t_samples)):
            i0, p0 = t_samples[j-1]
            i1, p1 = t_samples[j]
            if p0 is None or p1 is None: continue
            if p1 < p0:
                r0 = rsi[i0] if i0 < len(rsi) else None
                r1 = rsi[i1] if i1 < len(rsi) else None
                if r0 is not None and r1 is not None and r1 > r0:
                    divergences.append({"type":"bullish","indicator":"RSI","price_idx":int(i1),"price_date":str(dates[i1]),"price":float(p1),"indicator_prev":float(r0),"indicator_now":float(r1)})
                m0 = macd[i0] if i0 < len(macd) else None
                m1 = macd[i1] if i1 < len(macd) else None
                if m0 is not None and m1 is not None and m1 > m0:
                    divergences.append({"type":"bullish","indicator":"MACD","price_idx":int(i1),"price_date":str(dates[i1]),"price":float(p1),"indicator_prev":float(m0),"indicator_now":float(m1)})
        # summarise
        recent = sorted(divergences, key=lambda x: x["price_idx"])[-10:]
        summary = {
            "total_divergences": len(divergences),
            "recent_divergences": len(recent),
            "positive_divergences": len([d for d in recent if d["type"] == "bullish"]),
            "negative_divergences": len([d for d in recent if d["type"] == "bearish"]),
            "latest_divergences": recent
        }
        return summary
    except Exception as e:
        log.warning(f"divergence error: {e}")
        return {"error": str(e)}

def correlation_block(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    try:
        gold = data.get("GC=F", pd.DataFrame())
        dxy = data.get("^DXY", pd.DataFrame())
        spy = data.get("SPY", pd.DataFrame())
        out = {}
        if gold is None or gold.empty:
            return {"asset_correlations": {}, "strongest_correlation": None, "weakest_correlation": None}
        gr = safe_pct_change(gold["Close"])
        if dxy is not None and not dxy.empty:
            xr = safe_pct_change(dxy["Close"])
            a, b = gr.align(xr, join="inner")
            if len(a) > 5:
                out["USD"] = float(np.corrcoef(a.values, b.values)[0,1])
        if spy is not None and not spy.empty:
            sr = safe_pct_change(spy["Close"])
            a, b = gr.align(sr, join="inner")
            if len(a) > 5:
                out["SPY"] = float(np.corrcoef(a.values, b.values)[0,1])
        if not out:
            return {"asset_correlations": {}, "strongest_correlation": None, "weakest_correlation": None}
        strongest = max(out.items(), key=lambda kv: abs(kv[1]))
        weakest = min(out.items(), key=lambda kv: abs(kv[1]))
        return {"asset_correlations": out, "strongest_correlation": list(strongest), "weakest_correlation": list(weakest)}
    except Exception as e:
        log.warning(f"correlation_block error: {e}")
        return {"error": str(e)}

def advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        close = df["Close"]
        ret = safe_pct_change(close)
        if ret.empty:
            return {}
        ann_ret = float((1 + ret.mean()) ** 252 - 1)
        ann_vol = float(ret.std() * np.sqrt(252))
        rf = 0.02
        sharpe = float(((ret.mean() - rf/252) / ret.std()) * np.sqrt(252)) if ret.std() != 0 else None
        equity = (1 + ret).cumprod()
        run_max = equity.cummax()
        drawdown = (equity - run_max) / run_max
        mdd = float(drawdown.min()) if not drawdown.empty else None
        downside = ret[ret < 0]
        sortino = float(((ret.mean() - rf/252) / downside.std()) * np.sqrt(252)) if (not downside.empty and downside.std() != 0) else None
        calmar = float(ann_ret / abs(mdd)) if (mdd not in (None, 0) and not math.isnan(mdd)) else None
        wins = ret[ret > 0]
        losses = ret[ret < 0]
        win_rate = float(len(wins) / len(ret)) if len(ret) else None
        avg_win = float(wins.mean()) if not wins.empty else None
        avg_loss = float(losses.mean()) if not losses.empty else None
        profit_factor = float(abs(wins.sum() / losses.sum())) if (not wins.empty and not losses.empty and losses.sum() != 0) else None
        return {"annualized_return": ann_ret, "annualized_volatility": ann_vol, "sharpe_ratio": sharpe, "max_drawdown": mdd, "var_95": float(np.percentile(ret, 5)), "var_99": float(np.percentile(ret, 1)), "skewness": float(ret.skew()), "kurtosis": float(ret.kurtosis()), "calmar_ratio": calmar, "sortino_ratio": sortino, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": profit_factor}
    except Exception as e:
        log.warning(f"advanced_metrics error: {e}")
        return {"error": str(e)}

# ------------- Sentiment & signals -------------
def technical_sentiment(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict[str, Any]:
    try:
        out = {}
        close = df["Close"].iloc[-1]
        rsi = ind.get("rsi", pd.Series(dtype=float))
        bb_upper = ind.get("bb_upper", pd.Series(dtype=float))
        bb_lower = ind.get("bb_lower", pd.Series(dtype=float))
        macd = ind.get("macd", pd.Series(dtype=float))
        macd_sig = ind.get("macd_signal", pd.Series(dtype=float))
        adx = ind.get("adx", pd.Series(dtype=float))
        vol_avg = df["Volume"].rolling(20).mean()
        rsi_now = float(rsi.iloc[-1]) if (not rsi.empty and not math.isnan(rsi.iloc[-1])) else 50.0
        out["rsi_sentiment"] = "مفرط في الشراء" if rsi_now > 70 else ("مفرط في البيع" if rsi_now < 30 else "محايد")
        if not macd.empty and not macd_sig.empty:
            out["macd_sentiment"] = "إيجابي" if macd.iloc[-1] >= macd_sig.iloc[-1] else "سلبي"
        if not bb_upper.empty and not bb_lower.empty:
            if close > bb_upper.iloc[-1]:
                out["bb_sentiment"] = "مفرط في الشراء"
            elif close < bb_lower.iloc[-1]:
                out["bb_sentiment"] = "مفرط في البيع"
            else:
                out["bb_sentiment"] = "عادي"
        if not adx.empty:
            out["trend_strength"] = "قوي" if adx.iloc[-1] > 25 else "ضعيف"
        v = int(df["Volume"].iloc[-1])
        if not vol_avg.empty and not math.isnan(vol_avg.iloc[-1]) and vol_avg.iloc[-1] > 0:
            out["volume_sentiment"] = "مرتفع" if v > 1.5 * vol_avg.iloc[-1] else ("منخفض" if v < 0.5 * vol_avg.iloc[-1] else "عادي")
        return out
    except Exception as e:
        log.warning(f"technical_sentiment error: {e}")
        return {"error": str(e)}

def generate_signals(df: pd.DataFrame, ind: Dict[str, pd.Series], patterns: Dict[str, Any]) -> Dict[str, Any]:
    try:
        close_price = float(df["Close"].iloc[-1])
        out: Dict[str, Any] = {"current_price": close_price, "timestamp": dt.datetime.utcnow().isoformat()}
        sma20 = ind.get("sma_20", pd.Series(dtype=float))
        sma50 = ind.get("sma_50", pd.Series(dtype=float))
        sma200 = ind.get("sma_200", pd.Series(dtype=float))
        if not sma20.empty and not sma50.empty and not sma200.empty:
            if close_price > sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]:
                out["trend"] = "صاعد قوي"
            elif close_price > sma20.iloc[-1] > sma50.iloc[-1]:
                out["trend"] = "صاعد"
            elif close_price < sma20.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1]:
                out["trend"] = "هابط قوي"
            elif close_price < sma20.iloc[-1] < sma50.iloc[-1]:
                out["trend"] = "هابط"
            else:
                out["trend"] = "متذبذب"
        rsi = ind.get("rsi", pd.Series(dtype=float))
        if not rsi.empty:
            rv = float(rsi.iloc[-1])
            if rv < 30:
                out["rsi_signal"] = "شراء قوي"
            elif rv < 40:
                out["rsi_signal"] = "شراء"
            elif rv > 70:
                out["rsi_signal"] = "بيع قوي"
            elif rv > 60:
                out["rsi_signal"] = "بيع"
            else:
                out["rsi_signal"] = "محايد"
        macd = ind.get("macd", pd.Series(dtype=float))
        macd_sig = ind.get("macd_signal", pd.Series(dtype=float))
        if not macd.empty and not macd_sig.empty and len(macd) > 2:
            if macd.iloc[-1] > macd_sig.iloc[-1] and macd.iloc[-2] <= macd_sig.iloc[-2]:
                out["macd_signal"] = "شراء"
            elif macd.iloc[-1] < macd_sig.iloc[-1] and macd.iloc[-2] >= macd_sig.iloc[-2]:
                out["macd_signal"] = "بيع"
            else:
                out["macd_signal"] = "محايد"
        bbu = ind.get("bb_upper", pd.Series(dtype=float))
        bbl = ind.get("bb_lower", pd.Series(dtype=float))
        if not bbu.empty and not bbl.empty:
            out["bb_signal"] = "شراء" if close_price < bbl.iloc[-1] else ("بيع" if close_price > bbu.iloc[-1] else "محايد")
        pt_signals = []
        for k, v in patterns.items():
            if isinstance(v, (pd.Series, np.ndarray)):
                try:
                    if len(v) > 0 and v[-1] != 0:
                        pt_signals.append(k)
                except Exception:
                    pass
            elif isinstance(v, list) and v:
                if v[-1] == len(df) - 1:
                    pt_signals.append(k)
        out["patterns"] = pt_signals
        buy = 0.0
        sell = 0.0
        if out.get("rsi_signal") in ("شراء قوي", "شراء"): buy += 1
        if out.get("rsi_signal") in ("بيع قوي", "بيع"): sell += 1
        if out.get("macd_signal") == "شراء": buy += 1
        if out.get("macd_signal") == "بيع": sell += 1
        if out.get("bb_signal") == "شراء": buy += 1
        if out.get("bb_signal") == "بيع": sell += 1
        if out.get("trend") in ("صاعد", "صاعد قوي"): buy += 0.5
        if out.get("trend") in ("هابط", "هابط قوي"): sell += 0.5
        if buy > sell + 1:
            out["recommendation"], out["confidence"] = "شراء قوي", "عالية"
        elif buy > sell:
            out["recommendation"], out["confidence"] = "شراء", "متوسطة"
        elif sell > buy + 1:
            out["recommendation"], out["confidence"] = "بيع قوي", "عالية"
        elif sell > buy:
            out["recommendation"], out["confidence"] = "بيع", "متوسطة"
        else:
            out["recommendation"], out["confidence"] = "انتظار", "منخفضة"
        atr = ind.get("atr", pd.Series(dtype=float))
        try:
            if not atr.empty and not math.isnan(float(atr.iloc[-1])):
                out["stop_loss"] = float(close_price - 2 * float(atr.iloc[-1]))
                out["take_profit"] = float(close_price + 3 * float(atr.iloc[-1]))
                out["risk_level"] = "عالية" if (atr.iloc[-1] / close_price) > 0.03 else ("متوسطة" if (atr.iloc[-1] / close_price) > 0.015 else "منخفضة")
            else:
                out["stop_loss"] = float(close_price * 0.95)
                out["take_profit"] = float(close_price * 1.08)
                out["risk_level"] = "متوسطة"
        except Exception:
            out["stop_loss"], out["take_profit"], out["risk_level"] = None, None, "متوسطة"
        return out
    except Exception as e:
        log.warning(f"generate_signals error: {e}")
        return {"error": str(e)}

# ------------- News & Fundamentals (improved filter) -------------
def simple_sentiment_text(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    txt = text.lower()
    pos = ["rise", "rally", "surge", "gain", "bull", "safe haven", "cut", "dovish", "strong", "support"]
    neg = ["fall", "drop", "decline", "slump", "selloff", "hawkish", "raise", "weak", "pressure"]
    score = 0
    for w in pos:
        if w in txt:
            score += 1
    for w in neg:
        if w in txt:
            score -= 1
    return max(-1.0, min(1.0, score / 3.0))

def fetch_news_articles(api_key: str, q: str = None, page_size: int = 25) -> Dict[str, Any]:
    if not api_key:
        return {"error": "NEWS_API_KEY missing"}
    # improved query and preferred domains
    if q is None:
        q = '(gold OR "gold price" OR XAU OR bullion OR "precious metals" OR COMEX) AND (goldman OR bloomberg OR reuters OR fed OR inflation OR cpi OR dollar OR usd OR fed OR goldman OR etf OR mining OR "safe haven")'
    domains = "reuters.com,bloomberg.com,ft.com,wsj.com,marketwatch.com,barrons.com,investing.com,kitco.com,kitco,zerohedge.com,financialpost.com"
    url = "https://newsapi.org/v2/everything"
    params = {"q": q, "language": "en", "sortBy": "publishedAt", "pageSize": page_size, "apiKey": api_key, "domains": domains}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])[:page_size]
        parsed = []
        total = 0.0
        for a in arts:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            text = f"{title}. {desc}"
            s = simple_sentiment_text(text)
            total += s
            parsed.append({"title": title, "source": (a.get("source") or {}).get("name"), "publishedAt": a.get("publishedAt"), "url": a.get("url"), "sentiment": s})
        avg = total / max(1, len(parsed))
        impact = "dovish/bullish" if avg > 0.2 else ("hawkish/bearish" if avg < -0.2 else "neutral")
        return {"count": len(parsed), "average_sentiment": float(avg), "impact": impact, "articles": parsed}
    except Exception as e:
        log.warning(f"fetch_news_articles error: {e}")
        return {"error": str(e)}

FRED_SERIES = {
    "FEDFUNDS": "Effective Fed Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index: Broad, Goods",
    "DGS10": "10-Year Treasury Rate",
    "UNRATE": "Unemployment Rate"
}

def fetch_fred_series(series_id: str, api_key: str, obs: int = 24) -> Optional[pd.Series]:
    if not api_key:
        return None
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "limit": obs}
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        obs_list = js.get("observations", [])
        df = pd.DataFrame(obs_list)
        if df.empty or "value" not in df.columns:
            return None
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["value"].dropna().tail(obs)
        return s
    except Exception as e:
        log.warning(f"fetch_fred_series error: {e}")
        return None

def fundamental_block(api_key: str) -> Dict[str, Any]:
    if not api_key:
        return {"error": "FRED_API_KEY missing"}
    out = {}
    for sid, name in FRED_SERIES.items():
        s = fetch_fred_series(sid, api_key, obs=40)
        if s is None or s.empty:
            out[sid] = {"series": sid, "name": name, "error": "no data"}
            continue
        latest = clean_scalar(s.iloc[-1])
        prev_12 = clean_scalar(s.shift(12).iloc[-1]) if len(s) > 12 else None
        yoy = None
        if latest is not None and prev_12 not in (None, 0):
            try:
                yoy = float((latest - prev_12) / prev_12)
            except Exception:
                yoy = None
        out[sid] = {"series": sid, "name": name, "latest": latest, "yoy_change": yoy}
    bias = 0
    fed = out.get("FEDFUNDS", {})
    dxy = out.get("DTWEXBGS", {})
    cpi = out.get("CPIAUCSL", {})
    try:
        if fed.get("latest") is not None and fed["latest"] > 3: bias -= 1
        if dxy.get("yoy_change") is not None and dxy["yoy_change"] > 0.05: bias -= 1
        if cpi.get("yoy_change") is not None and cpi["yoy_change"] > 0.03: bias += 1
    except Exception:
        pass
    out["fundamental_bias"] = "dovish/bullish" if bias > 0 else ("hawkish/bearish" if bias < 0 else "neutral")
    return out

# ------------- Build report -------------
def build_unified_report(data: Dict[str, Any]) -> Dict[str, Any]:
    gold = data.get("GC=F", pd.DataFrame())
    if gold is None or gold.empty:
        return {"error": "GC=F data unavailable"}
    ind = technical_indicators(gold)
    patterns_candles = detect_candlestick_patterns(gold)
    patterns_price = detect_price_patterns(gold)
    patterns_all = {**patterns_candles, **patterns_price} if isinstance(patterns_candles, dict) else patterns_price
    low, high = float(gold["Low"].min()), float(gold["High"].max())
    fibo = fibonacci_levels(low, high)
    sr = support_resistance(gold)
    vp = volume_profile(gold)
    ms = market_structure(gold, ind)
    divs = detect_divergences(gold, ind)
    corr = correlation_block(data)
    adv = advanced_metrics(gold)
    senti = technical_sentiment(gold, ind)
    sig = generate_signals(gold, ind, patterns_all)
    fred = fundamental_block(FRED_API_KEY)
    news = fetch_news_articles(NEWS_API_KEY)
    report = {
        "metadata": {"version": "unified-full-v1", "symbol": "GC=F", "period": "1y", "analysis_date": dt.datetime.utcnow().isoformat(), "data_points": int(gold.shape[0]), "talib": TALIB_AVAILABLE},
        "used_symbols": data.get("used_symbols", {}),
        "current_market_data": {"current_price": clean_scalar(gold["Close"].iloc[-1]), "daily_change": clean_scalar(gold["Close"].iloc[-1] - gold["Close"].iloc[-2]) if gold.shape[0] > 1 else None, "daily_change_percent": clean_scalar(((gold["Close"].iloc[-1] / gold["Close"].iloc[-2]) - 1) * 100) if gold.shape[0] > 1 else None, "volume": clean_scalar(int(gold["Volume"].iloc[-1])), "high": clean_scalar(gold["High"].iloc[-1]), "low": clean_scalar(gold["Low"].iloc[-1])},
        "signals": convert_numpy_types(sig),
        "technical_indicators": convert_numpy_types({k: (v.iloc[-1] if (isinstance(v, pd.Series) and not v.empty) else None) for k, v in ind.items()}),
        "patterns": {k: (v.tolist() if isinstance(v, (pd.Series, np.ndarray)) else v) for k, v in patterns_all.items()},
        "enhancements": {"fibonacci": fibo, "support_resistance": sr, "volume_profile": vp, "market_structure": ms, "divergences": divs, "correlations": corr, "advanced_metrics": adv},
        "sentiment": convert_numpy_types(senti),
        "fundamentals": fred,
        "news": news,
        "summary": {"overall_recommendation": sig.get("recommendation"), "confidence": sig.get("confidence"), "risk_level": sig.get("risk_level"), "trend": sig.get("trend"), "key_support": sig.get("stop_loss"), "key_resistance": sig.get("take_profit"), "fundamental_bias": fred.get("fundamental_bias") if isinstance(fred, dict) else None, "news_impact": news.get("impact") if isinstance(news, dict) else None}
    }
    return convert_numpy_types(report)

def save_report(report: Dict[str, Any], filename: str = "gold_analysis_unified_full.json") -> bool:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        log.info(f"Saved {filename}")
        return True
    except Exception as e:
        log.error(f"save_report error: {e}")
        return False

def main():
    log.info("Starting unified full gold analysis...")
    data = fetch_market_universe(period="1y")
    report = build_unified_report(data)
    ok = save_report(report)
    if ok:
        log.info("Analysis complete. Output: gold_analysis_unified_full.json")
        print("✅ Done. File: gold_analysis_unified_full.json")
    else:
        log.error("Failed to save report.")
        print("❌ Failed to save report.")

if __name__ == "__main__":
    main()
