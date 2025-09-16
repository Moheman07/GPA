#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Analyzer — resilient fetch with Yahoo + Stooq fallbacks, caching, backoff, API/CLI/backtest
"""
import os
import math
import time
import json
import argparse
import logging
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import requests

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

CONFIG = {
    "symbols": {
        "gold_alts": ["GC=F", "XAUUSD=X", "GLD"],
        "usd_alts": ["^DXY", "DX-Y.NYB", "DX=F", "UUP"],
        "spy_alts": ["SPY", "^GSPC", "ES=F"]
    },
    "api_keys": {
        "fred": os.getenv("FRED_API_KEY", ""),
        "news": os.getenv("NEWS_API_KEY", "")
    },
    "fred_series": {
        "FEDFUNDS": "Effective Fed Funds Rate",
        "CPIAUCSL": "CPI (All Urban Consumers)",
        "DTWEXBGS": "Trade Weighted U.S. Dollar Index: Broad, Goods",
        "DGS10": "10-Year Treasury Rate",
        "UNRATE": "Unemployment Rate"
    },
    "defaults": {
        "period": "1y",
        "analysis_file": "gold_analysis_v7.json",
        "backtest_file": "gold_backtest_v7.json"
    },
    "backtest_params": {
        "adx_min": 20.0,
        "rsi_buy": 35.0,
        "rsi_sell": 65.0,
        "atr_mult_sl": 2.0,
        "atr_mult_tp": 3.0,
        "atr_trail_mult": 1.5,
        "commission_perc": 0.0005,
        "slippage_perc": 0.0005,
        "risk_per_trade": 0.01
    }
}

log = logging.getLogger("gold_analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def load_best_params():
    best_params_file = "best_params.json"
    try:
        if os.path.exists(best_params_file):
            with open(best_params_file, "r", encoding="utf-8") as f:
                best_params = json.load(f)
                CONFIG["backtest_params"].update(best_params)
                log.info(f"*** Successfully loaded and applied optimized parameters from {best_params_file} ***")
    except Exception as e:
        log.error(f"Could not load or apply {best_params_file}: {e}")

load_best_params()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import pandas as _pd
            if _pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)

def clean_scalar(x: Any) -> Optional[Any]:
    if x is None:
        return None
    if isinstance(x, (np.generic,)):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x

def safe_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan)

def _retry_sleep(base: float, attempt: int) -> float:
    import random
    return base * (2 ** attempt) + random.uniform(0, base)

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(symbol: str, period: str, src: str) -> str:
    safe = symbol.replace("=", "_").replace("^", "")
    return os.path.join(CACHE_DIR, f"{safe}_{period}_{src}.csv")

def _read_cache(path: str, max_age_seconds: int) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path) < max_age_seconds):
            df = pd.read_csv(path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            return df
    except Exception as e:
        log.warning(f"Cache read failed: {e}")
    return None

def _write_cache(path: str, df: pd.DataFrame):
    try:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out.to_csv(path, index_label="Date")
    except Exception as e:
        log.warning(f"Cache write failed: {e}")

def fetch_yahoo(symbol: str, period: str, tries: int = 3, pause: float = 1.0) -> pd.DataFrame:
    import yfinance as yf
    import requests as rq
    last_err = None
    sess = rq.Session()
    sess.headers.update({"User-Agent":"Mozilla/5.0"})
    for i in range(tries):
        try:
            log.info(f"Fetching {symbol} for period {period} via Yahoo (Attempt {i+1}/{tries})")
            df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, session=sess)
            if df is None or df.empty:
                tk = yf.Ticker(symbol, session=sess)
                df = tk.history(period=period, auto_adjust=False)
            if df is None or df.empty:
                raise ValueError("Empty dataframe returned from yfinance")
            for col in ["Open","High","Low","Close","Volume"]:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            df.dropna(how="all", inplace=True)
            return df
        except Exception as e:
            last_err = e
            log.warning(f"Yahoo failed {symbol} on attempt {i+1}: {e}")
            time.sleep(_retry_sleep(pause, i))
    log.error(f"Yahoo: all attempts failed for {symbol}. Last error: {last_err}")
    return pd.DataFrame()

def _stooq_url(stq_symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={stq_symbol}&i=d"

def fetch_stooq_symbol(stq_symbol: str) -> pd.DataFrame:
    url = _stooq_url(stq_symbol)
    try:
        log.info(f"Fetching {stq_symbol} via Stooq CSV")
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty or "Date" not in df.columns:
            return pd.DataFrame()
        # توحيد الأعمدة وتوليد Volume عند غيابه
        df.columns = [c.capitalize() for c in df.columns]
        for col in ["Open","High","Low","Close"]:
            if col not in df.columns:
                return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        out = df[["Open","High","Low","Close","Volume"]].copy()
        out = out.dropna(subset=["Open","High","Low","Close"])
        return out
    except Exception as e:
        log.warning(f"Stooq failed {stq_symbol}: {e}")
        return pd.DataFrame()

def stooq_candidates_for(y_symbol: str) -> List[str]:
    mapping = {
        "GC=F": ["xauusd"],
        "XAUUSD=X": ["xauusd"],
        "GLD": ["gld.us"],
        "^DXY": ["dxy", "uup.us"],
        "DX-Y.NYB": ["dxy", "uup.us"],
        "DX=F": ["dxy", "uup.us"],
        "UUP": ["uup.us"],
        "SPY": ["spy.us", "^spx"],
        "^GSPC": ["^spx"],
        "ES=F": ["es.f", "^spx"]
    }
    return mapping.get(y_symbol, [])

def fetch_any_with_fallback(symbols: List[str], period: str, ttl_seconds: int = 10800) -> Tuple[pd.DataFrame, Optional[str], str]:
    for s in symbols:
        cache_path = _cache_path(s, period, "yahoo")
        cached = _read_cache(cache_path, ttl_seconds)
        if cached is not None and not cached.empty:
            log.info(f"Using Yahoo cache for {s}")
            return cached, s, "yahoo"
        df = fetch_yahoo(s, period)
        if not df.empty:
            _write_cache(cache_path, df)
            return df, s, "yahoo"
        for stq in stooq_candidates_for(s):
            cache_path_s = _cache_path(stq, period, "stooq")
            cached_s = _read_cache(cache_path_s, ttl_seconds)
            if cached_s is not None and not cached_s.empty:
                log.info(f"Using Stooq cache for {stq} (mapped from {s})")
                return cached_s, s, "stooq"
            df_s = fetch_stooq_symbol(stq)
            if not df_s.empty:
                _write_cache(cache_path_s, df_s)
                return df_s, s, "stooq"
    generic = ["xauusd","gld.us","dxy","uup.us","spy.us","^spx","es.f"]
    for stq in generic:
        cache_path_s = _cache_path(stq, period, "stooq")
        cached_s = _read_cache(cache_path_s, ttl_seconds)
        if cached_s is not None and not cached_s.empty:
            return cached_s, stq, "stooq"
        df_s = fetch_stooq_symbol(stq)
        if not df_s.empty:
            _write_cache(cache_path_s, df_s)
            return df_s, stq, "stooq"
    return pd.DataFrame(), None, "none"

def fetch_market_data(period: str) -> Dict[str, Any]:
    log.info("--- Starting Market Data Fetch ---")
    out: Dict[str, Any] = {"used_symbols": {}, "used_sources": {}}
    g_df, g_used, g_src = fetch_any_with_fallback(CONFIG["symbols"]["gold_alts"], period)
    out["GC=F"] = g_df; out["used_symbols"]["GC=F"] = g_used; out["used_sources"]["GC=F"] = g_src
    d_df, d_used, d_src = fetch_any_with_fallback(CONFIG["symbols"]["usd_alts"], period)
    out["^DXY"] = d_df; out["used_symbols"]["^DXY"] = d_used; out["used_sources"]["^DXY"] = d_src
    s_df, s_used, s_src = fetch_any_with_fallback(CONFIG["symbols"]["spy_alts"], period)
    out["SPY"] = s_df; out["used_symbols"]["SPY"] = s_used; out["used_sources"]["SPY"] = s_src
    log.info(f"--- Market Data Fetch Complete (sources: gold={g_src}, usd={d_src}, spy={s_src}) ---")
    return out

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    log.info("Calculating technical indicators...")
    o,h,l,c,v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    ind: Dict[str, pd.Series] = {}
    if TALIB_AVAILABLE:
        ind["sma_20"] = talib.SMA(c, timeperiod=20)
        ind["sma_50"] = talib.SMA(c, timeperiod=50)
        ind["sma_200"] = talib.SMA(c, timeperiod=200)
        ind["ema_12"] = talib.EMA(c, timeperiod=12)
        ind["ema_26"] = talib.EMA(c, timeperiod=26)
        ind["rsi"] = talib.RSI(c, timeperiod=14)
        macd, macd_sig, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd, macd_sig, macd_hist
        upper, mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ind["bb_upper"], ind["bb_middle"], ind["bb_lower"] = upper, mid, lower
        tr = talib.TRANGE(h,l,c)
        ind["atr"] = talib.SMA(tr, timeperiod=14)
        ind["adx"] = talib.ADX(h, l, c, timeperiod=14)
    else:
        ind["sma_20"] = c.rolling(20).mean()
        ind["sma_50"] = c.rolling(50).mean()
        ind["sma_200"] = c.rolling(200).mean()
        ind["ema_12"] = ema(c, 12)
        ind["ema_26"] = ema(c, 26)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        ind["rsi"] = 100 - (100 / (1 + rs))
        macd = ind["ema_12"] - ind["ema_26"]
        ind["macd"] = macd
        ind["macd_signal"] = ema(macd, 9)
        ind["macd_hist"] = macd - ind["macd_signal"]
        m = c.rolling(20).mean(); s = c.rolling(20).std()
        ind["bb_upper"], ind["bb_middle"], ind["bb_lower"] = m+2*s, m, m-2*s
        tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(14).mean()
        ind["adx"] = pd.Series(index=df.index, dtype=float)
    bb_u, bb_m, bb_l = ind.get("bb_upper"), ind.get("bb_middle"), ind.get("bb_lower")
    if all(s is not None for s in [bb_u, bb_m, bb_l]):
        width = (bb_u - bb_l) / bb_m.replace(0,np.nan)
        ind["bb_width"] = width
        ind["bb_percent_b"] = (c - bb_l) / (bb_u - bb_l).replace(0,np.nan)
    return ind

def compute_regime(ind: Dict[str, pd.Series]) -> pd.Series:
    adx = ind.get("adx", pd.Series(dtype=float)).fillna(0)
    width = ind.get("bb_width", pd.Series(dtype=float)).ffill()
    med = width.rolling(60, min_periods=10).median()
    regime = pd.Series(index=adx.index, dtype=object)
    regime[(adx >= 20) & (width >= med.fillna(width.median()))] = "trend"
    regime[(adx < 20) | (width < med.fillna(width.median()))] = "range"
    return regime.fillna("range")

def volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
    try:
        c = df["Close"]; v = df["Volume"].astype(float)
        q1,q3 = v.quantile(0.25), v.quantile(0.75); iqr = q3-q1
        v_cap = v.clip(lower=max(0.0, q1-3*iqr), upper=q3+3*iqr)
        cats = pd.cut(c, bins=bins)
        vp = v_cap.groupby(cats, observed=False).sum().sort_values(ascending=False).head(6)
        prof = {str(k): int(val) for k,val in vp.items()}
        ret = safe_pct_change(c); val_v = v_cap.reindex(ret.index)
        corr = float(pd.Series(ret.values).corr(pd.Series(val_v.values))) if len(ret)>5 else None
        ma20 = v_cap.rolling(20).mean().iloc[-1] if len(v_cap)>=20 else np.nan
        trend = "مرتفع" if (not math.isnan(ma20) and v_cap.iloc[-1]>1.5*ma20) else ("منخفض" if (not math.isnan(ma20) and v_cap.iloc[-1]<0.5*ma20) else "عادي")
        return {"current_volume": int(v.iloc[-1]), "average_volume": float(v_cap.mean()), "volume_ratio": float(v.iloc[-1]/(ma20 or (v_cap.mean() or 1))), "volume_profile": prof, "price_volume_correlation": None if corr is None or math.isnan(corr) else corr, "volume_trend": trend}
    except Exception as e:
        log.warning(f"volume_profile error: {e}"); return {"error": str(e)}

def correlation_block(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    try:
        gold = data.get("GC=F", pd.DataFrame()); dxy = data.get("^DXY", pd.DataFrame()); spy = data.get("SPY", pd.DataFrame())
        out: Dict[str,float] = {}
        if gold.empty: return {"asset_correlations": {}}
        gr = safe_pct_change(gold["Close"])
        if not dxy.empty:
            xr = safe_pct_change(dxy["Close"]); a,b = gr.align(xr, join="inner"); val = a.corr(b) if len(a)>5 else None
            if pd.notna(val): out["USD"] = float(val)
        if not spy.empty:
            sr = safe_pct_change(spy["Close"]); a,b = gr.align(sr, join="inner"); val = a.corr(b) if len(a)>5 else None
            if pd.notna(val): out["SPY"] = float(val)
        return {"asset_correlations": out} if out else {"asset_correlations": {}}
    except Exception as e:
        log.warning(f"correlation_block error: {e}"); return {"asset_correlations": {}}

def detect_divergences(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict[str, Any]:
    try:
        close = df["Close"].astype(float); dates = df.index
        rsi = ind.get("rsi", pd.Series(dtype=float)).reindex(df.index).astype(float)
        macd = ind.get("macd", pd.Series(dtype=float)).reindex(df.index).astype(float)
        price = close.values
        pr_rng = np.nanmax(price) - np.nanmin(price)
        min_prom = max(0.002*pr_rng, 0.5)
        if SCIPY_AVAILABLE:
            peaks,_ = find_peaks(price, prominence=min_prom); troughs,_ = find_peaks(-price, prominence=min_prom)
        else:
            peaks = [i for i in range(1,len(price)-1) if price[i]>price[i-1] and price[i]>price[i+1]]
            troughs = [i for i in range(1,len(price)-1) if price[i]<price[i-1] and price[i]<price[i+1]]
        divs = []
        def add(tp, name, i1, prev, now):
            divs.append({"type":tp,"indicator":name,"price_idx":int(i1),"price_date":str(dates[i1]),"price":float(price[i1]),"indicator_prev":clean_scalar(prev),"indicator_now":clean_scalar(now)})
        for j in range(1,len(peaks)):
            i0,i1 = peaks[j-1], peaks[j]
            if price[i1] > price[i0]:
                r0,r1 = rsi.iat[i0], rsi.iat[i1]
                if pd.notna(r0) and pd.notna(r1) and r1 < r0: add("bearish","RSI",i1,r0,r1)
                m0,m1 = macd.iat[i0], macd.iat[i1]
                if pd.notna(m0) and pd.notna(m1) and m1 < m0: add("bearish","MACD",i1,m0,m1)
        for j in range(1,len(troughs)):
            i0,i1 = troughs[j-1], troughs[j]
            if price[i1] < price[i0]:
                r0,r1 = rsi.iat[i0], rsi.iat[i1]
                if pd.notna(r0) and pd.notna(r1) and r1 > r0: add("bullish","RSI",i1,r0,r1)
                m0,m1 = macd.iat[i0], macd.iat[i1]
                if pd.notna(m0) and pd.notna(m1) and m1 > m0: add("bullish","MACD",i1,m0,m1)
        recent = sorted(divs, key=lambda x: x["price_idx"])[-10:]
        return {"total_divergences": len(divs), "recent_divergences": len(recent), "positive_divergences": len([d for d in recent if d["type"]=="bullish"]), "negative_divergences": len([d for d in recent if d["type"]=="bearish"]), "latest_divergences": recent}
    except Exception as e:
        log.warning(f"divergence error: {e}"); return {"error": str(e)}

def technical_sentiment(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict[str, Any]:
    out: Dict[str,Any] = {}
    c = float(df["Close"].iloc[-1])
    rsi = ind.get("rsi", pd.Series(dtype=float)); r = float(rsi.iloc[-1]) if (not rsi.empty and pd.notna(rsi.iloc[-1])) else 50.0
    out["rsi_sentiment"] = "مفرط في الشراء" if r>70 else ("مفرط في البيع" if r<30 else "محايد")
    bb_u = ind.get("bb_upper", pd.Series(dtype=float)); bb_l = ind.get("bb_lower", pd.Series(dtype=float))
    if not bb_u.empty and not bb_l.empty:
        out["bb_sentiment"] = "مفرط في الشراء" if c>bb_u.iloc[-1] else ("مفرط في البيع" if c<bb_l.iloc[-1] else "عادي")
    adx = ind.get("adx", pd.Series(dtype=float))
    if not adx.empty and pd.notna(adx.iloc[-1]):
        out["trend_strength"] = "قوي" if adx.iloc[-1] > 25 else "ضعيف"
    return out

def generate_signals(df: pd.DataFrame, ind: Dict[str, pd.Series], regime: pd.Series) -> Dict[str, Any]:
    c = float(df["Close"].iloc[-1])
    out: Dict[str,Any] = {"current_price": c, "timestamp": dt.datetime.utcnow().replace(tzinfo=None).isoformat()}
    s20,s50,s200 = ind.get("sma_20",pd.Series(dtype=float)), ind.get("sma_50",pd.Series(dtype=float)), ind.get("sma_200",pd.Series(dtype=float))
    if not s20.empty and not s50.empty and not s200.empty:
        if c>s20.iloc[-1]>s50.iloc[-1]>s200.iloc[-1]: out["trend"]="صاعد قوي"
        elif c>s20.iloc[-1]>s50.iloc[-1]: out["trend"]="صاعد"
        elif c<s20.iloc[-1]<s50.iloc[-1]<s200.iloc[-1]: out["trend"]="هابط قوي"
        elif c<s20.iloc[-1]<s50.iloc[-1]: out["trend"]="هابط"
        else: out["trend"]="متذبذب"
    rsi = ind.get("rsi", pd.Series(dtype=float))
    if not rsi.empty and pd.notna(rsi.iloc[-1]):
        rv=float(rsi.iloc[-1])
        out["rsi_signal"] = "شراء قوي" if rv<30 else ("شراء" if rv<40 else ("بيع قوي" if rv>70 else ("بيع" if rv>60 else "محايد")))
    macd, macd_sig = ind.get("macd",pd.Series(dtype=float)), ind.get("macd_signal",pd.Series(dtype=float))
    if len(macd)>2 and len(macd_sig)>2:
        if macd.iloc[-1]>macd_sig.iloc[-1] and macd.iloc[-2]<=macd_sig.iloc[-2]: out["macd_signal"]="شراء"
        elif macd.iloc[-1]<macd_sig.iloc[-1] and macd.iloc[-2]>=macd_sig.iloc[-2]: out["macd_signal"]="بيع"
        else: out["macd_signal"]="محايد"
    bbu,bbl = ind.get("bb_upper",pd.Series(dtype=float)), ind.get("bb_lower",pd.Series(dtype=float))
    if not bbu.empty and not bbl.empty:
        out["bb_signal"] = "شراء" if c<bbl.iloc[-1] else ("بيع" if c>bbu.iloc[-1] else "محايد")
    adx = ind.get("adx", pd.Series(dtype=float))
    weak = (not adx.empty and pd.notna(adx.iloc[-1]) and adx.iloc[-1] < 20)
    regime_now = regime.iloc[-1] if not regime.empty else "range"
    buy=sell=0.0
    rsi_w = 0.6 if weak else 1.0
    macd_w = 1.0 if regime_now=="trend" else 0.7
    bb_w = 1.0 if regime_now=="range" else 0.6
    trend_w = 0.7 if regime_now=="trend" else 0.3
    if out.get("rsi_signal") in ("شراء قوي","شراء"): buy += rsi_w
    if out.get("rsi_signal") in ("بيع قوي","بيع"): sell += rsi_w
    if out.get("macd_signal")=="شراء": buy += macd_w
    if out.get("macd_signal")=="بيع": sell += macd_w
    if out.get("bb_signal")=="شراء": buy += bb_w
    if out.get("bb_signal")=="بيع": sell += bb_w
    if out.get("trend") in ("صاعد","صاعد قوي"): buy += trend_w
    if out.get("trend") in ("هابط","هابط قوي"): sell += trend_w
    if buy > sell + 1: out["recommendation"],out["confidence"]="شراء قوي","عالية"
    elif buy > sell: out["recommendation"],out["confidence"]="شراء","متوسطة"
    elif sell > buy + 1: out["recommendation"],out["confidence"]="بيع قوي","عالية"
    elif sell > buy: out["recommendation"],out["confidence"]="بيع","متوسطة"
    else: out["recommendation"],out["confidence"]="انتظار","منخفضة"
    atr = ind.get("atr", pd.Series(dtype=float))
    try:
        if not atr.empty and pd.notna(atr.iloc[-1]):
            out["stop_loss"]=float(c - 2*float(atr.iloc[-1]))
            out["take_profit"]=float(c + 3*float(atr.iloc[-1]))
            out["risk_level"]="عالية" if (atr.iloc[-1]/c) > 0.03 else ("متوسطة" if (atr.iloc[-1]/c)>0.015 else "منخفضة")
        else:
            out["stop_loss"]=float(c*0.95); out["take_profit"]=float(c*1.08); out["risk_level"]="متوسطة"
    except Exception:
        out["stop_loss"],out["take_profit"],out["risk_level"]=None,None,"متوسطة"
    out["regime"] = regime_now
    return out

def fetch_news(api_key: str, page_size: int = 20) -> Dict[str, Any]:
    log.info("Fetching news...")
    if not api_key:
        return {"error":"NEWS_API_KEY missing"}
    q = '(gold OR "gold price" OR XAU OR bullion) AND (fed OR inflation OR dollar OR usd OR etf OR mining OR "safe haven")'
    domains = "reuters.com,bloomberg.com,ft.com,wsj.com,marketwatch.com,investing.com,kitco.com,financialpost.com"
    url = "https://newsapi.org/v2/everything"
    params = {"q":q,"language":"en","sortBy":"publishedAt","pageSize":page_size,"apiKey":api_key,"domains":domains}
    def simple_sentiment(txt: str)->float:
        if not txt: return 0.0
        t = txt.lower()
        pos = ["rise","rally","surge","gain","bull","safe haven","cut","dovish","strong","support"]
        neg = ["fall","drop","decline","slump","selloff","hawkish","raise","weak","pressure"]
        s=0
        for w in pos:
            if w in t: s+=1
        for w in neg:
            if w in t: s-=1
        return max(-1.0,min(1.0,s/3.0))
    try:
        for i in range(3):
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                break
            except Exception:
                if i==2: raise
                time.sleep(_retry_sleep(1.0, i))
        arts = (r.json().get("articles", []) or [])[:page_size]
        parsed=[]; total=0.0
        for a in arts:
            title=a.get("title") or ""; desc=a.get("description") or ""
            s = simple_sentiment(f"{title}. {desc}"); total += s
            parsed.append({"title":title,"source":(a.get("source") or {}).get("name"),"publishedAt":a.get("publishedAt"),"url":a.get("url"),"sentiment":s})
        avg = total/max(1,len(parsed))
        impact = "dovish/bullish" if avg>0.2 else ("hawkish/bearish" if avg<-0.2 else "neutral")
        return {"count": len(parsed), "average_sentiment": float(avg), "impact": impact, "articles": parsed}
    except Exception as e:
        log.error(f"News fetch failed: {e}")
        return {"error": f"news_fetch_failed: {e}"}

def fetch_fred_series(series_id: str, api_key: str, obs: int = 24) -> Optional[pd.Series]:
    if not api_key:
        return None
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id":series_id,"api_key":api_key,"file_type":"json","limit":obs}
    try:
        for i in range(3):
            try:
                r = requests.get(base, params=params, timeout=15); r.raise_for_status()
                break
            except Exception:
                if i==2: raise
                time.sleep(_retry_sleep(1.0, i))
        df = pd.DataFrame((r.json() or {}).get("observations", []))
        if df.empty or "value" not in df.columns: return None
        df["value"]=pd.to_numeric(df["value"], errors="coerce"); df["date"]=pd.to_datetime(df["date"])
        return df.set_index("date")["value"].dropna().tail(obs)
    except Exception as e:
        log.error(f"Failed to fetch FRED series {series_id}: {e}")
        return None

def get_fundamentals() -> Dict[str, Any]:
    out: Dict[str,Any] = {}
    for sid,name in CONFIG["fred_series"].items():
        s = fetch_fred_series(sid, CONFIG["api_keys"]["fred"], obs=40)
        if s is None or s.empty:
            out[sid] = {"series":sid,"name":name,"error":"no data"}; continue
        latest = clean_scalar(s.iloc[-1]); prev_12 = clean_scalar(s.shift(12).iloc[-1]) if len(s)>12 else None
        yoy=None
        if latest is not None and prev_12 not in (None,0):
            try: yoy = float((latest - prev_12)/abs(prev_12))
            except Exception: yoy=None
        out[sid] = {"series":sid,"name":name,"latest":latest,"yoy_change":yoy}
    bias = 0
    try:
        fed = out.get("FEDFUNDS",{}); dxy = out.get("DTWEXBGS",{}); cpi = out.get("CPIAUCSL",{})
        if fed.get("latest") and fed["latest"]>3: bias -= 1
        if dxy.get("yoy_change") and dxy["yoy_change"]>0.05: bias -= 1
        if cpi.get("yoy_change") and cpi["yoy_change"]>0.03: bias += 1
    except Exception:
        pass
    out["fundamental_bias"] = "dovish/bullish" if bias>0 else ("hawkish/bearish" if bias<0 else "neutral")
    return out

def build_full_report(data: Dict[str, Any]) -> Dict[str, Any]:
    gold_df = data.get("GC=F", pd.DataFrame())
    if gold_df.empty:
        return {"error": "GC=F data unavailable"}
    gold_df = gold_df.copy()
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)
    indicators = calc_indicators(gold_df)
    regime = compute_regime(indicators)
    signals = generate_signals(gold_df, indicators, regime)
    volume = volume_profile(gold_df)
    divergences = detect_divergences(gold_df, indicators)
    correlations = correlation_block(data)
    tech_sentiment = technical_sentiment(gold_df, indicators)
    fundamentals = get_fundamentals()
    news = fetch_news(CONFIG["api_keys"]["news"])
    report = {
        "metadata": {
            "version": "v7-resilient",
            "symbol": data.get("used_symbols", {}).get("GC=F"),
            "source_gold": data.get("used_sources", {}).get("GC=F"),
            "analysis_date": dt.datetime.utcnow().isoformat(),
            "talib_enabled": TALIB_AVAILABLE,
            "scipy_enabled": SCIPY_AVAILABLE
        },
        "used_symbols": data.get("used_symbols", {}),
        "used_sources": data.get("used_sources", {}),
        "current_market_data": {
            "price": clean_scalar(gold_df["Close"].iloc[-1]),
            "chg": clean_scalar(gold_df["Close"].diff().iloc[-1]),
            "chg_pct": clean_scalar(gold_df["Close"].pct_change().iloc[-1] * 100),
            "high": clean_scalar(gold_df["High"].iloc[-1]),
            "low": clean_scalar(gold_df["Low"].iloc[-1]),
            "volume": clean_scalar(gold_df["Volume"].iloc[-1])
        },
        "signals": signals,
        "technical_snapshot": {k: clean_scalar(v.iloc[-1]) for k, v in indicators.items()},
        "enhancements": {"divergences": divergences, "correlations": correlations, "volume_profile": volume},
        "sentiment": tech_sentiment,
        "fundamentals": fundamentals,
        "news": news
    }
    report["ai_summary"] = {
        "nl_summary_ar": f"النظام: {signals.get('regime')}. الاتجاه: {signals.get('trend')}. التوصية: {signals.get('recommendation')} (ثقة {signals.get('confidence')}).",
        "actions": {
            "signal_type": signals.get("recommendation"),
            "size_hint": 0 if signals.get("recommendation") in ("انتظار", "محايد") else (1 if signals.get("confidence") == "منخفضة" else 2),
            "sl": signals.get("stop_loss"),
            "tp": signals.get("take_profit")
        }
    }
    return report

def to_compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    signals = report.get("signals", {})
    return {
        "symbol": report.get("metadata", {}).get("symbol"),
        "price": report.get("current_market_data", {}).get("price"),
        "trend": signals.get("trend"),
        "regime": signals.get("regime"),
        "rec": signals.get("recommendation"),
        "conf": signals.get("confidence"),
        "sl": signals.get("stop_loss"),
        "tp": signals.get("take_profit"),
        "ai_text_ar": report.get("ai_summary", {}).get("nl_summary_ar")
    }

def backtest_engine(df: pd.DataFrame, ind: Dict[str, pd.Series], regime: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
    close = df["Close"].astype(float)
    rsi = ind["rsi"]; macd, macd_sig = ind["macd"], ind["macd_signal"]
    bb_u, bb_l = ind["bb_upper"], ind["bb_lower"]
    adx = ind.get("adx", pd.Series(dtype=float))
    atr = ind["atr"].ffill()
    adx_min = params.get("adx_min", 20.0)
    rsi_buy = params.get("rsi_buy", 35.0)
    rsi_sell = params.get("rsi_sell", 65.0)
    atr_sl_mult = params.get("atr_mult_sl", 2.0)
    atr_tp_mult = params.get("atr_mult_tp", 3.0)
    trail_mult = params.get("atr_trail_mult", 1.5)
    commission = params.get("commission_perc", 0.0005)
    slippage = params.get("slippage_perc", 0.0005)
    risk_per_trade = params.get("risk_per_trade", 0.01)
    equity = 1.0; max_equity = equity
    in_pos = False; side=None; entry=None; size=0.0; sl=None; tp=None; trail=None
    trades=[]
    def apply_cost(price: float, is_buy: bool) -> float:
        return price * (1 + commission + slippage) if is_buy else price * (1 - commission - slippage)
    for i in range(2, len(close)):
        price = close.iat[i]
        reg_now = regime.iat[i] if i < len(regime) and pd.notna(regime.iat[i]) else "range"
        if in_pos:
            if side=="long":
                trail = max(trail, price - trail_mult*atr.iat[i]) if trail is not None else price - trail_mult*atr.iat[i]
                hit_trail = price <= trail; hit_sl = price <= sl; hit_tp = price >= tp
                exit_price = apply_cost(price, is_buy=False) if (hit_trail or hit_sl or hit_tp) else None
            else:
                trail = min(trail, price + trail_mult*atr.iat[i]) if trail is not None else price + trail_mult*atr.iat[i]
                hit_trail = price >= trail; hit_sl = price >= sl; hit_tp = price <= tp
                exit_price = apply_cost(price, is_buy=True) if (hit_trail or hit_sl or hit_tp) else None
            if exit_price is not None:
                pnl = (exit_price - entry) / entry if side=="long" else (entry - exit_price) / entry
                equity *= (1 + pnl * size)
                trades.append({"i": i, "side": side, "entry": float(entry), "exit": float(exit_price), "size": float(size), "pnl_pct": float(pnl)})
                in_pos=False; side=None; entry=None; size=0.0; sl=None; tp=None; trail=None
                max_equity = max(max_equity, equity)
        if in_pos:
            max_equity = max(max_equity, equity); continue
        if any(pd.isna(x) for x in [macd.iat[i], macd_sig.iat[i], atr.iat[i]]) or atr.iat[i] <= 0:
            max_equity = max(max_equity, equity); continue
        cross_up = macd.iat[i] > macd_sig.iat[i] and macd.iat[i-1] <= macd_sig.iat[i-1]
        cross_dn = macd.iat[i] < macd_sig.iat[i] and macd.iat[i-1] >= macd_sig.iat[i-1]
        strong = (pd.notna(adx.iat[i]) and adx.iat[i] >= adx_min) if not adx.empty else True
        avoid_long = (pd.notna(rsi.iat[i]) and rsi.iat[i] > rsi_sell) or (pd.notna(bb_u.iat[i]) and price > bb_u.iat[i])
        avoid_short = (pd.notna(rsi.iat[i]) and rsi.iat[i] < rsi_buy) or (pd.notna(bb_l.iat[i]) and price < bb_l.iat[i])
        enter_long = enter_short = False
        if reg_now == "trend":
            enter_long = cross_up and strong and not avoid_long
            enter_short = cross_dn and strong and not avoid_short
        else:
            enter_long = ((pd.notna(bb_l.iat[i]) and price <= bb_l.iat[i]) or (pd.notna(rsi.iat[i]) and rsi.iat[i] <= rsi_buy)) and not avoid_long
            enter_short = ((pd.notna(bb_u.iat[i]) and price >= bb_u.iat[i]) or (pd.notna(rsi.iat[i]) and rsi.iat[i] >= rsi_sell)) and not avoid_short
        if enter_long or enter_short:
            stop_dist = atr_sl_mult * atr.iat[i]
            if stop_dist <= 0:
                max_equity = max(max_equity, equity); continue
            size = min(1.0, max(0.0, risk_per_trade / (stop_dist / max(price, 1e-8))))
            if size <= 0:
                max_equity = max(max_equity, equity); continue
            if enter_long:
                entry = apply_cost(price, is_buy=True); side="long"
                sl = entry - atr_sl_mult*atr.iat[i]; tp = entry + atr_tp_mult*atr.iat[i]; trail = entry - trail_mult*atr.iat[i]
                in_pos=True
            elif enter_short:
                entry = apply_cost(price, is_buy=False); side="short"
                sl = entry + atr_sl_mult*atr.iat[i]; tp = entry - atr_tp_mult*atr.iat[i]; trail = entry + trail_mult*atr.iat[i]
                in_pos=True
        max_equity = max(max_equity, equity)
    rets = pd.Series([t["pnl_pct"]*t["size"] for t in trades], dtype=float)
    win = rets[rets>0]; loss = rets[rets<0]
    eq_path=[1.0]; eq=1.0
    for r in rets: eq *= (1+r); eq_path.append(eq)
    path = pd.Series(eq_path); run_max = path.cummax(); dd_series = (path-run_max)/run_max
    dd = float(dd_series.min()) if len(dd_series)>0 else 0.0
    sharpe = float((rets.mean()/rets.std())*np.sqrt(252)) if len(rets)>2 and rets.std()!=0 else None
    downside = rets[rets<0]
    sortino = float((rets.mean()/downside.std())*np.sqrt(252)) if len(downside)>1 and downside.std()!=0 else None
    pf = float(abs(win.sum()/loss.sum())) if len(win)>0 and len(loss)>0 and loss.sum()!=0 else None
    perf = {"trades": int(len(trades)), "win_rate": float(len(win)/len(rets)) if len(rets)>0 else None, "profit_factor": pf, "cagr": None, "max_drawdown": dd, "sharpe": sharpe, "sortino": sortino, "final_equity": float(eq_path[-1]) if eq_path else 1.0}
    return {"performance": perf, "last_trades_sample": trades[-10:]}

def run_analysis_for_api(period: Optional[str] = None) -> Dict[str, Any]:
    analysis_period = period if period else CONFIG["defaults"]["period"]
    market_data = fetch_market_data(analysis_period)
    report = build_full_report(market_data)
    return report

def run_cli_analyze(args):
    market_data = fetch_market_data(args.period)
    report = build_full_report(market_data)
    output_file = args.out or CONFIG["defaults"]["analysis_file"]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    if args.compact:
        compact_file = output_file.replace(".json", "_compact.json")
        with open(compact_file, "w", encoding="utf-8") as f:
            json.dump(to_compact_report(report), f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def run_cli_backtest(args):
    market_data = fetch_market_data(args.period)
    gold_df = market_data.get("GC=F", pd.DataFrame())
    if gold_df.empty:
        log.critical("Cannot run backtest without gold data.")
        return
    gold_df = gold_df.copy()
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)
    indicators = calc_indicators(gold_df)
    regime = compute_regime(indicators)
    params = CONFIG["backtest_params"]
    for key in params:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            params[key] = cli_value
    results = backtest_engine(gold_df, indicators, regime, params)
    output = {"metadata": {"version": "v7-resilient", "symbol": market_data.get("used_symbols", {}).get("GC=F"), "period": args.period, "strategy": "Regime_MACD_BB_RSI_ATR_Pro"}, "params": params, **results}
    output_file = args.out or CONFIG["defaults"]["backtest_file"]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def main():
    parser = argparse.ArgumentParser(description="Gold Analyzer (Yahoo + Stooq)")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Execution mode")
    p_an = subparsers.add_parser("analyze", help="Run a full analysis and save the report.")
    p_an.add_argument("--period", default=CONFIG["defaults"]["period"], help="Data period (e.g., 1y, 6mo)")
    p_an.add_argument("--out", default=CONFIG["defaults"]["analysis_file"], help="Output file for the report.")
    p_an.add_argument("--compact", action="store_true", help="Save a compact version of the report.")
    p_an.set_defaults(func=run_cli_analyze)
    p_bt = subparsers.add_parser("backtest", help="Run a backtest of the trading strategy.")
    p_bt.add_argument("--period", default="3y", help="Data period for backtesting (e.g., 5y, 10y)")
    p_bt.add_argument("--out", default=CONFIG["defaults"]["backtest_file"], help="Output file for backtest results.")
    for param, value in CONFIG["backtest_params"].items():
        p_bt.add_argument(f"--{param.replace('_', '-')}", type=type(value), default=None, help=f"Override {param} (default: {value})")
    p_bt.set_defaults(func=run_cli_backtest)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
