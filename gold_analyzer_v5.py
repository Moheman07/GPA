#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Analyzer V6 — Professional, n8n-ready
- Modes: analyze | backtest
- Features:
  - Regime filter (trend vs range) ويؤثر على الإشارات
  - وزن الإشارات حسب قوة الاتجاه (ADX) وتضييق/اتساع البولنجر
  - إدارة مخاطر احترافية: حجم صفقة كنسبة مخاطرة، عمولات، انزلاق، وقف متتالي ATR Trailing
  - باكتيست واقعي بمقاييس أداء مفصلة
  - أخبار وبيانات أساسية (FRED) وخلاصة AI
  - CLI مرن وJSON نظيف + نسخة compact
"""
import os
import math
import time
import json
import argparse
import logging
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Optional deps
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

log = logging.getLogger("gold-v6")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ----------------------- JSON Utils -----------------------
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
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
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

def safe_pct_change(series: pd.Series) -> pd.Series:
    try:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return pd.Series(dtype=float)

# ----------------------- Data -----------------------
def fetch_yf(symbol: str, period: str = "1y", tries: int = 3, pause: float = 1.0) -> pd.DataFrame:
    import yfinance as yf
    last_err = None
    for i in range(tries):
        try:
            log.info(f"Fetching {symbol} ({i+1}/{tries})")
            df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
            if df is None or df.empty:
                last_err = Exception("Empty dataframe")
                time.sleep(pause)
                continue
            for col in ["Open","High","Low","Close","Volume"]:
                if col not in df.columns:
                    raise ValueError(f"Missing {col}")
            df = df.dropna(how="all")
            return df
        except Exception as e:
            last_err = e
            time.sleep(pause)
    log.warning(f"Failed {symbol}: {last_err}")
    return pd.DataFrame()

def fetch_market(period: str, gold_sym: str, usd_syms: List[str], spy_sym: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"used_symbols": {}}
    g = fetch_yf(gold_sym, period)
    out["GC=F"] = g; out["used_symbols"]["GC=F"] = gold_sym if not g.empty else None
    used = None; usd_df = pd.DataFrame()
    for s in usd_syms:
        d = fetch_yf(s, period)
        if not d.empty:
            used, usd_df = s, d; break
    out["^DXY"] = usd_df; out["used_symbols"]["^DXY"] = used
    sp = fetch_yf(spy_sym, period)
    out["SPY"] = sp; out["used_symbols"]["SPY"] = spy_sym if not sp.empty else None
    return out

# ----------------------- Indicators -----------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
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
        ind["ema_12"] = ema(c, 12); ind["ema_26"] = ema(c, 26)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        ind["rsi"] = 100 - (100 / (1 + rs))
        macd = ind["ema_12"] - ind["ema_26"]; ind["macd"]=macd; ind["macd_signal"]=ema(macd,9); ind["macd_hist"]=macd-ind["macd_signal"]
        m=c.rolling(20).mean(); s=c.rolling(20).std()
        ind["bb_upper"], ind["bb_middle"], ind["bb_lower"] = m+2*s, m, m-2*s
        tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(14).mean()
        ind["adx"] = pd.Series(index=df.index, dtype=float)
    # Bollinger Bandwidth (اتساع الباند) + %B
    bb_u, bb_m, bb_l = ind.get("bb_upper"), ind.get("bb_middle"), ind.get("bb_lower")
    if bb_u is not None and bb_m is not None and bb_l is not None:
        width = (bb_u - bb_l) / bb_m.replace(0,np.nan)
        ind["bb_width"] = width
        ind["bb_percent_b"] = (c - bb_l) / (bb_u - bb_l).replace(0,np.nan)
    return ind

def compute_regime(ind: Dict[str, pd.Series]) -> pd.Series:
    """
    Regime filter:
    - trend if: ADX >= 20 and BB width >= median(width)
    - range otherwise
    Returns series: "trend" or "range"
    """
    adx = ind.get("adx", pd.Series(dtype=float)).fillna(0)
    width = ind.get("bb_width", pd.Series(dtype=float)).fillna(method="ffill")
    med = width.rolling(60, min_periods=10).median()
    regime = pd.Series(index=adx.index, dtype=object)
    regime[(adx >= 20) & (width >= med.fillna(width.median()))] = "trend"
    regime[(adx < 20) | (width < med.fillna(width.median()))] = "range"
    return regime.fillna("range")

# ----------------------- Volume/Correlation -----------------------
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

# ----------------------- Divergences -----------------------
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

# ----------------------- Sentiment & Signals -----------------------
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
    # وزن سياقي
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

# ----------------------- News & Fundamentals -----------------------
def fetch_news(api_key: str, page_size: int = 20) -> Dict[str, Any]:
    if not api_key: return {"error":"NEWS_API_KEY missing"}
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
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code==429: time.sleep(2*(attempt+1)); continue
            r.raise_for_status()
            arts = (r.json().get("articles", []) or [])[:page_size]
            parsed=[]; total=0.0
            for a in arts:
                title=a.get("title") or ""; desc=a.get("description") or ""
                s = simple_sentiment(f"{title}. {desc}"); total += s
                parsed.append({"title":title,"source":(a.get("source") or {}).get("name"),"publishedAt":a.get("publishedAt"),"url":a.get("url"),"sentiment":s})
            avg = total/max(1,len(parsed))
            impact = "dovish/bullish" if avg>0.2 else ("hawkish/bearish" if avg<-0.2 else "neutral")
            return {"count": len(parsed), "average_sentiment": float(avg), "impact": impact, "articles": parsed}
        except Exception:
            time.sleep(2*(attempt+1))
    return {"error":"news_fetch_failed"}

FRED_SERIES = {
    "FEDFUNDS": "Effective Fed Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index: Broad, Goods",
    "DGS10": "10-Year Treasury Rate",
    "UNRATE": "Unemployment Rate"
}

def fetch_fred_series(series_id: str, api_key: str, obs: int = 24) -> Optional[pd.Series]:
    if not api_key: return None
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id":series_id,"api_key":api_key,"file_type":"json","limit":obs}
    try:
        r = requests.get(base, params=params, timeout=15); r.raise_for_status()
        df = pd.DataFrame((r.json() or {}).get("observations", []))
        if df.empty or "value" not in df.columns: return None
        df["value"]=pd.to_numeric(df["value"], errors="coerce"); df["date"]=pd.to_datetime(df["date"])
        return df.set_index("date")["value"].dropna().tail(obs)
    except Exception:
        return None

def fundamentals_block() -> Dict[str, Any]:
    out: Dict[str,Any] = {}
    for sid,name in FRED_SERIES.items():
        s = fetch_fred_series(sid, FRED_API_KEY, obs=40)
        if s is None or s.empty:
            out[sid] = {"series":sid,"name":name,"error":"no data"}; continue
        latest = clean_scalar(s.iloc[-1]); prev_12 = clean_scalar(s.shift(12).iloc[-1]) if len(s)>12 else None
        yoy=None
        if latest is not None and prev_12 not in (None,0):
            try: yoy = float((latest - prev_12)/prev_12)
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

# ----------------------- Report -----------------------
def build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    gold = data.get("GC=F", pd.DataFrame())
    if gold.empty: return {"error":"GC=F data unavailable"}
    gold = gold.copy(); gold.index = pd.to_datetime(gold.index).tz_localize(None)
    ind = calc_indicators(gold)
    regime = compute_regime(ind)
    vp = volume_profile(gold)
    divs = detect_divergences(gold, ind)
    corr = correlation_block(data)
    senti = technical_sentiment(gold, ind)
    sig = generate_signals(gold, ind, regime)
    news = fetch_news(NEWS_API_KEY)
    fred = fundamentals_block()
    rep = {
        "metadata": {"version":"v6","symbol":"GC=F","period":"1y","analysis_date": dt.datetime.utcnow().replace(tzinfo=None).isoformat(), "talib": TALIB_AVAILABLE},
        "used_symbols": data.get("used_symbols", {}),
        "current_market_data": {
            "price": clean_scalar(gold["Close"].iloc[-1]),
            "chg": clean_scalar(gold["Close"].iloc[-1]-gold["Close"].iloc[-2]) if len(gold)>1 else None,
            "chg_pct": clean_scalar((gold["Close"].iloc[-1]/gold["Close"].iloc[-2]-1)*100) if len(gold)>1 else None,
            "high": clean_scalar(gold["High"].iloc[-1]),
            "low": clean_scalar(gold["Low"].iloc[-1]),
            "volume": clean_scalar(int(gold["Volume"].iloc[-1]))
        },
        "signals": sig,
        "technical": {k: (float(v.iloc[-1]) if isinstance(v,pd.Series) and not v.empty and pd.notna(v.iloc[-1]) else None) for k,v in ind.items()},
        "enhancements": {"divergences": divs, "correlations": corr, "volume_profile": vp},
        "sentiment": senti,
        "fundamentals": fred,
        "news": news
    }
    trend = sig.get("trend"); rec = sig.get("recommendation"); conf = sig.get("confidence"); regime_now=sig.get("regime")
    rep["ai"] = {
        "nl_summary_ar": f"النظام: {regime_now}. الاتجاه: {trend}. التوصية: {rec} (ثقة {conf}). راقب التشبّع واتساع البولنجر وقوة الاتجاه قبل الدخول.",
        "actions": {"signal_type": rec, "size_hint": 0 if rec in ("انتظار","محايد") else (1 if conf=="منخفضة" else 2), "sl": sig.get("stop_loss"), "tp": sig.get("take_profit")}
    }
    return rep

def to_compact(report: Dict[str,Any]) -> Dict[str,Any]:
    sig = report.get("signals", {})
    return {
        "symbol": report.get("metadata",{}).get("symbol"),
        "price": report.get("current_market_data",{}).get("price"),
        "trend": sig.get("trend"),
        "regime": sig.get("regime"),
        "rec": sig.get("recommendation"),
        "conf": sig.get("confidence"),
        "sl": sig.get("stop_loss"),
        "tp": sig.get("take_profit"),
        "ai_text_ar": report.get("ai",{}).get("nl_summary_ar")
    }

def save_json(obj: Dict[str,Any], path: str) -> bool:
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        log.info(f"Saved {path}"); return True
    except Exception as e:
        log.error(f"save_json error: {e}"); return False

def emit_webhook(url: Optional[str], payload: Dict[str,Any]) -> None:
    if not url: return
    try:
        r = requests.post(url, json=payload, timeout=10)
        log.info(f"Webhook status: {r.status_code}")
    except Exception as e:
        log.warning(f"Webhook post failed: {e}")

# ----------------------- Backtest (Pro) -----------------------
def backtest_engine(
    df: pd.DataFrame,
    ind: Dict[str, pd.Series],
    regime: pd.Series,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    - Rules:
      trend regime: استخدم إشارات MACD مع فلتر ADX، وتجاهل إشارات BB الضعيفة
      range regime: استخدم mean-reversion من BB/RSI وتخفيف MACD
    - Risk:
      حجم صفقة = risk_per_trade من رأس المال / مسافة الوقف (ATR*atr_sl)
      عمولة + انزلاق
      وقف متتالي ATR Trailing
    """
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
    commission = params.get("commission_perc", 0.0005)    # 5 bps
    slippage = params.get("slippage_perc", 0.0005)       # 5 bps
    risk_per_trade = params.get("risk_per_trade", 0.01)  # 1%
    equity = 1.0
    max_equity = equity

    in_pos = False; side=None; entry=None; size=0.0; sl=None; tp=None; trail=None
    trades=[]

    def apply_cost(price: float, is_buy: bool) -> float:
        # إدراج عمولة وانزلاق
        adj = price * (1 + commission + slippage) if is_buy else price * (1 - commission - slippage)
        return adj

    for i in range(2, len(close)):
        price = close.iat[i]
        reg_now = regime.iat[i] if i < len(regime) and pd.notna(regime.iat[i]) else "range"

        # إدارة الصفقة المفتوحة
        if in_pos:
            # وقف متتالي
            if side=="long":
                trail = max(trail, price - trail_mult*atr.iat[i]) if trail is not None else price - trail_mult*atr.iat[i]
                hit_trail = price <= trail
                hit_sl = price <= sl
                hit_tp = price >= tp
                exit_price = None
                if hit_trail or hit_sl or hit_tp:
                    exit_price = apply_cost(price, is_buy=False)
            else:
                trail = min(trail, price + trail_mult*atr.iat[i]) if trail is not None else price + trail_mult*atr.iat[i]
                hit_trail = price >= trail
                hit_sl = price >= sl
                hit_tp = price <= tp
                exit_price = None
                if hit_trail or hit_sl or hit_tp:
                    exit_price = apply_cost(price, is_buy=True)

            if exit_price is not None:
                pnl = (exit_price - entry) / entry if side=="long" else (entry - exit_price) / entry
                equity *= (1 + pnl * size)  # size يمثل نسبة التعرض من رأس المال
                trades.append({"i": i, "side": side, "entry": float(entry), "exit": float(exit_price), "size": float(size), "pnl_pct": float(pnl)})
                in_pos=False; side=None; entry=None; size=0.0; sl=None; tp=None; trail=None
                max_equity = max(max_equity, equity)

        # إشارات جديدة
        if in_pos:
            max_equity = max(max_equity, equity)
            continue

        # تحقق توفر المؤشرات
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
        else:  # range
            # mean-reversion: شراء قرب BB_L وRSI منخفض؛ بيع قرب BB_U وRSI مرتفع
            enter_long = (pd.notna(bb_l.iat[i]) and price <= bb_l.iat[i]) or (pd.notna(rsi.iat[i]) and rsi.iat[i] <= rsi_buy)
            enter_long = enter_long and not avoid_long
            enter_short = (pd.notna(bb_u.iat[i]) and price >= bb_u.iat[i]) or (pd.notna(rsi.iat[i]) and rsi.iat[i] >= rsi_sell)
            enter_short = enter_short and not avoid_short

        if enter_long or enter_short:
            # حجم الصفقة كنسبة مخاطرة ثابتة
            stop_dist = atr_sl_mult * atr.iat[i]
            if stop_dist <= 0:
                max_equity = max(max_equity, equity); continue
            # نسبة التعرض = المخاطرة / (مسافة الوقف / السعر)
            # تقريبًا: لو تحرك السعر بمقدار stop_dist ضدنا نخسر risk_per_trade من رأس المال
            size = min(1.0, max(0.0, risk_per_trade / (stop_dist / max(price, 1e-8))))
            if size <= 0: 
                max_equity = max(max_equity, equity); continue

            if enter_long:
                raw_entry = price
                entry = apply_cost(raw_entry, is_buy=True)
                side="long"
                sl = entry - atr_sl_mult*atr.iat[i]
                tp = entry + atr_tp_mult*atr.iat[i]
                trail = entry - trail_mult*atr.iat[i]
                in_pos=True
            elif enter_short:
                raw_entry = price
                entry = apply_cost(raw_entry, is_buy=False)
                side="short"
                sl = entry + atr_sl_mult*atr.iat[i]
                tp = entry - atr_tp_mult*atr.iat[i]
                trail = entry + trail_mult*atr.iat[i]
                in_pos=True

            max_equity = max(max_equity, equity)

    # Metrics
    rets = pd.Series([t["pnl_pct"]*t["size"] for t in trades], dtype=float)
    win = rets[rets>0]; loss = rets[rets<0]
    # Approx max drawdown from equity path
    eq_path=[1.0]; eq=1.0
    for r in rets:
        eq *= (1+r); eq_path.append(eq)
    path = pd.Series(eq_path)
    run_max = path.cummax(); dd_series = (path-run_max)/run_max
    dd = float(dd_series.min()) if len(dd_series)>0 else 0.0
    sharpe = float((rets.mean()/rets.std())*np.sqrt(252)) if len(rets)>2 and rets.std()!=0 else None
    downside = rets[rets<0]
    sortino = float((rets.mean()/downside.std())*np.sqrt(252)) if len(downside)>1 and downside.std()!=0 else None
    pf = float(abs(win.sum()/loss.sum())) if len(win)>0 and len(loss)>0 and loss.sum()!=0 else None

    perf = {
        "trades": int(len(trades)),
        "win_rate": float(len(win)/len(rets)) if len(rets)>0 else None,
        "profit_factor": pf,
        "cagr": None,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "final_equity": float(eq_path[-1]) if eq_path else 1.0
    }
    return {"performance": perf, "last_trades_sample": trades[-10:]}

# ----------------------- CLI ops -----------------------
def run_analyze(args):
    usd_candidates = args.usd if args.usd else ["DX-Y.NYB","^DXY","DXY","DX=F","USDX"]
    data = fetch_market(args.period, args.gold, usd_candidates, args.spy)
    report = build_report(data)
    save_json(report, args.out)
    if args.compact:
        save_json(to_compact(report), args.out.replace(".json","_compact.json"))
    if args.emit_webhook: emit_webhook(args.emit_webhook, report)
    print(f"✅ analyze done -> {args.out}{' + compact' if args.compact else ''}")

def run_backtest(args):
    data = fetch_market(args.period, args.gold, args.usd if args.usd else ["DX-Y.NYB","^DXY"], args.spy)
    gold = data.get("GC=F", pd.DataFrame())
    if gold.empty: print("❌ no gold data"); return
    gold = gold.copy(); gold.index = pd.to_datetime(gold.index).tz_localize(None)
    ind = calc_indicators(gold)
    regime = compute_regime(ind)
    params = {
        "adx_min": args.adx_min,
        "rsi_buy": args.rsi_buy,
        "rsi_sell": args.rsi_sell,
        "atr_mult_sl": args.atr_sl,
        "atr_mult_tp": args.atr_tp,
        "atr_trail_mult": args.atr_trail,
        "commission_perc": args.commission,
        "slippage_perc": args.slippage,
        "risk_per_trade": args.risk
    }
    res = backtest_engine(gold, ind, regime, params)
    out = {"metadata": {"version":"v6","symbol": args.gold,"period": args.period,"strategy":"Regime_MACD_BB_RSI_ATR_Pro"}, "params": params, **res}
    path = args.out.replace(".json","").replace("analysis","backtest") + ".json"
    save_json(out, path)
    if args.emit_webhook: emit_webhook(args.emit_webhook, out)
    print(f"✅ backtest done -> {path}")

def parse_args():
    p = argparse.ArgumentParser(description="Gold Analyzer V6 — Professional")
    p.add_argument("--mode", choices=["analyze","backtest"], default="analyze")
    p.add_argument("--period", default="1y")
    p.add_argument("--gold", default="GC=F")
    p.add_argument("--usd", nargs="*", default=["DX-Y.NYB","^DXY","DXY","DX=F","USDX"])
    p.add_argument("--spy", default="SPY")
    p.add_argument("--out", default="gold_analysis_v6.json")
    p.add_argument("--compact", action="store_true")
    p.add_argument("--emit-webhook", default=None)
    # backtest params (pro)
    p.add_argument("--adx-min", dest="adx_min", type=float, default=20.0)
    p.add_argument("--rsi-buy", dest="rsi_buy", type=float, default=35.0)
    p.add_argument("--rsi-sell", dest="rsi_sell", type=float, default=65.0)
    p.add_argument("--atr-sl", dest="atr_sl", type=float, default=2.0)
    p.add_argument("--atr-tp", dest="atr_tp", type=float, default=3.0)
    p.add_argument("--atr-trail", dest="atr_trail", type=float, default=1.5)
    p.add_argument("--commission", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--risk", type=float, default=0.01)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "analyze":
        run_analyze(args)
    else:
        run_backtest(args)

if __name__ == "__main__":
    main()
