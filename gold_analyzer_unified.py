#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Market Analyzer - Unified Stable
تحليل فني + محسّن + أساسي (FRED) + أخبار (NewsAPI) في ملف واحد JSON
"""

import os
import json
import time
import math
import warnings
import logging
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# اختياري: TA-Lib
try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

# إعدادات عامة
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("gold-unified")

# مفاتيح الـ API من Secrets (GitHub Actions → Settings → Secrets)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ============================= أدوات مساعدة =============================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            # تعامل مع NaN/inf
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        if isinstance(obj, (np.ndarray,)):
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
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
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

def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()

# ============================= جلب البيانات =============================

def fetch_yfinance(symbol: str, period: str = "1y", tries: int = 2, pause: float = 1.0) -> pd.DataFrame:
    """جلب بيانات من yfinance مع محاولات وتعافي"""
    import yfinance as yf
    last_err = None
    for i in range(tries):
        try:
            log.info(f"جاري جلب بيانات {symbol}...")
            data = yf.Ticker(symbol).history(period=period, auto_adjust=False)
            if not data.empty:
                data = data.dropna()
                # تأكد من الأعمدة المطلوبة
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in data.columns:
                        raise ValueError(f"Missing column {col}")
                log.info(f"تم جلب {len(data)} نقطة بيانات لـ {symbol}")
                return data
            last_err = Exception("Empty dataframe")
        except Exception as e:
            last_err = e
            log.warning(f"محاولة {i+1}/{tries} فشلت: {e}")
            time.sleep(pause)
    log.error(f"فشل جلب {symbol}: {last_err}")
    return pd.DataFrame()

def fetch_market_data(period: str = "1y") -> Dict[str, pd.DataFrame]:
    """GC=F الذهب + ^DXY الدولار + SPY للمقارنة"""
    return {
        "GC=F": fetch_yfinance("GC=F", period),
        "^DXY": fetch_yfinance("^DXY", period),
        "SPY": fetch_yfinance("SPY", period),
    }

# ============================= المؤشرات الفنية =============================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_macd(price: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(price, fast) - ema(price, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def technical_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    ind = {}
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # اتجاه
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

    # زخم
    if TALIB_AVAILABLE:
        ind["rsi"] = talib.RSI(close, timeperiod=14)
        macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd, macd_sig, macd_hist
        ind["williams_r"] = talib.WILLR(high, low, close, timeperiod=14)
        ind["cci"] = talib.CCI(high, low, close, timeperiod=14)
        ind["adx"] = talib.ADX(high, low, close, timeperiod=14)
        ind["trix"] = talib.TRIX(close, timeperiod=30)
        ind["ultosc"] = talib.ULTOSC(high, low, close)
        ind["stoch_k"], ind["stoch_d"] = talib.STOCH(high, low, close, 14, 3, 3)
    else:
        ind["rsi"] = calc_rsi(close, 14)
        macd, macd_sig, macd_hist = calc_macd(close)
        ind["macd"], ind["macd_signal"], ind["macd_hist"] = macd, macd_sig, macd_hist

    # تذبذب/نطاق
    if TALIB_AVAILABLE:
        upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ind["bb_upper"], ind["bb_middle"], ind["bb_lower"] = upper, mid, lower
        ind["atr"] = talib.ATR(high, low, close, timeperiod=14)
        ind["sar"] = talib.SAR(high, low)
        ind["dmi_plus"] = talib.PLUS_DI(high, low, close, timeperiod=14)
        ind["dmi_minus"] = talib.MINUS_DI(high, low, close, timeperiod=14)
        ind["obv"] = talib.OBV(close, vol)
        ind["mfi"] = talib.MFI(high, low, close, vol, timeperiod=14)
    else:
        # بولنجر بديل
        m = close.rolling(20).mean()
        s = close.rolling(20).std()
        ind["bb_upper"], ind["bb_middle"], ind["bb_lower"] = m + 2*s, m, m - 2*s
        # ATR بديل
        tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(14).mean()

    return ind

# ============================= الأنماط =============================

def detect_candles(df: pd.DataFrame) -> Dict[str, pd.Series]:
    if not TALIB_AVAILABLE:
        return {"note": "TA-Lib غير متوفر: كاشف الشموع غير مفعّل"}
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    return {
        "doji": talib.CDLDOJI(o, h, l, c),
        "hammer": talib.CDLHAMMER(o, h, l, c),
        "shooting_star": talib.CDLSHOOTINGSTAR(o, h, l, c),
        "engulfing": talib.CDLENGULFING(o, h, l, c),
        "morning_star": talib.CDLMORNINGSTAR(o, h, l, c),
        "evening_star": talib.CDLEVENINGSTAR(o, h, l, c),
    }

def detect_price_patterns(df: pd.DataFrame) -> Dict[str, List[int]]:
    # تبسيط—أماكن اكتشافات قابلة للتطوير
    highs = df["High"].rolling(5, center=True).max()
    lows = df["Low"].rolling(5, center=True).min()
    double_top, double_bottom = [], []
    for i in range(20, len(df)-20):
        try:
            if highs.iloc[i] == df["High"].iloc[i] and abs(df["High"].iloc[i]-df["High"].iloc[i-20:i].max()) < 0.01:
                double_top.append(i)
            if lows.iloc[i] == df["Low"].iloc[i] and abs(df["Low"].iloc[i]-df["Low"].iloc[i-20:i].min()) < 0.01:
                double_bottom.append(i)
        except Exception:
            pass
    return {
        "double_top": double_top,
        "double_bottom": double_bottom,
        "head_shoulders": [],
        "triangle": []
    }

# ============================= التحليل المحسّن =============================

def fibonacci_levels(low: float, high: float) -> Dict[str, float]:
    levels = {
        "0.0": low,
        "0.236": low + 0.236*(high-low),
        "0.382": low + 0.382*(high-low),
        "0.5": (low + high)/2,
        "0.618": low + 0.618*(high-low),
        "0.786": low + 0.786*(high-low),
        "1.0": high
    }
    ex = {
        "1.272": high + 0.272*(high-low),
        "1.618": high + 0.618*(high-low),
        "2.0": high + 1.0*(high-low),
        "2.618": high + 1.618*(high-low)
    }
    return {"levels": levels, "extensions": ex}

def support_resistance(df: pd.DataFrame) -> Dict:
    close = df["Close"]
    # دعم/مقاومة تاريخية بسيطة عبر القمم/القيعان البارزة
    window = 10
    local_max = close[(close.shift(1) < close) & (close.shift(-1) < close)]
    local_min = close[(close.shift(1) > close) & (close.shift(-1) > close)]
    supports = local_min.rolling(window).min().dropna().iloc[-20:].sort_values().unique().tolist()[-7:]
    resistances = local_max.rolling(window).max().dropna().iloc[-30:].sort_values().unique().tolist()[:8]

    # Pivot (كلاسيكي)
    recent = df.tail(20)
    P = (recent["High"].mean() + recent["Low"].mean() + recent["Close"].mean())/3
    R1 = 2*P - recent["Low"].mean()
    S1 = 2*P - recent["High"].mean()
    R2 = P + (recent["High"].mean()-recent["Low"].mean())
    S2 = P - (recent["High"].mean]-recent["Low"].mean())

    return {
        "pivot_points": {"pivot": P, "r1": R1, "s1": S1, "r2": R2, "s2": S2},
        "historical_support": supports,
        "historical_resistance": resistances,
    }

def volume_profile(df: pd.DataFrame, bins: int = 12) -> Dict:
    close = df["Close"]
    vol = df["Volume"].astype(float)
    try:
        cats = pd.cut(close, bins=bins)
        vp = vol.groupby(cats).sum().sort_values(ascending=False).head(5)
        profile = {str(k): int(v) for k, v in vp.items()}
    except Exception:
        profile = {}

    # اتجاه الحجم
    ret = safe_pct_change(close)
    vol_aligned = vol.reindex(ret.index)
    try:
        corr = np.corrcoef(ret.values, vol_aligned.values)[0,1]
        if math.isnan(corr):
            corr = None
    except Exception:
        corr = None

    up_vol = vol_aligned[ret > 0].mean() if not ret.empty else None
    down_vol = vol_aligned[ret < 0].mean() if not ret.empty else None
    v_trend = "مرتفع" if vol.iloc[-1] > vol.rolling(20).mean().iloc[-1]*1.5 else (
              "منخفض" if vol.iloc[-1] < vol.rolling(20).mean().iloc[-1]*0.5 else "عادي")

    return {
        "current_volume": int(vol.iloc[-1]),
        "average_volume": float(vol.rolling(252//12).mean().iloc[-1]) if len(vol) >= 21 else float(vol.mean()),
        "volume_ratio": float(vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] or 1.0)),
        "volume_profile": profile,
        "price_volume_correlation": corr,
        "uptrend_volume": float(up_vol) if up_vol is not None and not math.isnan(up_vol) else None,
        "downtrend_volume": float(down_vol) if down_vol is not None and not math.isnan(down_vol) else None,
        "volume_trend": v_trend
    }

def market_structure(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict:
    close = df["Close"]
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    troughs = (close.shift(1) > close) & (close.shift(-1) > close)
    peaks_count = int(peaks.sum())
    troughs_count = int(troughs.sum())

    sma20 = ind.get("sma_20", pd.Series(dtype=float))
    sma50 = ind.get("sma_50", pd.Series(dtype=float))
    sma200 = ind.get("sma_200", pd.Series(dtype=float))

    dirn = "متذبذب"
    if not sma20.empty and not sma50.empty and not sma200.empty:
        if close.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]:
            dirn = "اتجاه صاعد"
        elif close.iloc[-1] < sma20.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1]:
            dirn = "اتجاه هابط"

    rs = float((sma20.iloc[-1] / sma50.iloc[-1]) if (not sma20.empty and not sma50.empty and sma50.iloc[-1]!=0) else 1.0)
    conf = "عالية" if dirn != "متذبذب" else "متوسطة"
    return {
        "market_structure": dirn,
        "peak_trend": "صاعد" if close[peaks].diff().dropna().mean() and (close[peaks].diff().dropna().mean()>0) else "صاعد",
        "trough_trend": "صاعد" if close[troughs].diff().dropna().mean() and (close[troughs].diff().dropna().mean()>0) else "صاعد",
        "peaks_count": peaks_count,
        "troughs_count": troughs_count,
        "relative_strength": rs,
        "structure_confidence": conf
    }

def detect_divergences(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict:
    # تبسيط: نحسب اختلاف اتجاه آخر 20 شمعة بين السعر ومؤشرين
    def slope(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 3: return 0.0
        y = x.values
        X = np.arange(len(y))
        b = np.polyfit(X, y, 1)[0]
        return float(b)

    window = 20
    price_slope = slope(df["Close"].tail(window))
    rsi_slope = slope(ind.get("rsi", pd.Series(dtype=float)).tail(window))
    macd_slope = slope(ind.get("macd", pd.Series(dtype=float)).tail(window))

    rsi_div = (price_slope > 0 and rsi_slope < 0) or (price_slope < 0 and rsi_slope > 0)
    macd_div = (price_slope > 0 and macd_slope < 0) or (price_slope < 0 and macd_slope > 0)

    latest = []
    if rsi_div: latest.append({"type": "RSI", "window": window})
    if macd_div: latest.append({"type": "MACD", "window": window})

    return {
        "total_divergences": int(rsi_div) + int(macd_div),
        "recent_divergences": int(rsi_div) + int(macd_div),
        "positive_divergences": int(price_slope < 0 and (rsi_slope > 0 or macd_slope > 0)),
        "negative_divergences": int(price_slope > 0 and (rsi_slope < 0 or macd_slope < 0)),
        "rsi_divergences": int(rsi_div),
        "macd_divergences": int(macd_div),
        "latest_divergences": latest
    }

def correlation_block(data: Dict[str, pd.DataFrame]) -> Dict:
    try:
        gold = data.get("GC=F", pd.DataFrame())
        dxy = data.get("^DXY", pd.DataFrame())
        spy = data.get("SPY", pd.DataFrame())
        out = {}

        if not gold.empty:
            gr = safe_pct_change(gold["Close"])
            if not dxy.empty:
                xr = safe_pct_change(dxy["Close"])
                aligned = gr.align(xr, join="inner")
                if len(aligned[0]) > 5:
                    out["USD"] = float(np.corrcoef(aligned[0], aligned[1])[0,1])
            if not spy.empty:
                sr = safe_pct_change(spy["Close"])
                aligned = gr.align(sr, join="inner")
                if len(aligned[0]) > 5:
                    out["SPY"] = float(np.corrcoef(aligned[0], aligned[1])[0,1])
        if not out:
            return {"asset_correlations": {}, "time_correlation": None, "volume_correlation": None, "volatility_correlation": None,
                    "strongest_correlation": None, "weakest_correlation": None}
        strongest = max(out.items(), key=lambda kv: abs(kv[1]))
        weakest = min(out.items(), key=lambda kv: abs(kv[1]))
        return {
            "asset_correlations": out,
            "time_correlation": None,
            "volume_correlation": None,
            "volatility_correlation": None,
            "strongest_correlation": list(strongest),
            "weakest_correlation": list(weakest),
        }
    except Exception as e:
        log.warning(f"correlation error: {e}")
        return {"error": str(e)}

def advanced_metrics(df: pd.DataFrame) -> Dict:
    try:
        close = df["Close"]
        ret = safe_pct_change(close)
        if ret.empty:
            return {}

        ann_ret = float((1+ret.mean())**252 - 1)
        ann_vol = float(ret.std()*np.sqrt(252))
        rf = 0.02
        sharpe = float(((ret.mean()-rf/252)/ret.std())*np.sqrt(252)) if ret.std()!=0 else None

        # MDD
        equity = (1+ret).cumprod()
        run_max = equity.cummax()
        dd = (equity-run_max)/run_max
        mdd = float(dd.min()) if not dd.empty else None

        # Sortino
        downside = ret[ret < 0]
        d_vol = downside.std()*np.sqrt(252) if not downside.empty else None
        sortino = float((ret.mean()-rf/252)/(downside.std() if downside.std()!=0 else np.nan)*np.sqrt(252)) if d_vol not in [None, 0, np.nan] else None

        # Calmar
        calmar = float(ann_ret/abs(mdd)) if (mdd not in [None, 0] and not math.isnan(mdd)) else None

        # Win rate/Profit factor
        wins = ret[ret > 0]
        losses = ret[ret < 0]
        win_rate = float(len(wins)/len(ret)) if len(ret) else None
        avg_win = float(wins.mean()) if not wins.empty else None
        avg_loss = float(losses.mean()) if not losses.empty else None
        profit_factor = float(abs(wins.sum()/losses.sum())) if (not wins.empty and not losses.empty and losses.sum()!=0) else None

        return {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "var_95": float(np.percentile(ret, 5)),
            "var_99": float(np.percentile(ret, 1)),
            "skewness": float(ret.skew()),
            "kurtosis": float(ret.kurtosis()),
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor
        }
    except Exception as e:
        return {"error": str(e)}

# ============================= إشارات + مشاعر =============================

def technical_sentiment(df: pd.DataFrame, ind: Dict[str, pd.Series]) -> Dict:
    out = {}
    close = df["Close"].iloc[-1]
    rsi = ind.get("rsi", pd.Series(dtype=float))
    bb_u = ind.get("bb_upper", pd.Series(dtype=float))
    bb_l = ind.get("bb_lower", pd.Series(dtype=float))
    macd = ind.get("macd", pd.Series(dtype=float))
    macd_sig = ind.get("macd_signal", pd.Series(dtype=float))
    adx = ind.get("adx", pd.Series(dtype=float))
    vol_avg = df["Volume"].rolling(20).mean()

    # RSI
    rsi_now = rsi.iloc[-1] if not rsi.empty else 50.0
    out["rsi_sentiment"] = "مفرط في الشراء" if rsi_now > 70 else ("مفرط في البيع" if rsi_now < 30 else "محايد")

    # MACD
    if not macd.empty and not macd_sig.empty:
        out["macd_sentiment"] = "إيجابي" if macd.iloc[-1] >= macd_sig.iloc[-1] else "سلبي"

    # Bollinger
    if not bb_u.empty and not bb_l.empty:
        if close > bb_u.iloc[-1]:
            out["bb_sentiment"] = "مفرط في الشراء"
        elif close < bb_l.iloc[-1]:
            out["bb_sentiment"] = "مفرط في البيع"
        else:
            out["bb_sentiment"] = "عادي"

    # ADX
    if not adx.empty:
        out["trend_strength"] = "قوي" if adx.iloc[-1] > 25 else "ضعيف"

    # حجم
    v = df["Volume"].iloc[-1]
    if not vol_avg.empty and not math.isnan(vol_avg.iloc[-1]) and vol_avg.iloc[-1] != 0:
        out["volume_sentiment"] = "مرتفع" if v > 1.5*vol_avg.iloc[-1] else ("منخفض" if v < 0.5*vol_avg.iloc[-1] else "عادي")

    return out

def generate_signals(df: pd.DataFrame, ind: Dict[str, pd.Series], patterns_all: Dict) -> Dict:
    close = df["Close"].iloc[-1]
    out = {"current_price": float(close), "timestamp": now_iso()}

    sma20, sma50, sma200 = (ind.get("sma_20", pd.Series(dtype=float)),
                            ind.get("sma_50", pd.Series(dtype=float)),
                            ind.get("sma_200", pd.Series(dtype=float)))
    if not sma20.empty and not sma50.empty and not sma200.empty:
        if close > sma20.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]:
            out["trend"] = "صاعد قوي"
        elif close > sma20.iloc[-1] > sma50.iloc[-1]:
            out["trend"] = "صاعد"
        elif close < sma20.iloc[-1] < sma50.iloc[-1] < sma200.iloc[-1]:
            out["trend"] = "هابط قوي"
        elif close < sma20.iloc[-1] < sma50.iloc[-1]:
            out["trend"] = "هابط"
        else:
            out["trend"] = "متذبذب"

    # RSI signal
    rsi = ind.get("rsi", pd.Series(dtype=float))
    if not rsi.empty:
        rv = rsi.iloc[-1]
        out["rsi_signal"] = "شراء قوي" if rv < 30 else ("شراء" if rv < 40 else ("بيع قوي" if rv > 70 else ("بيع" if rv > 60 else "محايد")))

    # MACD cross
    macd = ind.get("macd", pd.Series(dtype=float))
    macd_sig = ind.get("macd_signal", pd.Series(dtype=float))
    if not macd.empty and not macd_sig.empty and len(macd) > 2:
        if macd.iloc[-1] > macd_sig.iloc[-1] and macd.iloc[-2] <= macd_sig.iloc[-2]:
            out["macd_signal"] = "شراء"
        elif macd.iloc[-1] < macd_sig.iloc[-1] and macd.iloc[-2] >= macd_sig.iloc[-2]:
            out["macd_signal"] = "بيع"
        else:
            out["macd_signal"] = "محايد"

    # Bollinger
    bbu = ind.get("bb_upper", pd.Series(dtype=float))
    bbl = ind.get("bb_lower", pd.Series(dtype=float))
    if not bbu.empty and not bbl.empty:
        out["bb_signal"] = "شراء" if close < bbl.iloc[-1] else ("بيع" if close > bbu.iloc[-1] else "محايد")

    # أنماط (شموع/سعر) – إذا آخر شمعة فيها إشارة
    pt_signals = []
    for name, val in patterns_all.items():
        if isinstance(val, pd.Series) and not val.empty:
            if val.iloc[-1] != 0:
                pt_signals.append(name)
        elif isinstance(val, list) and len(val)>0 and val[-1] == len(df)-1:
            pt_signals.append(name)
    out["patterns"] = pt_signals

    # توصية ختامية موزونة
    buy, sell = 0.0, 0.0
    if out.get("rsi_signal") in ["شراء قوي", "شراء"]: buy += 1
    if out.get("rsi_signal") in ["بيع قوي", "بيع"]:  sell += 1
    if out.get("macd_signal") == "شراء": buy += 1
    if out.get("macd_signal") == "بيع":  sell += 1
    if out.get("bb_signal") == "شراء":   buy += 1
    if out.get("bb_signal") == "بيع":    sell += 1
    if out.get("trend") in ["صاعد", "صاعد قوي"]: buy += 0.5
    if out.get("trend") in ["هابط", "هابط قوي"]: sell += 0.5

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

    # إدارة المخاطر عبر ATR
    atr = ind.get("atr", pd.Series(dtype=float))
    if not atr.empty and not math.isnan(atr.iloc[-1]):
        out["risk_level"] = "عالية" if atr.iloc[-1]/close > 0.015 else ("متوسطة" if atr.iloc[-1]/close > 0.01 else "منخفضة")
        out["stop_loss"] = float(close - 2*atr.iloc[-1])
        out["take_profit"] = float(close + 3*atr.iloc[-1])
    else:
        out["risk_level"] = "متوسطة"
        out["stop_loss"] = float(close*0.95)
        out["take_profit"] = float(close*1.08)

    return out

# ============================= أخبار (NewsAPI) =============================

def simple_sentiment_score(text: str) -> float:
    """تحليل مشاعر بسيط  -1..+1"""
    if not text:
        return 0.0
    text = text.lower()
    pos_kw = ["rally", "surge", "gain", "safe haven", "bull", "up", "beat", "strong"]
    neg_kw = ["drop", "fall", "slump", "risk", "hawkish", "down", "miss", "weak", "selloff"]
    score = 0
    for w in pos_kw:
        if w in text:
            score += 1
    for w in neg_kw:
        if w in text:
            score -= 1
    return max(-1.0, min(1.0, score/3.0))

def fetch_news(news_api_key: str, q: str = "gold OR XAU OR bullion OR GC=F", page_size: int = 30) -> Dict:
    if not news_api_key:
        return {"error": "NEWS_API_KEY مفقود"}
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": news_api_key,
    }
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
            s = simple_sentiment_score(f"{title}. {desc}")
            total += s
            parsed.append({
                "title": title,
                "source": (a.get("source") or {}).get("name"),
                "publishedAt": a.get("publishedAt"),
                "url": a.get("url"),
                "sentiment": s
            })
        avg = total/max(1, len(parsed))
        # تقدير التأثير (بسيط)
        impact = "داعم للذهب" if avg > 0.2 else ("ضاغط على الذهب" if avg < -0.2 else "محايد")
        return {"count": len(parsed), "average_sentiment": avg, "impact": impact, "articles": parsed}
    except Exception as e:
        return {"error": str(e)}

# ============================= بيانات أساسية (FRED) =============================

FRED_SERIES = {
    "FEDFUNDS": "Effective Federal Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers, SA)",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index: Broad, Goods",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "UNRATE": "Unemployment Rate",
}

def fetch_fred_series(series_id: str, api_key: str, obs: int = 24) -> Optional[pd.Series]:
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": (dt.date.today()-dt.timedelta(days=4000)).isoformat()}
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        obs_list = js.get("observations", [])
        if not obs_list:
            return None
        df = pd.DataFrame(obs_list)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["value"].dropna().tail(obs)
        return s
    except Exception:
        return None

def fundamental_block(api_key: str) -> Dict:
    if not api_key:
        return {"error": "FRED_API_KEY مفقود"}
    out = {}
    for sid, name in FRED_SERIES.items():
        s = fetch_fred_series(sid, api_key)
        if s is None or s.empty:
            out[sid] = {"series": sid, "name": name, "error": "no data"}
            continue
        latest = clean_scalar(s.iloc[-1])
        prev_12 = clean_scalar(s.shift(12).iloc[-1]) if len(s) > 12 else None
        yoy = None
        if latest is not None and prev_12 not in [None, 0]:
            yoy = float((latest - prev_12)/prev_12) if prev_12 else None
        out[sid] = {
            "series": sid,
            "name": name,
            "latest": latest,
            "yoy_change": yoy
        }
    # تقييم مبسّط لتأثير الأساسيات على الذهب:
    try:
        fed = out.get("FEDFUNDS", {})
        dxy = out.get("DTWEXBGS", {})
        cpi = out.get("CPIAUCSL", {})
        bias = 0
        if fed.get("latest") is not None and fed["latest"] > 3: bias -= 1  # تشدد نقدي سلبي للذهب
        if dxy.get("yoy_change") is not None and dxy["yoy_change"] > 0.05: bias -= 1  # قوة الدولار سلبية
        if cpi.get("yoy_change") is not None and cpi["yoy_change"] > 0.03: bias += 1  # تضخم داعم
        stance = "داعم للذهب" if bias > 0 else ("ضاغط على الذهب" if bias < 0 else "محايد")
        out["fundamental_bias"] = stance
    except Exception:
        out["fundamental_bias"] = "محايد"
    return out

# ============================= إنشاء التقرير =============================

def build_report(data: Dict[str, pd.DataFrame]) -> Dict:
    gold = data.get("GC=F", pd.DataFrame())
    if gold.empty:
        return {"error": "لا توجد بيانات GC=F"}

    ind = technical_indicators(gold)
    candles = detect_candles(gold) if TALIB_AVAILABLE else {}
    price_pats = detect_price_patterns(gold)
    patt_all = {**candles, **price_pats} if candles else price_pats

    # محسّنات
    lo, hi = float(gold["Low"].min()), float(gold["High"].max())
    fibo = fibonacci_levels(lo, hi)
    sr = support_resistance(gold)
    vp = volume_profile(gold)
    ms = market_structure(gold, ind)
    divs = detect_divergences(gold, ind)
    corr = correlation_block(data)
    adv = advanced_metrics(gold)

    # مشاعر وإشارات
    senti = technical_sentiment(gold, ind)
    sig = generate_signals(gold, ind, patt_all)

    # أساسيات + أخبار
    fred = fundamental_block(FRED_API_KEY)
    news = fetch_news(NEWS_API_KEY)

    report = {
        "metadata": {
            "version": "unified-stable",
            "symbol": "GC=F",
            "period": "1y",
            "analysis_date": now_iso(),
            "data_points": int(gold.shape[0]),
            "talib": TALIB_AVAILABLE
        },
        "current_market_data": {
            "current_price": clean_scalar(gold["Close"].iloc[-1]),
            "daily_change": clean_scalar(gold["Close"].iloc[-1] - gold["Close"].iloc[-2]) if gold.shape[0] > 1 else None,
            "daily_change_percent": clean_scalar(((gold["Close"].iloc[-1] / gold["Close"].iloc[-2]) - 1) * 100) if gold.shape[0] > 1 else None,
            "volume": clean_scalar(gold["Volume"].iloc[-1]),
            "high": clean_scalar(gold["High"].iloc[-1]),
            "low": clean_scalar(gold["Low"].iloc[-1]),
        },
        "signals": convert_numpy_types(sig),
        "technical_indicators": convert_numpy_types({
            "rsi": ind.get("rsi", pd.Series(dtype=float)).iloc[-1] if not ind.get("rsi", pd.Series(dtype=float)).empty else None,
            "macd": ind.get("macd", pd.Series(dtype=float)).iloc[-1] if not ind.get("macd", pd.Series(dtype=float)).empty else None,
            "macd_signal": ind.get("macd_signal", pd.Series(dtype=float)).iloc[-1] if not ind.get("macd_signal", pd.Series(dtype=float)).empty else None,
            "sma_20": ind.get("sma_20", pd.Series(dtype=float)).iloc[-1] if not ind.get("sma_20", pd.Series(dtype=float)).empty else None,
            "sma_50": ind.get("sma_50", pd.Series(dtype=float)).iloc[-1] if not ind.get("sma_50", pd.Series(dtype=float)).empty else None,
            "sma_200": ind.get("sma_200", pd.Series(dtype=float)).iloc[-1] if not ind.get("sma_200", pd.Series(dtype=float)).empty else None,
            "bb_upper": ind.get("bb_upper", pd.Series(dtype=float)).iloc[-1] if not ind.get("bb_upper", pd.Series(dtype=float)).empty else None,
            "bb_lower": ind.get("bb_lower", pd.Series(dtype=float)).iloc[-1] if not ind.get("bb_lower", pd.Series(dtype=float)).empty else None,
            "atr": ind.get("atr", pd.Series(dtype=float)).iloc[-1] if not ind.get("atr", pd.Series(dtype=float)).empty else None
        }),
        "patterns": {k: (v.tolist() if isinstance(v, pd.Series) else v) for k, v in patt_all.items()},
        "enhancements": {
            "fibonacci_analysis": {
                **fibo,
                "current_price": clean_scalar(gold["Close"].iloc[-1]),
                "nearest_support": min(fibo["levels"].keys(), key=lambda k: abs(fibo["levels"][k] - gold["Close"].iloc[-1] )) if fibo["levels"] else None,
                "nearest_resistance": "1.0" if fibo["levels"].get("1.0", 0) >= gold["Close"].iloc[-1] else None,
            },
            "support_resistance_analysis": {**sr, "current_price": clean_scalar(gold["Close"].iloc[-1])},
            "volume_profile_analysis": vp,
            "market_structure_analysis": ms,
            "divergence_analysis": divs,
            "correlation_analysis": corr,
            "advanced_metrics": adv
        },
        "sentiment": convert_numpy_types(senti),
        "fundamentals_fred": fred,
        "news_analysis": news,
        "summary": {
            "overall_recommendation": sig.get("recommendation"),
            "confidence_level": sig.get("confidence"),
            "risk_level": sig.get("risk_level"),
            "trend_direction": sig.get("trend"),
            "key_support": sig.get("stop_loss"),
            "key_resistance": sig.get("take_profit"),
            "fundamental_bias": fred.get("fundamental_bias", "محايد"),
            "news_impact": news.get("impact") if isinstance(news, dict) else None
        }
    }
    return convert_numpy_types(report)

# ============================= حفظ وتشغيل =============================

def save_report(report: Dict, filename: str = "gold_analysis_unified.json") -> bool:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        log.info(f"تم حفظ التقرير في {filename}")
        return True
    except Exception as e:
        log.error(f"خطأ في حفظ التقرير: {e}")
        return False

def main():
    log.info("بدء التحليل الموحد للذهب (نسخة مستقرة)...")
    data = fetch_market_data(period="1y")
    gold = data.get("GC=F", pd.DataFrame())
    if gold.empty:
        log.error("فشل الحصول على بيانات GC=F. سيتم إنهاء العملية.")
        report = {"error": "GC=F data unavailable"}
        save_report(report)
        print("❌ فشل التحليل - لا توجد بيانات GC=F")
        return

    report = build_report(data)
    if not report:
        log.error("فشل بناء التقرير")
        print("❌ فشل بناء التقرير")
        return

    if save_report(report):
        log.info("تم إكمال التحليل الموحد بنجاح!")
        print("✅ تم حفظ التقرير: gold_analysis_unified.json")
    else:
        print("⚠️ تم إنشاء التقرير لكن فشل الحفظ")

if __name__ == "__main__":
    main()
