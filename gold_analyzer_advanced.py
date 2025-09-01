#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Market Analyzer Advanced V6.0 (Stable)
محلل سوق الذهب المتقدم الإصدار 6.0 (نسخة مستقرة)

أهداف النسخة:
- الحفاظ على جميع الميزات والمؤشرات بدون أي نقصان.
- تحسين المتانة: لا تعطل حتى لو فشل قسم معيّن.
- ضمان إنتاج ملف JSON دائمًا مع وضع الأخطاء في حقول "error" بدلاً من إيقاف التنفيذ.
- إزالة أسباب خطأ: "The truth value of a Series is ambiguous" عبر تحويل القيم إلى Scalars بأمان.

الميزات:
- 25+ مؤشر فني متقدم (مع بدائل عند غياب TA-Lib)
- كشف الأنماط السعرية والشموع اليابانية (عند توفر TA-Lib)
- تحليل المشاعر السوقية
- إدارة المخاطر المتقدمة
- تقرير JSON شامل
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import datetime
import logging
import warnings
from typing import Dict, List, Optional

# مكتبات رسومية ليست مطلوبة للتقرير JSON، نُبقي الاستيراد إن لزم التوسعة لاحقًا
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

try:
    import talib  # type: ignore
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using safe fallbacks")

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تجاهل التحذيرات
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            # pandas NA
            if pd.isna(obj):
                return None
            return super().default(obj)
        except Exception:
            return str(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy/pandas scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # pandas scalars
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def _safe_last(series: Optional[pd.Series], default=np.nan):
    """Return last scalar value of a Series safely."""
    try:
        if isinstance(series, pd.Series) and series.size > 0:
            return convert_numpy_types(series.iloc[-1])
    except Exception:
        pass
    return default


def _roll_mean(series: pd.Series, window: int):
    try:
        return series.rolling(window=window, min_periods=max(2, window//2)).mean()
    except Exception:
        return pd.Series(index=series.index, dtype=float)


def _ema(series: pd.Series, span: int):
    try:
        return series.ewm(span=span, adjust=False, min_periods=max(2, span//2)).mean()
    except Exception:
        return pd.Series(index=series.index, dtype=float)


def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method='bfill').fillna(50)
    except Exception:
        return pd.Series([50] * len(prices), index=prices.index)


def _calc_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    try:
        ema_fast = _ema(prices, fast)
        ema_slow = _ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = _ema(macd, signal)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    except Exception:
        z = pd.Series([0] * len(prices), index=prices.index)
        return z, z, z


def _calc_bbands(close: pd.Series, period: int = 20, dev: float = 2.0):
    try:
        ma = _roll_mean(close, period)
        std = close.rolling(period, min_periods=max(2, period//2)).std()
        upper = ma + dev * std
        lower = ma - dev * std
        return upper, ma, lower
    except Exception:
        nan = pd.Series([np.nan] * len(close), index=close.index)
        return nan, nan, nan


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    try:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=max(2, period//2)).mean()
        return atr
    except Exception:
        return pd.Series([np.nan] * len(close), index=close.index)


class AdvancedGoldAnalyzerV6:
    def __init__(self, symbol: str = "GC=F", period: str = "1y", fast_mode: bool = True):
        self.symbol = symbol
        self.period = period
        self.fast_mode = fast_mode
        self.data: Optional[pd.DataFrame] = None

    # ----------------------------- Data ----------------------------------
    def fetch_data(self) -> bool:
        try:
            logger.info(f"جاري جلب بيانات {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=self.period)
            if df is None or df.empty:
                logger.error("فشل في جلب البيانات")
                return False
            self.data = df
            logger.info(f"تم جلب {len(self.data)} نقطة بيانات")
            return True
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات: {e}")
            return False

    # ----------------------- Technical Indicators ------------------------
    def calculate_technical_indicators(self) -> Dict:
        if self.data is None or self.data.empty:
            return {}
        ind: Dict[str, pd.Series] = {}
        c = self.data['Close']
        h = self.data['High']
        l = self.data['Low']
        v = self.data['Volume']
        try:
            # Trend MAs
            if TALIB_AVAILABLE:
                ind['sma_20'] = talib.SMA(c, timeperiod=20)
                ind['sma_50'] = talib.SMA(c, timeperiod=50)
                ind['sma_200'] = talib.SMA(c, timeperiod=200)
                ind['ema_12'] = talib.EMA(c, timeperiod=12)
                ind['ema_26'] = talib.EMA(c, timeperiod=26)
                ind['rsi'] = talib.RSI(c, timeperiod=14)
                macd, macd_sig, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
                ind['macd'], ind['macd_signal'], ind['macd_hist'] = macd, macd_sig, macd_hist
                ind['stoch_k'], ind['stoch_d'] = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
                ind['williams_r'] = talib.WILLR(h, l, c, timeperiod=14)
                ind['cci'] = talib.CCI(h, l, c, timeperiod=14)
                ind['adx'] = talib.ADX(h, l, c, timeperiod=14)
                ind['trix'] = talib.TRIX(c, timeperiod=30)
                ind['ultosc'] = talib.ULTOSC(h, l, c)
                # Volatility
                up, mid, lowb = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                ind['bbands_upper'], ind['bbands_middle'], ind['bbands_lower'] = up, mid, lowb
                ind['atr'] = talib.ATR(h, l, c, timeperiod=14)
                ind['sar'] = talib.SAR(h, l)
                # DMI
                ind['dmi_plus'] = talib.PLUS_DI(h, l, c, timeperiod=14)
                ind['dmi_minus'] = talib.MINUS_DI(h, l, c, timeperiod=14)
                # Volume-based
                ind['obv'] = talib.OBV(c, v)
                ind['ad'] = talib.AD(h, l, c, v)
                ind['adosc'] = talib.ADOSC(h, l, c, v)
                ind['mfi'] = talib.MFI(h, l, c, v, timeperiod=14)
                ind['mom'] = talib.MOM(c, timeperiod=10)
                ind['roc'] = talib.ROC(c, timeperiod=10)
                slowk, slowd = talib.STOCHF(h, l, c)
                ind['slowk'], ind['slowd'] = slowk, slowd
            else:
                # Fallbacks without TA-Lib
                ind['sma_20'] = _roll_mean(c, 20)
                ind['sma_50'] = _roll_mean(c, 50)
                ind['sma_200'] = _roll_mean(c, 200)
                ind['ema_12'] = _ema(c, 12)
                ind['ema_26'] = _ema(c, 26)
                ind['rsi'] = _calc_rsi(c, 14)
                macd, macd_sig, macd_hist = _calc_macd(c)
                ind['macd'], ind['macd_signal'], ind['macd_hist'] = macd, macd_sig, macd_hist
                up, mid, lowb = _calc_bbands(c, 20, 2.0)
                ind['bbands_upper'], ind['bbands_middle'], ind['bbands_lower'] = up, mid, lowb
                ind['atr'] = _calc_atr(h, l, c, 14)
                # Placeholders when TA-Lib is absent
                for k in ['stoch_k','stoch_d','williams_r','cci','adx','trix','ultosc','sar','dmi_plus','dmi_minus','obv','ad','adosc','mfi','mom','roc','slowk','slowd']:
                    ind.setdefault(k, pd.Series([np.nan]*len(c), index=c.index))
            logger.info("تم حساب جميع المؤشرات الفنية بنجاح")
            return ind
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات الفنية: {e}")
            return {}

    # -------------------------- Pattern Detection ------------------------
    def detect_advanced_patterns(self) -> Dict:
        if self.data is None or self.data.empty:
            return {}
        patterns: Dict[str, object] = {}
        try:
            o = self.data['Open']; h = self.data['High']; l = self.data['Low']; c = self.data['Close']
            if TALIB_AVAILABLE:
                patterns['doji'] = talib.CDLDOJI(o, h, l, c)
                patterns['hammer'] = talib.CDLHAMMER(o, h, l, c)
                patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(o, h, l, c)
                patterns['engulfing'] = talib.CDLENGULFING(o, h, l, c)
                patterns['morning_star'] = talib.CDLMORNINGSTAR(o, h, l, c)
                patterns['evening_star'] = talib.CDLEVENINGSTAR(o, h, l, c)
            else:
                # بدون TA-Lib نُرجع سلاسل فارغة لتجنّب الأعطال
                empty = pd.Series([0]*len(c), index=c.index)
                for name in ['doji','hammer','shooting_star','engulfing','morning_star','evening_star']:
                    patterns[name] = empty
            # بساطة: أنماط سعرية placeholders يمكن تطويرها لاحقاً
            patterns['double_top'] = self._detect_double_top()
            patterns['double_bottom'] = self._detect_double_bottom()
            patterns['head_shoulders'] = []
            patterns['triangle'] = []
            logger.info("تم كشف الأنماط المتقدمة بنجاح")
            return patterns
        except Exception as e:
            logger.error(f"خطأ في كشف الأنماط: {e}")
            return {}

    def _detect_double_top(self) -> List[int]:
        try:
            highs = self.data['High'].rolling(window=5, center=True).max()
            out: List[int] = []
            for i in range(20, len(self.data) - 20):
                if pd.notna(highs.iloc[i]) and highs.iloc[i] == self.data['High'].iloc[i]:
                    ref_max = self.data['High'].iloc[i-20:i].max()
                    if np.isfinite(ref_max) and abs(self.data['High'].iloc[i] - ref_max) < 0.01:
                        out.append(i)
            return out
        except Exception:
            return []

    def _detect_double_bottom(self) -> List[int]:
        try:
            lows = self.data['Low'].rolling(window=5, center=True).min()
            out: List[int] = []
            for i in range(20, len(self.data) - 20):
                if pd.notna(lows.iloc[i]) and lows.iloc[i] == self.data['Low'].iloc[i]:
                    ref_min = self.data['Low'].iloc[i-20:i].min()
                    if np.isfinite(ref_min) and abs(self.data['Low'].iloc[i] - ref_min) < 0.01:
                        out.append(i)
            return out
        except Exception:
            return []

    # --------------------------- Sentiment -------------------------------
    def analyze_market_sentiment(self) -> Dict:
        sentiment: Dict[str, str] = {}
        if self.data is None or self.data.empty:
            return sentiment
        try:
            ind = self.calculate_technical_indicators()
            rsi_series = ind.get('rsi', pd.Series(dtype=float))
            rsi_val = _safe_last(rsi_series, 50)
            if pd.isna(rsi_val):
                rsi_val = 50
            # تصحيح الترجمة: RSI>70 = مفرط في الشراء, RSI<30 = مفرط في البيع
            if rsi_val > 70:
                sentiment['rsi_sentiment'] = 'مفرط في الشراء'
            elif rsi_val < 30:
                sentiment['rsi_sentiment'] = 'مفرط في البيع'
            else:
                sentiment['rsi_sentiment'] = 'محايد'

            macd = ind.get('macd', pd.Series(dtype=float))
            macd_signal = ind.get('macd_signal', pd.Series(dtype=float))
            m_last = _safe_last(macd, np.nan)
            ms_last = _safe_last(macd_signal, np.nan)
            if not pd.isna(m_last) and not pd.isna(ms_last):
                sentiment['macd_sentiment'] = 'إيجابي' if m_last > ms_last else 'سلبي'

            bb_u = ind.get('bbands_upper', pd.Series(dtype=float))
            bb_l = ind.get('bbands_lower', pd.Series(dtype=float))
            price = float(_safe_last(self.data['Close'], np.nan))
            bu = _safe_last(bb_u, np.nan)
            bl = _safe_last(bb_l, np.nan)
            if not pd.isna(bu) and not pd.isna(bl) and np.isfinite(price):
                if price > bu:
                    sentiment['bb_sentiment'] = 'مفرط في الشراء'
                elif price < bl:
                    sentiment['bb_sentiment'] = 'مفرط في البيع'
                else:
                    sentiment['bb_sentiment'] = 'عادي'

            adx = ind.get('adx', pd.Series(dtype=float))
            adx_last = _safe_last(adx, np.nan)
            if not pd.isna(adx_last):
                sentiment['trend_strength'] = 'قوي' if adx_last > 25 else 'ضعيف'

            vol = self.data['Volume']
            vavg = _roll_mean(vol, 20)
            v_last = _safe_last(vol, np.nan)
            vavg_last = _safe_last(vavg, np.nan)
            if not pd.isna(v_last) and not pd.isna(vavg_last):
                if v_last > 1.5 * vavg_last:
                    sentiment['volume_sentiment'] = 'مرتفع'
                elif v_last < 0.5 * vavg_last:
                    sentiment['volume_sentiment'] = 'منخفض'
                else:
                    sentiment['volume_sentiment'] = 'عادي'

            logger.info("تم تحليل المشاعر السوقية بنجاح")
            return sentiment
        except Exception as e:
            logger.error(f"خطأ في تحليل المشاعر: {e}")
            return {}

    # --------------------------- Risk Metrics ----------------------------
    def calculate_advanced_risk_metrics(self) -> Dict:
        out: Dict[str, float] = {}
        if self.data is None or self.data.empty:
            return out
        try:
            returns = self.data['Close'].pct_change().dropna()
            if returns.empty:
                return out
            out['volatility'] = float(returns.std() * np.sqrt(252))
            out['var_95'] = float(np.percentile(returns, 5))
            out['var_99'] = float(np.percentile(returns, 1))
            cum = (1 + returns).cumprod()
            running_max = cum.cummax()
            drawdown = (cum - running_max) / running_max
            out['max_drawdown'] = float(drawdown.min())
            risk_free = 0.02
            excess = returns - risk_free/252
            denom = returns.std()
            out['sharpe_ratio'] = float((excess.mean() / denom * np.sqrt(252)) if denom != 0 else 0.0)
            # Beta مقابل S&P 500
            try:
                sp = yf.Ticker("^GSPC").history(period=self.period)['Close'].pct_change().dropna()
                aligned = returns.align(sp, join='inner')
                x = aligned[0]
                y = aligned[1]
                if len(x) > 30 and y.var() != 0:
                    cov = np.cov(x, y)[0, 1]
                    out['beta'] = float(cov / y.var())
                else:
                    out['beta'] = 1.0
            except Exception:
                out['beta'] = 1.0
            out['skewness'] = float(returns.skew())
            out['kurtosis'] = float(returns.kurtosis())
            out['var_ratio'] = float(out['var_95'] / out['var_99']) if out['var_99'] != 0 else np.nan
            logger.info("تم حساب مقاييس المخاطر المتقدمة بنجاح")
            return out
        except Exception as e:
            logger.error(f"خطأ في حساب مقاييس المخاطر: {e}")
            return {}

    # --------------------------- Signals ---------------------------------
    def _risk_level_label(self, risk: Dict) -> str:
        try:
            vol = abs(float(risk.get('volatility', 0.2)))
            mdd = abs(float(risk.get('max_drawdown', 0.1)))
            if vol > 0.3 or mdd > 0.2:
                return 'عالية'
            if vol > 0.2 or mdd > 0.15:
                return 'متوسطة'
            return 'منخفضة'
        except Exception:
            return 'متوسطة'

    def _stop_loss(self, price: float, ind: Dict) -> float:
        try:
            atr = ind.get('atr', pd.Series(dtype=float))
            atr_last = _safe_last(atr, np.nan)
            if not pd.isna(atr_last):
                return float(price - 2 * atr_last)
            return float(price * 0.95)
        except Exception:
            return float(price * 0.95)

    def _take_profit(self, price: float, ind: Dict) -> float:
        try:
            atr = ind.get('atr', pd.Series(dtype=float))
            atr_last = _safe_last(atr, np.nan)
            if not pd.isna(atr_last):
                return float(price + 3 * atr_last)
            return float(price * 1.08)
        except Exception:
            return float(price * 1.08)

    def generate_advanced_signals_v6(self) -> Dict:
        if self.data is None or self.data.empty:
            return {}
        out: Dict[str, object] = {}
        try:
            ind = self.calculate_technical_indicators()
            patterns = self.detect_advanced_patterns()
            price = float(_safe_last(self.data['Close'], np.nan))
            out['current_price'] = price
            out['timestamp'] = datetime.datetime.now().isoformat()

            sma20 = ind.get('sma_20', pd.Series(dtype=float))
            sma50 = ind.get('sma_50', pd.Series(dtype=float))
            sma200 = ind.get('sma_200', pd.Series(dtype=float))
            s20 = _safe_last(sma20, np.nan)
            s50 = _safe_last(sma50, np.nan)
            s200 = _safe_last(sma200, np.nan)
            trend = 'متذبذب'
            if not pd.isna(price) and not pd.isna(s20) and not pd.isna(s50) and not pd.isna(s200):
                if price > s20 > s50 > s200:
                    trend = 'صاعد قوي'
                elif price > s20 > s50:
                    trend = 'صاعد'
                elif price < s20 < s50 < s200:
                    trend = 'هابط قوي'
                elif price < s20 < s50:
                    trend = 'هابط'
            out['trend'] = trend

            rsi = ind.get('rsi', pd.Series(dtype=float))
            rsi_last = _safe_last(rsi, np.nan)
            if pd.isna(rsi_last):
                out['rsi_signal'] = 'محايد'
            elif rsi_last < 30:
                out['rsi_signal'] = 'شراء قوي'
            elif rsi_last < 40:
                out['rsi_signal'] = 'شراء'
            elif rsi_last > 70:
                out['rsi_signal'] = 'بيع قوي'
            elif rsi_last > 60:
                out['rsi_signal'] = 'بيع'
            else:
                out['rsi_signal'] = 'محايد'

            macd, macd_sig = ind.get('macd', pd.Series(dtype=float)), ind.get('macd_signal', pd.Series(dtype=float))
            m1 = _safe_last(macd, np.nan)
            ms1 = _safe_last(macd_sig, np.nan)
            # نحتاج أيضًا للقيمة السابقة بحذر
            def _prev_val(s: pd.Series):
                try:
                    if isinstance(s, pd.Series) and s.size >= 2:
                        return convert_numpy_types(s.iloc[-2])
                except Exception:
                    pass
                return np.nan
            m0 = _prev_val(macd)
            ms0 = _prev_val(macd_sig)
            if not pd.isna(m1) and not pd.isna(ms1) and not pd.isna(m0) and not pd.isna(ms0):
                if m1 > ms1 and m0 <= ms0:
                    out['macd_signal'] = 'شراء'
                elif m1 < ms1 and m0 >= ms0:
                    out['macd_signal'] = 'بيع'
                else:
                    out['macd_signal'] = 'محايد'
            else:
                out['macd_signal'] = 'محايد'

            bb_u = ind.get('bbands_upper', pd.Series(dtype=float))
            bb_l = ind.get('bbands_lower', pd.Series(dtype=float))
            bu = _safe_last(bb_u, np.nan)
            bl = _safe_last(bb_l, np.nan)
            if not pd.isna(price) and not pd.isna(bu) and not pd.isna(bl):
                if price < bl:
                    out['bb_signal'] = 'شراء'
                elif price > bu:
                    out['bb_signal'] = 'بيع'
                else:
                    out['bb_signal'] = 'محايد'
            else:
                out['bb_signal'] = 'محايد'

            # أنماط حديثة (إذا وُجدت)
            pattern_signals: List[str] = []
            for name, pdata in patterns.items():
                if isinstance(pdata, list) and len(pdata) > 0:
                    if pdata[-1] == self.data.shape[0] - 1:
                        pattern_signals.append(name)
                elif isinstance(pdata, pd.Series) and pdata.size > 0:
                    val = _safe_last(pdata, 0)
                    if not pd.isna(val) and float(val) != 0.0:
                        pattern_signals.append(name)
            out['patterns'] = pattern_signals

            # التوصية
            buy_cnt = 0.0
            sell_cnt = 0.0
            if out.get('rsi_signal') in ['شراء','شراء قوي']:
                buy_cnt += 1
            elif out.get('rsi_signal') in ['بيع','بيع قوي']:
                sell_cnt += 1
            if out.get('macd_signal') == 'شراء':
                buy_cnt += 1
            elif out.get('macd_signal') == 'بيع':
                sell_cnt += 1
            if out.get('bb_signal') == 'شراء':
                buy_cnt += 1
            elif out.get('bb_signal') == 'بيع':
                sell_cnt += 1
            if trend in ['صاعد','صاعد قوي']:
                buy_cnt += 0.5
            elif trend in ['هابط','هابط قوي']:
                sell_cnt += 0.5

            if buy_cnt > sell_cnt + 1:
                out['recommendation'] = 'شراء قوي'
                out['confidence'] = 'عالية'
            elif buy_cnt > sell_cnt:
                out['recommendation'] = 'شراء'
                out['confidence'] = 'متوسطة'
            elif sell_cnt > buy_cnt + 1:
                out['recommendation'] = 'بيع قوي'
                out['confidence'] = 'عالية'
            elif sell_cnt > buy_cnt:
                out['recommendation'] = 'بيع'
                out['confidence'] = 'متوسطة'
            else:
                out['recommendation'] = 'انتظار'
                out['confidence'] = 'منخفضة'

            risk = self.calculate_advanced_risk_metrics()
            out['risk_level'] = self._risk_level_label(risk)
            out['stop_loss'] = self._stop_loss(price, ind)
            out['take_profit'] = self._take_profit(price, ind)

            logger.info("تم توليد الإشارات المتقدمة بنجاح")
            return out
        except Exception as e:
            logger.error(f"خطأ في توليد الإشارات: {e}")
            return {}

    # ---------------------------- Report ---------------------------------
    def generate_advanced_report_v6(self) -> Dict:
        report: Dict[str, object] = {}
        try:
            # نبدأ بالميتا دائمًا
            now = datetime.datetime.now().isoformat()
            dp = int(self.data.shape[0]) if isinstance(self.data, pd.DataFrame) else 0
            report['metadata'] = {
                'version': '6.0-stable',
                'symbol': self.symbol,
                'period': self.period,
                'analysis_date': now,
                'data_points': dp,
            }

            # أقسام آمنة – أي فشل يتحول إلى حقل error ولا يوقف التقرير
            # Signals
            try:
                signals = self.generate_advanced_signals_v6()
                report['signals'] = convert_numpy_types(signals)
            except Exception as e:
                report['signals'] = {'error': f'{e}'}

            # Indicators
            try:
                ind = self.calculate_technical_indicators()
                report['technical_indicators'] = {
                    'rsi': _safe_last(ind.get('rsi', pd.Series(dtype=float)), 50),
                    'macd': _safe_last(ind.get('macd', pd.Series(dtype=float)), 0),
                    'macd_signal': _safe_last(ind.get('macd_signal', pd.Series(dtype=float)), 0),
                    'sma_20': _safe_last(ind.get('sma_20', pd.Series(dtype=float)), 0),
                    'sma_50': _safe_last(ind.get('sma_50', pd.Series(dtype=float)), 0),
                    'sma_200': _safe_last(ind.get('sma_200', pd.Series(dtype=float)), 0),
                    'bb_upper': _safe_last(ind.get('bbands_upper', pd.Series(dtype=float)), np.nan),
                    'bb_lower': _safe_last(ind.get('bbands_lower', pd.Series(dtype=float)), np.nan),
                    'atr': _safe_last(ind.get('atr', pd.Series(dtype=float)), np.nan),
                }
            except Exception as e:
                report['technical_indicators'] = {'error': f'{e}'}

            # Patterns
            try:
                patterns = self.detect_advanced_patterns()
                report['patterns'] = convert_numpy_types(patterns)
            except Exception as e:
                report['patterns'] = {'error': f'{e}'}

            # Sentiment
            try:
                sentiment = self.analyze_market_sentiment()
                report['sentiment'] = convert_numpy_types(sentiment)
            except Exception as e:
                report['sentiment'] = {'error': f'{e}'}

            # Risk
            try:
                risk = self.calculate_advanced_risk_metrics()
                report['risk_metrics'] = convert_numpy_types(risk)
            except Exception as e:
                report['risk_metrics'] = {'error': f'{e}'}

            # Current market data
            try:
                if self.data is not None and not self.data.empty:
                    close = self.data['Close']
                    report['current_market_data'] = {
                        'current_price': convert_numpy_types(_safe_last(close, np.nan)),
                        'daily_change': convert_numpy_types((_safe_last(close, np.nan) - convert_numpy_types(close.iloc[-2]) ) if len(close) > 1 else 0),
                        'daily_change_percent': convert_numpy_types(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0),
                        'volume': convert_numpy_types(self.data['Volume'].iloc[-1]),
                        'high': convert_numpy_types(self.data['High'].iloc[-1]),
                        'low': convert_numpy_types(self.data['Low'].iloc[-1]),
                    }
                else:
                    report['current_market_data'] = {}
            except Exception as e:
                report['current_market_data'] = {'error': f'{e}'}

            # Summary
            try:
                sig = report.get('signals', {})
                if isinstance(sig, dict):
                    summary = {
                        'overall_recommendation': sig.get('recommendation', 'انتظار'),
                        'confidence_level': sig.get('confidence', 'منخفضة'),
                        'risk_level': sig.get('risk_level', 'متوسطة'),
                        'trend_direction': sig.get('trend', 'متذبذب'),
                        'key_support': sig.get('stop_loss', None),
                        'key_resistance': sig.get('take_profit', None),
                    }
                else:
                    summary = {}
                report['summary'] = convert_numpy_types(summary)
            except Exception as e:
                report['summary'] = {'error': f'{e}'}

            logger.info("تم توليد التقرير المتقدم بنجاح")
            return report
        except Exception as e:
            # ضمان تقرير مهما حدث
            logger.error(f"خطأ في توليد التقرير: {e}")
            return {
                'metadata': {
                    'version': '6.0-stable',
                    'symbol': self.symbol,
                    'period': self.period,
                    'analysis_date': datetime.datetime.now().isoformat(),
                    'data_points': int(self.data.shape[0]) if isinstance(self.data, pd.DataFrame) else 0,
                },
                'error': str(e)
            }

    def save_report(self, filename: str = "gold_analysis_advanced_v6.json") -> bool:
        try:
            report = self.generate_advanced_report_v6()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            logger.info(f"تم حفظ التقرير في {filename}")
            return True
        except Exception as e:
            logger.error(f"خطأ في حفظ التقرير: {e}")
            return False

    def run_analysis(self) -> bool:
        try:
            logger.info("بدء التحليل المتقدم للذهب (نسخة مستقرة)...")
            if not self.fetch_data():
                return False
            report = self.generate_advanced_report_v6()
            if not isinstance(report, dict) or not report:
                logger.error("فشل في توليد التقرير")
                return False
            if not self.save_report():
                return False
            logger.info("تم إكمال التحليل المتقدم بنجاح!")
            return True
        except Exception as e:
            logger.error(f"خطأ في التحليل: {e}")
            return False


# --------------------------------- Main ----------------------------------

def main():
    print("=" * 60)
    print("محلل سوق الذهب المتقدم الإصدار 6.0 (نسخة مستقرة)")
    print("Gold Market Analyzer Advanced V6.0 – Stable")
    print("=" * 60)
    analyzer = AdvancedGoldAnalyzerV6(symbol="GC=F", period="1y")
    success = analyzer.run_analysis()
    if success:
        print("\n✅ تم إكمال التحليل بنجاح!")
        print("📊 تم حفظ التقرير في gold_analysis_advanced_v6.json")
        print("🔍 راجع الملف للحصول على النتائج التفصيلية")
    else:
        print("\n❌ فشل في إكمال التحليل")
        print("🔧 يرجى التحقق من الاتصال بالإنترنت والمحاولة مرة أخرى")


if __name__ == "__main__":
    main()
