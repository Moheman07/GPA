#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Analyzer Enhancements
محسنات محلل الذهب

مميزات إضافية:
- مستويات فيبوناتشي
- كشف الدعم والمقاومة
- تحليل الحجم
- تحليل البنية السوقية
- كشف التباعدات
- تحليل الارتباطات
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using alternative indicators")
import warnings
import json
import datetime
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.cluster import KMeans

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تجاهل التحذيرات
warnings.filterwarnings('ignore')

class GoldAnalyzerEnhancements:
    """محسنات محلل الذهب"""
    
    def __init__(self, symbol: str = "GC=F", period: str = "1y"):
        """
        تهيئة المحسنات
        
        Args:
            symbol: رمز الذهب
            period: الفترة الزمنية
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.enhancements = {}
        
    def fetch_data(self) -> bool:
        """جلب بيانات الذهب"""
        try:
            logger.info(f"جاري جلب بيانات {self.symbol} للمحسنات...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                logger.error("فشل في جلب البيانات")
                return False
                
            logger.info(f"تم جلب {len(self.data)} نقطة بيانات للمحسنات")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات: {e}")
            return False
    
    def calculate_fibonacci_levels(self) -> Dict:
        """حساب مستويات فيبوناتشي"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # العثور على أعلى وأدنى نقطة
            high = self.data['High'].max()
            low = self.data['Low'].min()
            diff = high - low
            
            # مستويات فيبوناتشي
            fib_levels = {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.5': low + 0.5 * diff,
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high
            }
            
            # مستويات فيبوناتشي الموسعة
            fib_extensions = {
                '1.272': high + 0.272 * diff,
                '1.618': high + 0.618 * diff,
                '2.0': high + diff,
                '2.618': high + 1.618 * diff
            }
            
            current_price = self.data['Close'].iloc[-1]
            
            # تحديد المستويات القريبة
            nearest_support = None
            nearest_resistance = None
            
            for level, price in fib_levels.items():
                if price < current_price:
                    if nearest_support is None or price > fib_levels[nearest_support]:
                        nearest_support = level
                else:
                    if nearest_resistance is None or price < fib_levels[nearest_resistance]:
                        nearest_resistance = level
            
            result = {
                'levels': fib_levels,
                'extensions': fib_extensions,
                'current_price': current_price,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_price': fib_levels[nearest_support] if nearest_support else None,
                'resistance_price': fib_levels[nearest_resistance] if nearest_resistance else None
            }
            
            logger.info("تم حساب مستويات فيبوناتشي بنجاح")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في حساب مستويات فيبوناتشي: {e}")
            return {}
    
    def detect_support_resistance(self) -> Dict:
        """كشف مستويات الدعم والمقاومة"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # استخدام Pivot Points
            high = self.data['High'].iloc[-1]
            low = self.data['Low'].iloc[-1]
            close = self.data['Close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            # كشف مستويات الدعم والمقاومة التاريخية
            support_levels = []
            resistance_levels = []
            
            # استخدام النقاط المحورية
            for i in range(20, len(self.data) - 20):
                # دعم محتمل
                if (self.data['Low'].iloc[i] < self.data['Low'].iloc[i-1] and 
                    self.data['Low'].iloc[i] < self.data['Low'].iloc[i+1] and
                    self.data['Low'].iloc[i] < self.data['Low'].iloc[i-5:i].min() and
                    self.data['Low'].iloc[i] < self.data['Low'].iloc[i+1:i+6].min()):
                    support_levels.append(self.data['Low'].iloc[i])
                
                # مقاومة محتملة
                if (self.data['High'].iloc[i] > self.data['High'].iloc[i-1] and 
                    self.data['High'].iloc[i] > self.data['High'].iloc[i+1] and
                    self.data['High'].iloc[i] > self.data['High'].iloc[i-5:i].max() and
                    self.data['High'].iloc[i] > self.data['High'].iloc[i+1:i+6].max()):
                    resistance_levels.append(self.data['High'].iloc[i])
            
            # تجميع المستويات القريبة
            def cluster_levels(levels, tolerance=0.01):
                if not levels:
                    return []
                
                clusters = []
                levels = sorted(levels)
                
                current_cluster = [levels[0]]
                
                for level in levels[1:]:
                    if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                        current_cluster.append(level)
                    else:
                        clusters.append(np.mean(current_cluster))
                        current_cluster = [level]
                
                if current_cluster:
                    clusters.append(np.mean(current_cluster))
                
                return clusters
            
            clustered_support = cluster_levels(support_levels)
            clustered_resistance = cluster_levels(resistance_levels)
            
            current_price = self.data['Close'].iloc[-1]
            
            # تحديد أقرب مستويات الدعم والمقاومة
            nearest_support = None
            nearest_resistance = None
            
            for level in clustered_support:
                if level < current_price:
                    if nearest_support is None or level > nearest_support:
                        nearest_support = level
            
            for level in clustered_resistance:
                if level > current_price:
                    if nearest_resistance is None or level < nearest_resistance:
                        nearest_resistance = level
            
            result = {
                'pivot_points': {
                    'pivot': pivot,
                    'r1': r1, 'r2': r2, 'r3': r3,
                    's1': s1, 's2': s2, 's3': s3
                },
                'historical_support': clustered_support,
                'historical_resistance': clustered_resistance,
                'current_price': current_price,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
            logger.info("تم كشف مستويات الدعم والمقاومة بنجاح")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في كشف الدعم والمقاومة: {e}")
            return {}
    
    def analyze_volume_profile(self) -> Dict:
        """تحليل ملف الحجم"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # تحليل الحجم
            volume_data = self.data['Volume']
            price_data = self.data['Close']
            
            # متوسط الحجم
            avg_volume = volume_data.mean()
            current_volume = volume_data.iloc[-1]
            
            # نسبة الحجم الحالي
            volume_ratio = current_volume / avg_volume
            
            # تحليل توزيع الحجم حسب السعر
            price_bins = pd.cut(price_data, bins=20)
            volume_profile = price_data.groupby(price_bins).agg({
                'Volume': 'sum'
            }).sort_values('Volume', ascending=False)
            
            # تحديد مستويات الحجم العالي
            high_volume_levels = volume_profile.head(5)
            
            # تحليل الحجم المرافق للحركة السعرية
            price_change = price_data.pct_change()
            volume_change = volume_data.pct_change()
            
            # الارتباط بين التغير السعري والحجم
            correlation = price_change.corr(volume_change)
            
            # تحليل الحجم في الاتجاهات
            uptrend_volume = volume_data[price_change > 0].mean()
            downtrend_volume = volume_data[price_change < 0].mean()
            
            result = {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_profile': high_volume_levels.to_dict(),
                'price_volume_correlation': correlation,
                'uptrend_volume': uptrend_volume,
                'downtrend_volume': downtrend_volume,
                'volume_trend': 'مرتفع' if volume_ratio > 1.5 else 'عادي' if volume_ratio > 0.8 else 'منخفض'
            }
            
            logger.info("تم تحليل ملف الحجم بنجاح")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في تحليل ملف الحجم: {e}")
            return {}
    
    def analyze_market_structure(self) -> Dict:
        """تحليل البنية السوقية"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # تحديد النقاط المحورية
            highs = self.data['High']
            lows = self.data['Low']
            
            # كشف القمم والقيعان
            peaks = []
            troughs = []
            
            for i in range(5, len(self.data) - 5):
                # قمة
                if (highs.iloc[i] == highs.iloc[i-5:i+6].max() and
                    highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]):
                    peaks.append((i, highs.iloc[i]))
                
                # قاع
                if (lows.iloc[i] == lows.iloc[i-5:i+6].min() and
                    lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]):
                    troughs.append((i, lows.iloc[i]))
            
            # تحليل الاتجاه
            if len(peaks) >= 2 and len(troughs) >= 2:
                # اتجاه القمم
                peak_trend = 'صاعد' if peaks[-1][1] > peaks[-2][1] else 'هابط'
                
                # اتجاه القيعان
                trough_trend = 'صاعد' if troughs[-1][1] > troughs[-2][1] else 'هابط'
                
                # تحديد البنية السوقية
                if peak_trend == 'صاعد' and trough_trend == 'صاعد':
                    structure = 'اتجاه صاعد'
                elif peak_trend == 'هابط' and trough_trend == 'هابط':
                    structure = 'اتجاه هابط'
                elif peak_trend == 'صاعد' and trough_trend == 'هابط':
                    structure = 'مثلث صاعد'
                elif peak_trend == 'هابط' and trough_trend == 'صاعد':
                    structure = 'مثلث هابط'
                else:
                    structure = 'متذبذب'
            else:
                structure = 'غير محدد'
                peak_trend = 'غير محدد'
                trough_trend = 'غير محدد'
            
            # تحليل القوة النسبية
            current_price = self.data['Close'].iloc[-1]
            sma_20 = self.data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = self.data['Close'].rolling(window=50).mean().iloc[-1]
            
            relative_strength = (current_price / sma_20) * (sma_20 / sma_50)
            
            result = {
                'market_structure': structure,
                'peak_trend': peak_trend,
                'trough_trend': trough_trend,
                'peaks_count': len(peaks),
                'troughs_count': len(troughs),
                'relative_strength': relative_strength,
                'structure_confidence': 'عالية' if len(peaks) >= 3 and len(troughs) >= 3 else 'متوسطة' if len(peaks) >= 2 and len(troughs) >= 2 else 'منخفضة'
            }
            
            logger.info("تم تحليل البنية السوقية بنجاح")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في تحليل البنية السوقية: {e}")
            return {}
    
    def detect_divergences(self) -> Dict:
        """كشف التباعدات"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # حساب RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(self.data['Close'], timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(self.data['Close'])
            else:
                # بدائل بسيطة
                rsi = self._calculate_rsi(self.data['Close'], 14)
                macd = self._calculate_macd(self.data['Close'])
                macd_signal = macd.ewm(span=9).mean()
            
            # كشف التباعدات
            divergences = []
            
            # التباعد السعري مع RSI
            for i in range(20, len(self.data) - 20):
                # تباعد إيجابي (سعر هابط، RSI صاعد)
                if (self.data['Close'].iloc[i] < self.data['Close'].iloc[i-10] and
                    rsi.iloc[i] > rsi.iloc[i-10] and
                    self.data['Close'].iloc[i] < self.data['Close'].iloc[i-20:i].min() and
                    rsi.iloc[i] > rsi.iloc[i-20:i].min()):
                    divergences.append({
                        'type': 'إيجابي',
                        'indicator': 'RSI',
                        'date': self.data.index[i],
                        'price': self.data['Close'].iloc[i],
                        'indicator_value': rsi.iloc[i]
                    })
                
                # تباعد سلبي (سعر صاعد، RSI هابط)
                elif (self.data['Close'].iloc[i] > self.data['Close'].iloc[i-10] and
                      rsi.iloc[i] < rsi.iloc[i-10] and
                      self.data['Close'].iloc[i] > self.data['Close'].iloc[i-20:i].max() and
                      rsi.iloc[i] < rsi.iloc[i-20:i].max()):
                    divergences.append({
                        'type': 'سلبي',
                        'indicator': 'RSI',
                        'date': self.data.index[i],
                        'price': self.data['Close'].iloc[i],
                        'indicator_value': rsi.iloc[i]
                    })
            
            # التباعد مع MACD
            for i in range(20, len(self.data) - 20):
                # تباعد إيجابي
                if (self.data['Close'].iloc[i] < self.data['Close'].iloc[i-10] and
                    macd.iloc[i] > macd.iloc[i-10] and
                    self.data['Close'].iloc[i] < self.data['Close'].iloc[i-20:i].min() and
                    macd.iloc[i] > macd.iloc[i-20:i].min()):
                    divergences.append({
                        'type': 'إيجابي',
                        'indicator': 'MACD',
                        'date': self.data.index[i],
                        'price': self.data['Close'].iloc[i],
                        'indicator_value': macd.iloc[i]
                    })
                
                # تباعد سلبي
                elif (self.data['Close'].iloc[i] > self.data['Close'].iloc[i-10] and
                      macd.iloc[i] < macd.iloc[i-10] and
                      self.data['Close'].iloc[i] > self.data['Close'].iloc[i-20:i].max() and
                      macd.iloc[i] < macd.iloc[i-20:i].max()):
                    divergences.append({
                        'type': 'سلبي',
                        'indicator': 'MACD',
                        'date': self.data.index[i],
                        'price': self.data['Close'].iloc[i],
                        'indicator_value': macd.iloc[i]
                    })
            
            # تصفية التباعدات الحديثة
            recent_divergences = [d for d in divergences if d['date'] >= self.data.index[-30]]
            
            result = {
                'total_divergences': len(divergences),
                'recent_divergences': len(recent_divergences),
                'positive_divergences': len([d for d in recent_divergences if d['type'] == 'إيجابي']),
                'negative_divergences': len([d for d in recent_divergences if d['type'] == 'سلبي']),
                'rsi_divergences': len([d for d in recent_divergences if d['indicator'] == 'RSI']),
                'macd_divergences': len([d for d in recent_divergences if d['indicator'] == 'MACD']),
                'latest_divergences': recent_divergences[-5:] if recent_divergences else []
            }
            
            logger.info("تم كشف التباعدات بنجاح")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في كشف التباعدات: {e}")
            return {}
    
    def analyze_correlations(self) -> Dict:
        """تحليل الارتباطات"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # جلب بيانات الأصول المرتبطة
            assets = {
                'USD': 'DX-Y.NYB',  # مؤشر الدولار
                'SPY': 'SPY',       # S&P 500
                'TLT': 'TLT',        # السندات طويلة الأجل
                'VIX': '^VIX',       # مؤشر الخوف
                'OIL': 'USO'         # النفط
            }
            
            correlations = {}
            
            for asset_name, asset_symbol in assets.items():
                try:
                    asset_data = yf.Ticker(asset_symbol).history(period=self.period)
                    if not asset_data.empty:
                        # محاذاة البيانات
                        aligned_data = self.data['Close'].align(asset_data['Close'], join='inner')
                        if len(aligned_data[0]) > 30:
                            correlation = aligned_data[0].corr(aligned_data[1])
                            correlations[asset_name] = correlation
                except:
                    continue
            
            # تحليل الارتباط مع الوقت
            time_correlation = self.data['Close'].corr(pd.Series(range(len(self.data))))
            
            # تحليل الارتباط مع الحجم
            volume_correlation = self.data['Close'].corr(self.data['Volume'])
            
            # تحليل الارتباط مع التذبذب
            volatility = self.data['Close'].pct_change().rolling(window=20).std()
            volatility_correlation = self.data['Close'].corr(volatility)
            
            result = {
                'asset_correlations': correlations,
                'time_correlation': time_correlation,
                'volume_correlation': volume_correlation,
                'volatility_correlation': volatility_correlation,
                'strongest_correlation': max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None,
                'weakest_correlation': min(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
            }
            
            logger.info("تم تحليل الارتباطات بنجاح")
            return result
            
        except Exception as e:
            logger.error(f"خطأ في تحليل الارتباطات: {e}")
            return {}
    
    def calculate_advanced_metrics(self) -> Dict:
        """حساب المقاييس المتقدمة"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # حساب العوائد
            returns = self.data['Close'].pct_change().dropna()
            
            # مقاييس متقدمة
            metrics = {
                'annualized_return': returns.mean() * 252,
                'annualized_volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'max_drawdown': (self.data['Close'] / self.data['Close'].expanding().max() - 1).min(),
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'calmar_ratio': (returns.mean() * 252) / abs((self.data['Close'] / self.data['Close'].expanding().max() - 1).min()),
                'sortino_ratio': (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() > 0 else 0
            }
            
            # مؤشرات إضافية
            metrics['win_rate'] = len(returns[returns > 0]) / len(returns)
            metrics['avg_win'] = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            metrics['avg_loss'] = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
            metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
            
            logger.info("تم حساب المقاييس المتقدمة بنجاح")
            return metrics
            
        except Exception as e:
            logger.error(f"خطأ في حساب المقاييس المتقدمة: {e}")
            return {}
    
    def generate_enhanced_signals(self) -> Dict:
        """توليد الإشارات المحسنة"""
        if self.data is None or self.data.empty:
            return {}
            
        try:
            # جمع جميع التحليلات
            fib_levels = self.calculate_fibonacci_levels()
            support_resistance = self.detect_support_resistance()
            volume_profile = self.analyze_volume_profile()
            market_structure = self.analyze_market_structure()
            divergences = self.detect_divergences()
            correlations = self.analyze_correlations()
            metrics = self.calculate_advanced_metrics()
            
            current_price = self.data['Close'].iloc[-1]
            
            # توليد الإشارات
            signals = {
                'fibonacci_signals': {},
                'support_resistance_signals': {},
                'volume_signals': {},
                'structure_signals': {},
                'divergence_signals': {},
                'correlation_signals': {},
                'risk_signals': {}
            }
            
            # إشارات فيبوناتشي
            if fib_levels:
                if current_price < fib_levels.get('support_price', float('inf')):
                    signals['fibonacci_signals']['support_test'] = 'اختبار دعم فيبوناتشي'
                elif current_price > fib_levels.get('resistance_price', 0):
                    signals['fibonacci_signals']['resistance_test'] = 'اختبار مقاومة فيبوناتشي'
            
            # إشارات الدعم والمقاومة
            if support_resistance:
                nearest_support = support_resistance.get('nearest_support')
                nearest_resistance = support_resistance.get('nearest_resistance')
                
                if nearest_support and current_price <= nearest_support * 1.01:
                    signals['support_resistance_signals']['support_bounce'] = 'ارتداد من مستوى الدعم'
                elif nearest_resistance and current_price >= nearest_resistance * 0.99:
                    signals['support_resistance_signals']['resistance_rejection'] = 'رفض من مستوى المقاومة'
            
            # إشارات الحجم
            if volume_profile:
                volume_ratio = volume_profile.get('volume_ratio', 1)
                if volume_ratio > 1.5:
                    signals['volume_signals']['high_volume'] = 'حجم مرتفع - تأكيد الحركة'
                elif volume_ratio < 0.5:
                    signals['volume_signals']['low_volume'] = 'حجم منخفض - شك في الحركة'
            
            # إشارات البنية السوقية
            if market_structure:
                structure = market_structure.get('market_structure', '')
                if 'صاعد' in structure:
                    signals['structure_signals']['uptrend'] = 'بنية سوق صاعدة'
                elif 'هابط' in structure:
                    signals['structure_signals']['downtrend'] = 'بنية سوق هابطة'
            
            # إشارات التباعدات
            if divergences:
                recent_pos = divergences.get('positive_divergences', 0)
                recent_neg = divergences.get('negative_divergences', 0)
                
                if recent_pos > recent_neg:
                    signals['divergence_signals']['positive_divergence'] = 'تباعد إيجابي - إشارة شراء'
                elif recent_neg > recent_pos:
                    signals['divergence_signals']['negative_divergence'] = 'تباعد سلبي - إشارة بيع'
            
            # إشارات المخاطر
            if metrics:
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd = abs(metrics.get('max_drawdown', 0))
                
                if sharpe > 1:
                    signals['risk_signals']['good_risk_reward'] = 'مخاطرة/مكافأة جيدة'
                elif sharpe < 0:
                    signals['risk_signals']['poor_risk_reward'] = 'مخاطرة/مكافأة ضعيفة'
                
                if max_dd > 0.2:
                    signals['risk_signals']['high_drawdown'] = 'سحب مرتفع - توخي الحذر'
            
            logger.info("تم توليد الإشارات المحسنة بنجاح")
            return signals
            
        except Exception as e:
            logger.error(f"خطأ في توليد الإشارات المحسنة: {e}")
            return {}
    
    def create_enhanced_report(self) -> Dict:
        """إنشاء التقرير المحسن"""
        try:
            # جمع جميع التحليلات
            fib_levels = self.calculate_fibonacci_levels()
            support_resistance = self.detect_support_resistance()
            volume_profile = self.analyze_volume_profile()
            market_structure = self.analyze_market_structure()
            divergences = self.detect_divergences()
            correlations = self.analyze_correlations()
            metrics = self.calculate_advanced_metrics()
            signals = self.generate_enhanced_signals()
            
            # إنشاء التقرير
            report = {
                'metadata': {
                    'version': 'enhanced',
                    'symbol': self.symbol,
                    'period': self.period,
                    'analysis_date': datetime.datetime.now().isoformat(),
                    'data_points': len(self.data) if self.data is not None else 0
                },
                'fibonacci_analysis': fib_levels,
                'support_resistance_analysis': support_resistance,
                'volume_profile_analysis': volume_profile,
                'market_structure_analysis': market_structure,
                'divergence_analysis': divergences,
                'correlation_analysis': correlations,
                'advanced_metrics': metrics,
                'enhanced_signals': signals,
                'summary': {
                    'current_price': self.data['Close'].iloc[-1] if self.data is not None else 0,
                    'key_levels': {
                        'nearest_support': fib_levels.get('support_price') or support_resistance.get('nearest_support'),
                        'nearest_resistance': fib_levels.get('resistance_price') or support_resistance.get('nearest_resistance')
                    },
                    'market_structure': market_structure.get('market_structure', 'غير محدد'),
                    'volume_trend': volume_profile.get('volume_trend', 'عادي'),
                    'risk_level': 'عالية' if metrics.get('max_drawdown', 0) < -0.2 else 'متوسطة' if metrics.get('max_drawdown', 0) < -0.1 else 'منخفضة'
                }
            }
            
            logger.info("تم إنشاء التقرير المحسن بنجاح")
            return report
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء التقرير المحسن: {e}")
            return {}
    
    def save_enhanced_report(self, filename: str = "gold_analysis_enhancements.json") -> bool:
        """حفظ التقرير المحسن"""
        try:
            report = self.create_enhanced_report()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"تم حفظ التقرير المحسن في {filename}")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في حفظ التقرير المحسن: {e}")
            return False
    
    def run_enhanced_analysis(self) -> bool:
        """تشغيل التحليل المحسن الكامل"""
        try:
            logger.info("بدء التحليل المحسن للذهب...")
            
            # جلب البيانات
            if not self.fetch_data():
                return False
            
            # تشغيل التحليل المحسن
            report = self.create_enhanced_report()
            
            if not report:
                logger.error("فشل في إنشاء التقرير المحسن")
                return False
            
            # حفظ التقرير
            if not self.save_enhanced_report():
                return False
            
            logger.info("تم إكمال التحليل المحسن بنجاح!")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في التحليل المحسن: {e}")
            return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """حساب RSI بدون talib"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """حساب MACD بدون talib"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series([0] * len(prices), index=prices.index)

def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("محسنات محلل الذهب")
    print("Gold Analyzer Enhancements")
    print("=" * 60)
    
    # إنشاء المحسنات
    enhancements = GoldAnalyzerEnhancements(symbol="GC=F", period="1y")
    
    # تشغيل التحليل المحسن
    success = enhancements.run_enhanced_analysis()
    
    if success:
        print("\n✅ تم إكمال التحليل المحسن بنجاح!")
        print("📊 تم حفظ التقرير في gold_analysis_enhancements.json")
        print("🔍 راجع الملف للحصول على النتائج التفصيلية")
    else:
        print("\n❌ فشل في إكمال التحليل المحسن")
        print("🔧 يرجى التحقق من الاتصال بالإنترنت والمحاولة مرة أخرى")

if __name__ == "__main__":
    main()
