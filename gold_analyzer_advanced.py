#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold Market Analyzer Advanced V6.0
محلل سوق الذهب المتقدم الإصدار 6.0

مميزات النسخة المتقدمة:
- 25+ مؤشر فني متقدم
- كشف الأنماط السعرية والشموع اليابانية
- تحليل المشاعر السوقية
- إدارة المخاطر المتقدمة
- تحليل متعدد الأطر الزمنية
- تقارير تفاعلية شاملة
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

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تجاهل التحذيرات
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """مشفر JSON للتعامل مع أنواع numpy"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """تحويل أنواع numpy إلى أنواع Python القياسية"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):  # للتعامل مع pandas scalar types
        return obj.item()
    return obj

class AdvancedGoldAnalyzerV6:
    """محلل الذهب المتقدم الإصدار 6.0"""
    
    def __init__(self, symbol: str = "GC=F", period: str = "1y", fast_mode: bool = True):
        """
        تهيئة المحلل المتقدم
        
        Args:
            symbol: رمز الذهب (GC=F للعقود الآجلة)
            period: الفترة الزمنية
            fast_mode: وضع التشغيل السريع (يقلل من دقة التحليل لزيادة السرعة)
        """
        self.symbol = symbol
        self.period = period
        self.fast_mode = fast_mode
        self.data = None
        self.analysis_results = {}
        self.risk_metrics = {}
        self.signals = {}
        
    def fetch_data(self) -> bool:
        """جلب بيانات الذهب"""
        try:
            logger.info(f"جاري جلب بيانات {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                logger.error("فشل في جلب البيانات")
                return False
                
            logger.info(f"تم جلب {len(self.data)} نقطة بيانات")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في جلب البيانات: {e}")
            return False
    
    def calculate_technical_indicators(self) -> Dict:
        """حساب المؤشرات الفنية المتقدمة"""
        if self.data is None or self.data.empty:
            return {}
            
        indicators = {}
        
        try:
            if TALIB_AVAILABLE:
                # مؤشرات الاتجاه
                indicators['sma_20'] = talib.SMA(self.data['Close'], timeperiod=20)
                indicators['sma_50'] = talib.SMA(self.data['Close'], timeperiod=50)
                indicators['sma_200'] = talib.SMA(self.data['Close'], timeperiod=200)
                indicators['ema_12'] = talib.EMA(self.data['Close'], timeperiod=12)
                indicators['ema_26'] = talib.EMA(self.data['Close'], timeperiod=26)
            else:
                # استخدام pandas للبدائل
                indicators['sma_20'] = self.data['Close'].rolling(window=20).mean()
                indicators['sma_50'] = self.data['Close'].rolling(window=50).mean()
                indicators['sma_200'] = self.data['Close'].rolling(window=200).mean()
                indicators['ema_12'] = self.data['Close'].ewm(span=12).mean()
                indicators['ema_26'] = self.data['Close'].ewm(span=26).mean()
            
            # مؤشرات الزخم
            if TALIB_AVAILABLE:
                indicators['rsi'] = talib.RSI(self.data['Close'], timeperiod=14)
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(
                    self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
                indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                    self.data['High'], self.data['Low'], self.data['Close'],
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                indicators['williams_r'] = talib.WILLR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
                indicators['cci'] = talib.CCI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
                indicators['adx'] = talib.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
                indicators['trix'] = talib.TRIX(self.data['Close'], timeperiod=30)
                indicators['ultosc'] = talib.ULTOSC(self.data['High'], self.data['Low'], self.data['Close'])
            else:
                # بدائل بسيطة
                indicators['rsi'] = self._calculate_rsi(self.data['Close'], 14)
                indicators['macd'] = self._calculate_macd(self.data['Close'])
                indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
                indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            
            # مؤشرات التذبذب
            indicators['bbands_upper'], indicators['bbands_middle'], indicators['bbands_lower'] = talib.BBANDS(
                self.data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            indicators['atr'] = talib.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
            indicators['sar'] = talib.SAR(self.data['High'], self.data['Low'])
            
            # مؤشرات الاتجاه
            indicators['dmi_plus'] = talib.PLUS_DI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
            indicators['dmi_minus'] = talib.MINUS_DI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
            
            # مؤشرات الحجم
            indicators['obv'] = talib.OBV(self.data['Close'], self.data['Volume'])
            indicators['ad'] = talib.AD(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])
            indicators['adosc'] = talib.ADOSC(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])
            
            # مؤشرات إضافية
            indicators['mfi'] = talib.MFI(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], timeperiod=14)
            indicators['mom'] = talib.MOM(self.data['Close'], timeperiod=10)
            indicators['roc'] = talib.ROC(self.data['Close'], timeperiod=10)
            indicators['slowk'], indicators['slowd'] = talib.STOCHF(self.data['High'], self.data['Low'], self.data['Close'])
            
            logger.info("تم حساب جميع المؤشرات الفنية بنجاح")
            return indicators
            
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات الفنية: {e}")
            return {}
    
    def detect_advanced_patterns(self) -> Dict:
        """كشف الأنماط المتقدمة"""
        if self.data is None or self.data.empty:
            return {}
            
        patterns = {}
        
        try:
            # أنماط الشموع اليابانية
            patterns['doji'] = talib.CDLDOJI(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
            patterns['hammer'] = talib.CDLHAMMER(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
            patterns['engulfing'] = talib.CDLENGULFING(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
            patterns['morning_star'] = talib.CDLMORNINGSTAR(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
            patterns['evening_star'] = talib.CDLEVENINGSTAR(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
            
            # أنماط السعر
            patterns['double_top'] = self._detect_double_top()
            patterns['double_bottom'] = self._detect_double_bottom()
            patterns['head_shoulders'] = self._detect_head_shoulders()
            patterns['triangle'] = self._detect_triangle()
            
            logger.info("تم كشف الأنماط المتقدمة بنجاح")
            return patterns
            
        except Exception as e:
            logger.error(f"خطأ في كشف الأنماط: {e}")
            return {}
    
    def _detect_double_top(self) -> List[int]:
        """كشف القمة المزدوجة"""
        try:
            highs = self.data['High'].rolling(window=5, center=True).max()
            double_tops = []
            
            for i in range(20, len(self.data) - 20):
                if (highs.iloc[i] == self.data['High'].iloc[i] and
                    abs(self.data['High'].iloc[i] - self.data['High'].iloc[i-20:i].max()) < 0.01):
                    double_tops.append(i)
                    
            return double_tops
        except:
            return []
    
    def _detect_double_bottom(self) -> List[int]:
        """كشف القاع المزدوج"""
        try:
            lows = self.data['Low'].rolling(window=5, center=True).min()
            double_bottoms = []
            
            for i in range(20, len(self.data) - 20):
                if (lows.iloc[i] == self.data['Low'].iloc[i] and
                    abs(self.data['Low'].iloc[i] - self.data['Low'].iloc[i-20:i].min()) < 0.01):
                    double_bottoms.append(i)
                    
            return double_bottoms
        except:
            return []
    
    def _detect_head_shoulders(self) -> List[int]:
        """كشف نمط الرأس والكتفين"""
        try:
            # تبسيط الكشف - يمكن تطويره أكثر
            return []
        except:
            return []
    
    def _detect_triangle(self) -> List[int]:
        """كشف الأنماط المثلثية"""
        try:
            # تبسيط الكشف - يمكن تطويره أكثر
            return []
        except:
            return []
    
    def analyze_market_sentiment(self) -> Dict:
        """تحليل المشاعر السوقية"""
        if self.data is None or self.data.empty:
            return {}
            
        sentiment = {}
        
        try:
            # تحليل المشاعر بناءً على المؤشرات الفنية
            indicators = self.calculate_technical_indicators()
            
            # RSI
            current_rsi = indicators.get('rsi', pd.Series()).iloc[-1] if not indicators.get('rsi', pd.Series()).empty else 50
            if current_rsi > 70:
                sentiment['rsi_sentiment'] = 'مفرط في البيع'
            elif current_rsi < 30:
                sentiment['rsi_sentiment'] = 'مفرط في الشراء'
            else:
                sentiment['rsi_sentiment'] = 'محايد'
            
            # MACD
            macd = indicators.get('macd', pd.Series())
            macd_signal = indicators.get('macd_signal', pd.Series())
            if not macd.empty and not macd_signal.empty:
                if macd.iloc[-1] > macd_signal.iloc[-1]:
                    sentiment['macd_sentiment'] = 'إيجابي'
                else:
                    sentiment['macd_sentiment'] = 'سلبي'
            
            # Bollinger Bands
            bb_upper = indicators.get('bbands_upper', pd.Series())
            bb_lower = indicators.get('bbands_lower', pd.Series())
            current_price = self.data['Close'].iloc[-1]
            
            if not bb_upper.empty and not bb_lower.empty:
                if current_price > bb_upper.iloc[-1]:
                    sentiment['bb_sentiment'] = 'مفرط في الشراء'
                elif current_price < bb_lower.iloc[-1]:
                    sentiment['bb_sentiment'] = 'مفرط في البيع'
                else:
                    sentiment['bb_sentiment'] = 'عادي'
            
            # ADX (قوة الاتجاه)
            adx = indicators.get('adx', pd.Series())
            if not adx.empty:
                if adx.iloc[-1] > 25:
                    sentiment['trend_strength'] = 'قوي'
                else:
                    sentiment['trend_strength'] = 'ضعيف'
            
            # الحجم
            volume_avg = self.data['Volume'].rolling(window=20).mean()
            current_volume = self.data['Volume'].iloc[-1]
            if current_volume > volume_avg.iloc[-1] * 1.5:
                sentiment['volume_sentiment'] = 'مرتفع'
            elif current_volume < volume_avg.iloc[-1] * 0.5:
                sentiment['volume_sentiment'] = 'منخفض'
            else:
                sentiment['volume_sentiment'] = 'عادي'
            
            logger.info("تم تحليل المشاعر السوقية بنجاح")
            return sentiment
            
        except Exception as e:
            logger.error(f"خطأ في تحليل المشاعر: {e}")
            return {}
    
    def calculate_advanced_risk_metrics(self) -> Dict:
        """حساب مقاييس المخاطر المتقدمة"""
        if self.data is None or self.data.empty:
            return {}
            
        risk_metrics = {}
        
        try:
            # حساب العوائد
            returns = self.data['Close'].pct_change().dropna()
            
            # التذبذب
            risk_metrics['volatility'] = returns.std() * np.sqrt(252)  # سنوي
            
            # Value at Risk (VaR)
            risk_metrics['var_95'] = np.percentile(returns, 5)
            risk_metrics['var_99'] = np.percentile(returns, 1)
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            risk_metrics['max_drawdown'] = drawdown.min()
            
            # Sharpe Ratio
            risk_free_rate = 0.02  # 2% سنوياً
            excess_returns = returns - risk_free_rate/252
            risk_metrics['sharpe_ratio'] = excess_returns.mean() / returns.std() * np.sqrt(252)
            
            # Beta (مقارنة مع S&P 500)
            try:
                sp500 = yf.Ticker("^GSPC").history(period=self.period)
                sp500_returns = sp500['Close'].pct_change().dropna()
                
                # محاذاة البيانات
                aligned_returns = returns.align(sp500_returns, join='inner')[0]
                aligned_sp500_returns = returns.align(sp500_returns, join='inner')[1]
                
                if len(aligned_returns) > 30:
                    covariance = np.cov(aligned_returns, aligned_sp500_returns)[0, 1]
                    sp500_variance = np.var(aligned_sp500_returns)
                    risk_metrics['beta'] = covariance / sp500_variance
                else:
                    risk_metrics['beta'] = 1.0
            except:
                risk_metrics['beta'] = 1.0
            
            # مؤشرات إضافية
            risk_metrics['skewness'] = returns.skew()
            risk_metrics['kurtosis'] = returns.kurtosis()
            risk_metrics['var_ratio'] = risk_metrics['var_95'] / risk_metrics['var_99']
            
            logger.info("تم حساب مقاييس المخاطر المتقدمة بنجاح")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"خطأ في حساب مقاييس المخاطر: {e}")
            return {}
    
    def generate_advanced_signals_v6(self) -> Dict:
        """توليد الإشارات المتقدمة V6.0"""
        if self.data is None or self.data.empty:
            return {}
            
        signals = {}
        
        try:
            indicators = self.calculate_technical_indicators()
            patterns = self.detect_advanced_patterns()
            sentiment = self.analyze_market_sentiment()
            
            current_price = self.data['Close'].iloc[-1]
            signals['current_price'] = current_price
            signals['timestamp'] = datetime.datetime.now().isoformat()
            
            # تحليل الاتجاه
            sma_20 = indicators.get('sma_20', pd.Series())
            sma_50 = indicators.get('sma_50', pd.Series())
            sma_200 = indicators.get('sma_200', pd.Series())
            
            if not sma_20.empty and not sma_50.empty and not sma_200.empty:
                if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
                    signals['trend'] = 'صاعد قوي'
                elif current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                    signals['trend'] = 'صاعد'
                elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]:
                    signals['trend'] = 'هابط قوي'
                elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                    signals['trend'] = 'هابط'
                else:
                    signals['trend'] = 'متذبذب'
            
            # تحليل RSI
            rsi = indicators.get('rsi', pd.Series())
            if not rsi.empty:
                current_rsi = rsi.iloc[-1]
                if current_rsi < 30:
                    signals['rsi_signal'] = 'شراء قوي'
                elif current_rsi < 40:
                    signals['rsi_signal'] = 'شراء'
                elif current_rsi > 70:
                    signals['rsi_signal'] = 'بيع قوي'
                elif current_rsi > 60:
                    signals['rsi_signal'] = 'بيع'
                else:
                    signals['rsi_signal'] = 'محايد'
            
            # تحليل MACD
            macd = indicators.get('macd', pd.Series())
            macd_signal = indicators.get('macd_signal', pd.Series())
            if not macd.empty and not macd_signal.empty:
                if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
                    signals['macd_signal'] = 'شراء'
                elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
                    signals['macd_signal'] = 'بيع'
                else:
                    signals['macd_signal'] = 'محايد'
            
            # تحليل Bollinger Bands
            bb_upper = indicators.get('bbands_upper', pd.Series())
            bb_lower = indicators.get('bbands_lower', pd.Series())
            if not bb_upper.empty and not bb_lower.empty:
                if current_price < bb_lower.iloc[-1]:
                    signals['bb_signal'] = 'شراء'
                elif current_price > bb_upper.iloc[-1]:
                    signals['bb_signal'] = 'بيع'
                else:
                    signals['bb_signal'] = 'محايد'
            
            # تحليل الأنماط
            pattern_signals = []
            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, list) and len(pattern_data) > 0:
                    if pattern_data[-1] == self.data.shape[0] - 1:  # نمط حديث
                        pattern_signals.append(pattern_name)
                elif isinstance(pattern_data, pd.Series) and not pattern_data.empty:
                    if pattern_data.iloc[-1] != 0:  # نمط موجود
                        pattern_signals.append(pattern_name)
            
            signals['patterns'] = pattern_signals
            
            # التوصية النهائية
            buy_signals = 0
            sell_signals = 0
            
            if signals.get('rsi_signal') in ['شراء', 'شراء قوي']:
                buy_signals += 1
            elif signals.get('rsi_signal') in ['بيع', 'بيع قوي']:
                sell_signals += 1
                
            if signals.get('macd_signal') == 'شراء':
                buy_signals += 1
            elif signals.get('macd_signal') == 'بيع':
                sell_signals += 1
                
            if signals.get('bb_signal') == 'شراء':
                buy_signals += 1
            elif signals.get('bb_signal') == 'بيع':
                sell_signals += 1
            
            # وزن الاتجاه
            if signals.get('trend') in ['صاعد', 'صاعد قوي']:
                buy_signals += 0.5
            elif signals.get('trend') in ['هابط', 'هابط قوي']:
                sell_signals += 0.5
            
            # التوصية النهائية
            if buy_signals > sell_signals + 1:
                signals['recommendation'] = 'شراء قوي'
                signals['confidence'] = 'عالية'
            elif buy_signals > sell_signals:
                signals['recommendation'] = 'شراء'
                signals['confidence'] = 'متوسطة'
            elif sell_signals > buy_signals + 1:
                signals['recommendation'] = 'بيع قوي'
                signals['confidence'] = 'عالية'
            elif sell_signals > buy_signals:
                signals['recommendation'] = 'بيع'
                signals['confidence'] = 'متوسطة'
            else:
                signals['recommendation'] = 'انتظار'
                signals['confidence'] = 'منخفضة'
            
            # إدارة المخاطر
            risk_metrics = self.calculate_advanced_risk_metrics()
            signals['risk_level'] = self._calculate_risk_level(risk_metrics)
            signals['stop_loss'] = self._calculate_stop_loss(current_price, risk_metrics)
            signals['take_profit'] = self._calculate_take_profit(current_price, risk_metrics)
            
            logger.info("تم توليد الإشارات المتقدمة بنجاح")
            return signals
            
        except Exception as e:
            logger.error(f"خطأ في توليد الإشارات: {e}")
            return {}
    
    def _calculate_risk_level(self, risk_metrics: Dict) -> str:
        """حساب مستوى المخاطر"""
        try:
            volatility = risk_metrics.get('volatility', 0.2)
            max_dd = abs(risk_metrics.get('max_drawdown', 0.1))
            
            if volatility > 0.3 or max_dd > 0.2:
                return 'عالية'
            elif volatility > 0.2 or max_dd > 0.15:
                return 'متوسطة'
            else:
                return 'منخفضة'
        except:
            return 'متوسطة'
    
    def _calculate_stop_loss(self, current_price: float, risk_metrics: Dict) -> float:
        """حساب مستوى وقف الخسارة"""
        try:
            atr = self.calculate_technical_indicators().get('atr', pd.Series())
            if not atr.empty:
                return current_price - (atr.iloc[-1] * 2)
            else:
                return current_price * 0.95  # 5% خسارة
        except:
            return current_price * 0.95
    
    def _calculate_take_profit(self, current_price: float, risk_metrics: Dict) -> float:
        """حساب مستوى جني الأرباح"""
        try:
            atr = self.calculate_technical_indicators().get('atr', pd.Series())
            if not atr.empty:
                return current_price + (atr.iloc[-1] * 3)
            else:
                return current_price * 1.08  # 8% ربح
        except:
            return current_price * 1.08
    
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
    
    def generate_advanced_report_v6(self) -> Dict:
        """توليد التقرير المتقدم V6.0"""
        try:
            # جمع جميع البيانات
            signals = self.generate_advanced_signals_v6()
            indicators = self.calculate_technical_indicators()
            patterns = self.detect_advanced_patterns()
            sentiment = self.analyze_market_sentiment()
            risk_metrics = self.calculate_advanced_risk_metrics()
            
            # إنشاء التقرير
            report = {
                'metadata': {
                    'version': '6.0',
                    'symbol': self.symbol,
                    'period': self.period,
                    'analysis_date': datetime.datetime.now().isoformat(),
                    'data_points': self.data.shape[0] if self.data is not None else 0
                },
                'current_market_data': {
                    'current_price': convert_numpy_types(self.data['Close'].iloc[-1]) if self.data is not None else 0,
                    'daily_change': convert_numpy_types(self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]) if self.data is not None and self.data.shape[0] > 1 else 0,
                    'daily_change_percent': convert_numpy_types(((self.data['Close'].iloc[-1] - self.data['Close'].iloc[-2]) / self.data['Close'].iloc[-2] * 100)) if self.data is not None and self.data.shape[0] > 1 else 0,
                    'volume': convert_numpy_types(self.data['Volume'].iloc[-1]) if self.data is not None else 0,
                    'high': convert_numpy_types(self.data['High'].iloc[-1]) if self.data is not None else 0,
                    'low': convert_numpy_types(self.data['Low'].iloc[-1]) if self.data is not None else 0
                },
                'signals': convert_numpy_types(signals),
                'technical_indicators': {
                    'rsi': convert_numpy_types(indicators.get('rsi', pd.Series()).iloc[-1]) if not indicators.get('rsi', pd.Series()).empty else 50,
                    'macd': convert_numpy_types(indicators.get('macd', pd.Series()).iloc[-1]) if not indicators.get('macd', pd.Series()).empty else 0,
                    'macd_signal': convert_numpy_types(indicators.get('macd_signal', pd.Series()).iloc[-1]) if not indicators.get('macd_signal', pd.Series()).empty else 0,
                    'sma_20': convert_numpy_types(indicators.get('sma_20', pd.Series()).iloc[-1]) if not indicators.get('sma_20', pd.Series()).empty else 0,
                    'sma_50': convert_numpy_types(indicators.get('sma_50', pd.Series()).iloc[-1]) if not indicators.get('sma_50', pd.Series()).empty else 0,
                    'sma_200': convert_numpy_types(indicators.get('sma_200', pd.Series()).iloc[-1]) if not indicators.get('sma_200', pd.Series()).empty else 0,
                    'bb_upper': convert_numpy_types(indicators.get('bbands_upper', pd.Series()).iloc[-1]) if not indicators.get('bbands_upper', pd.Series()).empty else 0,
                    'bb_lower': convert_numpy_types(indicators.get('bbands_lower', pd.Series()).iloc[-1]) if not indicators.get('bbands_lower', pd.Series()).empty else 0,
                    'atr': convert_numpy_types(indicators.get('atr', pd.Series()).iloc[-1]) if not indicators.get('atr', pd.Series()).empty else 0
                },
                'patterns': convert_numpy_types(patterns),
                'sentiment': convert_numpy_types(sentiment),
                'risk_metrics': convert_numpy_types(risk_metrics),
                'summary': {
                    'overall_recommendation': signals.get('recommendation', 'انتظار'),
                    'confidence_level': signals.get('confidence', 'منخفضة'),
                    'risk_level': signals.get('risk_level', 'متوسطة'),
                    'trend_direction': signals.get('trend', 'متذبذب'),
                    'key_support': convert_numpy_types(signals.get('stop_loss', 0)),
                    'key_resistance': convert_numpy_types(signals.get('take_profit', 0))
                }
            }
            
            logger.info("تم توليد التقرير المتقدم بنجاح")
            return report
            
        except Exception as e:
            logger.error(f"خطأ في توليد التقرير: {e}")
            return {}
    
    def save_report(self, filename: str = "gold_analysis_advanced_v6.json") -> bool:
        """حفظ التقرير في ملف JSON"""
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
        """تشغيل التحليل الكامل"""
        try:
            logger.info("بدء التحليل المتقدم للذهب...")
            
            # جلب البيانات
            if not self.fetch_data():
                return False
            
            # تشغيل التحليل
            report = self.generate_advanced_report_v6()
            
            if not report:
                logger.error("فشل في توليد التقرير")
                return False
            
            # حفظ التقرير
            if not self.save_report():
                return False
            
            logger.info("تم إكمال التحليل المتقدم بنجاح!")
            return True
            
        except Exception as e:
            logger.error(f"خطأ في التحليل: {e}")
            return False

# فئات إضافية للميزات المتقدمة (قيد التطوير)
class AdvancedMLPredictor:
    """محلل التعلم الآلي المتقدم"""
    pass

class AdvancedMultiTimeframeAnalyzer:
    """محلل متعدد الأطر الزمنية المتقدم"""
    pass

class AdvancedNewsAnalyzer:
    """محلل الأخبار المتقدم"""
    pass

class AdvancedDatabaseManager:
    """مدير قاعدة البيانات المتقدم"""
    pass

class AdvancedBacktester:
    """محلل الاختبار المتقدم"""
    pass

class AdvancedRiskManager:
    """مدير المخاطر المتقدم"""
    pass

class PatternDetector:
    """كاشف الأنماط المتقدم"""
    pass

class SentimentAnalyzer:
    """محلل المشاعر المتقدم"""
    pass

def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("محلل سوق الذهب المتقدم الإصدار 6.0")
    print("Gold Market Analyzer Advanced V6.0")
    print("=" * 60)
    
    # إنشاء المحلل
    analyzer = AdvancedGoldAnalyzerV6(symbol="GC=F", period="1y")
    
    # تشغيل التحليل
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
