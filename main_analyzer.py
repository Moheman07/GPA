#!/usr/bin/env python3
"""
محلل الذهب الاحترافي - الإصدار 5.0
نسخة كاملة ومحسنة للتداول الاحترافي
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sqlite3
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from textblob import TextBlob
import spacy
import backtrader as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import requests

warnings.filterwarnings('ignore')

# تحميل نموذج spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ProfessionalGoldAnalyzerV5:
    """محلل الذهب الاحترافي الإصدار 5.0"""
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'dxy': 'DX-Y.NYB',
            'vix': '^VIX', 'treasury': '^TNX', 'oil': 'CL=F',
            'spy': 'SPY', 'usdeur': 'EURUSD=X', 'silver': 'SI=F'
        }
        
        # إعدادات محسنة
        self.overbought_threshold = 70
        self.oversold_threshold = 30
        self.extreme_overbought = 80
        self.extreme_oversold = 20
        
        # مكونات النظام
        self.ml_predictor = MLPredictor()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.news_analyzer = AdvancedNewsAnalyzer(os.getenv("NEWS_API_KEY"))
        self.db_manager = DatabaseManager()
        self.backtester = ProfessionalBacktester(self)
        
        # APIs
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
    
    def fetch_multi_timeframe_data(self):
        """جلب بيانات متعددة الأطر الزمنية - محسن"""
        print("📊 جلب بيانات متعددة الأطر الزمنية...")
        try:
            # البيانات اليومية
            daily_data = yf.download(list(self.symbols.values()), 
                                    period="3y", interval="1d", 
                                    group_by='ticker', progress=False)
            
            # بيانات 4 ساعات
            hourly_data = yf.download(self.symbols['gold'], 
                                     period="1mo", interval="1h", 
                                     progress=False)
            
            # بيانات أسبوعية
            weekly_data = yf.download(self.symbols['gold'], 
                                     period="2y", interval="1wk", 
                                     progress=False)
            
            if daily_data.empty: 
                raise ValueError("فشل جلب البيانات")
            
            return {
                'daily': daily_data, 
                'hourly': hourly_data,
                'weekly': weekly_data
            }
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب - محسن"""
        print("🔍 استخراج بيانات الذهب...")
        try:
            daily_data = market_data['daily']
            gold_symbol = self.symbols['gold']
            
            if not (gold_symbol in daily_data.columns.levels[0] and 
                   not daily_data[gold_symbol].dropna().empty):
                gold_symbol = self.symbols['gold_etf']
                if not (gold_symbol in daily_data.columns.levels[0] and 
                       not daily_data[gold_symbol].dropna().empty):
                    raise ValueError("لا توجد بيانات للذهب")
            
            gold_daily = daily_data[gold_symbol].copy()
            gold_daily.dropna(subset=['Close'], inplace=True)
            
            if len(gold_daily) < 200: 
                raise ValueError("بيانات غير كافية")
            
            print(f"✅ بيانات يومية نظيفة: {len(gold_daily)} يوم")
            return gold_daily
        except Exception as e:
            print(f"❌ خطأ في استخراج بيانات الذهب: {e}")
            return None
    
    def calculate_professional_indicators(self, gold_data):
        """حساب المؤشرات الاحترافية المحسّنة"""
        print("📊 حساب المؤشرات الاحترافية المحسّنة...")
        try:
            df = gold_data.copy()
            
            # المتوسطات المتحركة
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # EMA
            df['EMA_9'] = df['Close'].ewm(span=9).mean()
            df['EMA_21'] = df['Close'].ewm(span=21).mean()
            
            # التقاطعات الذهبية/الموت
            df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)
            
            # RSI محسّن
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain / loss))
            
            # RSI Divergence
            df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
            
            # MACD محسّن
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            
            # Bollinger Bands محسّن
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA_20'] + (std * 2)
            df['BB_Lower'] = df['SMA_20'] - (std * 2)
            df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']) * 100
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR & Volatility
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
            
            # Volume Analysis محسّن
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
            df['Volume_Price_Trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
            
            # مؤشرات إضافية
            df['ROC'] = ((df['Close'] - df['Close'].shift(14)) / df['Close'].shift(14)) * 100
            df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                                (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
            
            # Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
            
            # Ichimoku Cloud
            high_9 = df['High'].rolling(9).max()
            low_9 = df['Low'].rolling(9).min()
            df['Tenkan_sen'] = (high_9 + low_9) / 2
            
            high_26 = df['High'].rolling(26).max()
            low_26 = df['Low'].rolling(26).min()
            df['Kijun_sen'] = (high_26 + low_26) / 2
            
            df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
            
            high_52 = df['High'].rolling(52).max()
            low_52 = df['Low'].rolling(52).min()
            df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
            
            return df.dropna()
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return gold_data
    
    def analyze_market_conditions_enhanced(self, data):
        """تحليل حالة السوق المحسن"""
        try:
            latest = data.iloc[-1]
            
            # تحليل RSI المحسن
            rsi = latest.get('RSI', 50)
            rsi_condition = self._analyze_rsi_enhanced(rsi)
            
            # تحليل Bollinger Bands المحسن
            bb_position = latest.get('BB_Position', 0.5)
            bb_condition = self._analyze_bollinger_bands_enhanced(bb_position)
            
            # تحليل الحجم المحسن
            volume_ratio = latest.get('Volume_Ratio', 1)
            volume_condition = self._analyze_volume_enhanced(volume_ratio)
            
            # تحليل الاتجاه المحسن
            trend_condition = self._analyze_trend_enhanced(latest)
            
            # تحليل MACD
            macd_condition = self._analyze_macd_enhanced(latest)
            
            # تحليل Stochastic
            stoch_condition = self._analyze_stochastic_enhanced(latest)
            
            return {
                'rsi_condition': rsi_condition,
                'bb_condition': bb_condition,
                'volume_condition': volume_condition,
                'trend_condition': trend_condition,
                'macd_condition': macd_condition,
                'stoch_condition': stoch_condition,
                'overall_risk': self._calculate_overall_risk_enhanced(
                    rsi_condition, bb_condition, volume_condition, 
                    trend_condition, macd_condition, stoch_condition
                )
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل حالة السوق: {e}")
            return {}
    
    def _analyze_rsi_enhanced(self, rsi):
        """تحليل RSI محسن"""
        if rsi >= self.extreme_overbought:
            return {
                'status': 'extreme_overbought',
                'signal': 'sell',
                'risk': 'very_high',
                'strength': 3,
                'message': f'RSI في منطقة التشبع الشديد ({rsi:.1f}) - خطر هبوط كبير'
            }
        elif rsi >= self.overbought_threshold:
            return {
                'status': 'overbought',
                'signal': 'sell',
                'risk': 'high',
                'strength': 2,
                'message': f'RSI في منطقة التشبع ({rsi:.1f}) - تجنب الشراء'
            }
        elif rsi <= self.extreme_oversold:
            return {
                'status': 'extreme_oversold',
                'signal': 'buy',
                'risk': 'low',
                'strength': 3,
                'message': f'RSI في منطقة الذروة البيعية ({rsi:.1f}) - فرصة شراء'
            }
        elif rsi <= self.oversold_threshold:
            return {
                'status': 'oversold',
                'signal': 'buy',
                'risk': 'medium',
                'strength': 2,
                'message': f'RSI في منطقة البيع ({rsi:.1f}) - مراقبة فرص الشراء'
            }
        else:
            return {
                'status': 'neutral',
                'signal': 'hold',
                'risk': 'medium',
                'strength': 0,
                'message': f'RSI محايد ({rsi:.1f}) - انتظار إشارات واضحة'
            }
    
    def _analyze_bollinger_bands_enhanced(self, bb_position):
        """تحليل Bollinger Bands محسن"""
        if bb_position > 1.0:
            return {
                'status': 'above_upper',
                'signal': 'sell',
                'risk': 'high',
                'strength': 2,
                'message': f'السعر فوق الحد العلوي (BB: {bb_position:.2f}) - احتمال تصحيح'
            }
        elif bb_position < 0.0:
            return {
                'status': 'below_lower',
                'signal': 'buy',
                'risk': 'low',
                'strength': 2,
                'message': f'السعر تحت الحد السفلي (BB: {bb_position:.2f}) - فرصة شراء'
            }
        else:
            return {
                'status': 'within_bands',
                'signal': 'hold',
                'risk': 'medium',
                'strength': 0,
                'message': f'السعر ضمن النطاق الطبيعي (BB: {bb_position:.2f})'
            }
    
    def _analyze_volume_enhanced(self, volume_ratio):
        """تحليل الحجم محسن"""
        if volume_ratio > 3.0:
            return {
                'status': 'extremely_high',
                'signal': 'caution',
                'risk': 'high',
                'strength': 2,
                'message': f'حجم استثنائي ({volume_ratio:.1f}x) - احتمال حركة قوية'
            }
        elif volume_ratio > 1.5:
            return {
                'status': 'high',
                'signal': 'positive',
                'risk': 'medium',
                'strength': 1,
                'message': f'حجم فوق المتوسط ({volume_ratio:.1f}x) - اهتمام متزايد'
            }
        elif volume_ratio < 0.5:
            return {
                'status': 'low',
                'signal': 'caution',
                'risk': 'medium',
                'strength': 1,
                'message': f'حجم ضعيف ({volume_ratio:.1f}x) - حذر من الحركة الوهمية'
            }
        else:
            return {
                'status': 'normal',
                'signal': 'neutral',
                'risk': 'low',
                'strength': 0,
                'message': f'حجم طبيعي ({volume_ratio:.1f}x)'
            }
    
    def _analyze_trend_enhanced(self, latest):
        """تحليل الاتجاه محسن"""
        try:
            close = latest['Close']
            sma_20 = latest.get('SMA_20', close)
            sma_50 = latest.get('SMA_50', close)
            sma_200 = latest.get('SMA_200', close)
            
            # حساب قوة الاتجاه
            trend_strength = 0
            
            if close > sma_200:
                trend_strength += 2
                if close > sma_50:
                    trend_strength += 1
                    if close > sma_20:
                        trend_strength += 1
            else:
                trend_strength -= 2
                if close < sma_50:
                    trend_strength -= 1
                    if close < sma_20:
                        trend_strength -= 1
            
            # تحليل التقاطعات
            golden_cross = latest.get('Golden_Cross', 0)
            death_cross = latest.get('Death_Cross', 0)
            
            if golden_cross == 1:
                trend_strength += 2
            elif death_cross == 1:
                trend_strength -= 2
            
            if trend_strength >= 4:
                return {
                    'status': 'very_strong_uptrend',
                    'signal': 'buy',
                    'strength': trend_strength,
                    'message': 'اتجاه صاعد قوي جداً'
                }
            elif trend_strength >= 2:
                return {
                    'status': 'strong_uptrend',
                    'signal': 'buy',
                    'strength': trend_strength,
                    'message': 'اتجاه صاعد قوي'
                }
            elif trend_strength >= 1:
                return {
                    'status': 'uptrend',
                    'signal': 'buy',
                    'strength': trend_strength,
                    'message': 'اتجاه صاعد'
                }
            elif trend_strength <= -4:
                return {
                    'status': 'very_strong_downtrend',
                    'signal': 'sell',
                    'strength': trend_strength,
                    'message': 'اتجاه هابط قوي جداً'
                }
            elif trend_strength <= -2:
                return {
                    'status': 'strong_downtrend',
                    'signal': 'sell',
                    'strength': trend_strength,
                    'message': 'اتجاه هابط قوي'
                }
            elif trend_strength <= -1:
                return {
                    'status': 'downtrend',
                    'signal': 'sell',
                    'strength': trend_strength,
                    'message': 'اتجاه هابط'
                }
            else:
                return {
                    'status': 'sideways',
                    'signal': 'hold',
                    'strength': trend_strength,
                    'message': 'سوق عرضي'
                }
        except Exception as e:
            return {
                'status': 'error',
                'signal': 'hold',
                'message': f'خطأ في تحليل الاتجاه: {e}'
            }
    
    def _analyze_macd_enhanced(self, latest):
        """تحليل MACD محسن"""
        try:
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_histogram = latest.get('MACD_Histogram', 0)
            
            if macd > macd_signal:
                if macd_histogram > 0:
                    return {
                        'status': 'bullish_momentum',
                        'signal': 'buy',
                        'strength': 2,
                        'message': 'MACD إيجابي مع زخم صاعد'
                    }
                else:
                    return {
                        'status': 'bullish_weakening',
                        'signal': 'buy',
                        'strength': 1,
                        'message': 'MACD إيجابي لكن الزخم يضعف'
                    }
            else:
                if macd_histogram < 0:
                    return {
                        'status': 'bearish_momentum',
                        'signal': 'sell',
                        'strength': 2,
                        'message': 'MACD سلبي مع زخم هابط'
                    }
                else:
                    return {
                        'status': 'bearish_weakening',
                        'signal': 'sell',
                        'strength': 1,
                        'message': 'MACD سلبي لكن الزخم يضعف'
                    }
        except Exception as e:
            return {
                'status': 'error',
                'signal': 'hold',
                'message': f'خطأ في تحليل MACD: {e}'
            }
    
    def _analyze_stochastic_enhanced(self, latest):
        """تحليل Stochastic محسن"""
        try:
            stoch_k = latest.get('Stoch_K', 50)
            stoch_d = latest.get('Stoch_D', 50)
            
            if stoch_k > 80 and stoch_d > 80:
                return {
                    'status': 'overbought',
                    'signal': 'sell',
                    'strength': 2,
                    'message': f'Stochastic في منطقة التشبع ({stoch_k:.1f}/{stoch_d:.1f})'
                }
            elif stoch_k < 20 and stoch_d < 20:
                return {
                    'status': 'oversold',
                    'signal': 'buy',
                    'strength': 2,
                    'message': f'Stochastic في منطقة البيع ({stoch_k:.1f}/{stoch_d:.1f})'
                }
            elif stoch_k > stoch_d:
                return {
                    'status': 'bullish',
                    'signal': 'buy',
                    'strength': 1,
                    'message': f'Stochastic إيجابي ({stoch_k:.1f} > {stoch_d:.1f})'
                }
            else:
                return {
                    'status': 'bearish',
                    'signal': 'sell',
                    'strength': 1,
                    'message': f'Stochastic سلبي ({stoch_k:.1f} < {stoch_d:.1f})'
                }
        except Exception as e:
            return {
                'status': 'error',
                'signal': 'hold',
                'message': f'خطأ في تحليل Stochastic: {e}'
            }
    
    def _calculate_overall_risk_enhanced(self, rsi_cond, bb_cond, volume_cond, 
                                       trend_cond, macd_cond, stoch_cond):
        """حساب المخاطر الإجمالية المحسن"""
        risk_score = 0
        
        # RSI Risk
        if rsi_cond.get('risk') == 'very_high':
            risk_score += 3
        elif rsi_cond.get('risk') == 'high':
            risk_score += 2
        elif rsi_cond.get('risk') == 'low':
            risk_score -= 1
        
        # Bollinger Bands Risk
        if bb_cond.get('risk') == 'high':
            risk_score += 2
        elif bb_cond.get('risk') == 'low':
            risk_score -= 1
        
        # Volume Risk
        if volume_cond.get('risk') == 'high':
            risk_score += 1
        
        # Trend Risk
        trend_strength = trend_cond.get('strength', 0)
        if trend_strength >= 4:
            risk_score -= 2  # اتجاه قوي يقلل المخاطر
        elif trend_strength <= -4:
            risk_score += 2  # اتجاه هابط قوي يزيد المخاطر
        
        # MACD Risk
        if macd_cond.get('signal') == 'sell':
            risk_score += 1
        
        # Stochastic Risk
        if stoch_cond.get('status') == 'overbought':
            risk_score += 1
        elif stoch_cond.get('status') == 'oversold':
            risk_score -= 1
        
        return {
            'score': risk_score,
            'level': 'very_high' if risk_score >= 5 else 'high' if risk_score >= 3 else 'medium' if risk_score >= 1 else 'low',
            'recommendation': self._get_risk_recommendation_enhanced(risk_score)
        }
    
    def _get_risk_recommendation_enhanced(self, risk_score):
        """توصية محسنة بناءً على مستوى المخاطر"""
        if risk_score >= 5:
            return "تجنب التداول تماماً - مخاطر عالية جداً"
        elif risk_score >= 3:
            return "تداول بحذر شديد - حجم صغير جداً (1-2%)"
        elif risk_score >= 1:
            return "تداول بحذر - حجم صغير (2-5%)"
        elif risk_score >= 0:
            return "تداول عادي - مراقبة المخاطر (5-10%)"
        else:
            return "فرص جيدة - يمكن التداول بحجم أكبر (10-20%)"
    
    def generate_professional_signals_v5(self, market_conditions, data):
        """توليد إشارات احترافية V5"""
        print("🎯 توليد إشارات احترافية V5...")
        
        try:
            latest = data.iloc[-1]
            current_price = latest['Close']
            
            # تحليل المخاطر
            overall_risk = market_conditions.get('overall_risk', {})
            risk_level = overall_risk.get('level', 'medium')
            
            # حساب نقاط القوة الإجمالية
            total_strength = 0
            signals = []
            
            for condition_name, condition in market_conditions.items():
                if condition_name != 'overall_risk':
                    strength = condition.get('strength', 0)
                    signal = condition.get('signal', 'hold')
                    total_strength += strength
                    signals.append((signal, strength))
            
            # منطق الإشارات المحسن V5
            if risk_level == 'very_high':
                signal = "Hold"
                confidence = "Low"
                action = "انتظار - مخاطر عالية جداً"
                position_size = "عدم التداول"
            elif risk_level == 'high':
                # تحليل إضافي للمخاطر العالية
                rsi_cond = market_conditions.get('rsi_condition', {})
                if rsi_cond.get('status') == 'extreme_overbought':
                    signal = "Sell"
                    confidence = "High"
                    action = "بيع - ذروة شراء شديدة"
                    position_size = "حجم متوسط"
                elif rsi_cond.get('status') == 'extreme_oversold':
                    signal = "Buy"
                    confidence = "High"
                    action = "شراء - ذروة بيع شديدة"
                    position_size = "حجم متوسط"
                else:
                    signal = "Hold"
                    confidence = "Medium"
                    action = "انتظار - مخاطر عالية"
                    position_size = "حجم صغير جداً"
            elif risk_level == 'medium':
                # تحليل متقدم للمخاطر المتوسطة
                buy_signals = sum(1 for s, _ in signals if s == 'buy')
                sell_signals = sum(1 for s, _ in signals if s == 'sell')
                
                if buy_signals > sell_signals and total_strength >= 3:
                    signal = "Buy"
                    confidence = "Medium"
                    action = "شراء - إشارات إيجابية"
                    position_size = "حجم متوسط"
                elif sell_signals > buy_signals and total_strength >= 3:
                    signal = "Sell"
                    confidence = "Medium"
                    action = "بيع - إشارات سلبية"
                    position_size = "حجم متوسط"
                else:
                    signal = "Hold"
                    confidence = "Low"
                    action = "انتظار - عدم وضوح"
                    position_size = "حجم صغير"
            else:
                # مخاطر منخفضة - تحليل الاتجاه
                trend_cond = market_conditions.get('trend_condition', {})
                if trend_cond.get('status') in ['very_strong_uptrend', 'strong_uptrend']:
                    signal = "Buy"
                    confidence = "High"
                    action = "شراء - اتجاه قوي"
                    position_size = "حجم كبير"
                elif trend_cond.get('status') in ['very_strong_downtrend', 'strong_downtrend']:
                    signal = "Sell"
                    confidence = "High"
                    action = "بيع - اتجاه هابط قوي"
                    position_size = "حجم كبير"
                else:
                    signal = "Hold"
                    confidence = "Medium"
                    action = "انتظار - سوق عرضي"
                    position_size = "حجم متوسط"
            
            # إدارة المخاطر المحسنة V5
            atr = self._calculate_atr_enhanced(data)
            risk_management = {
                'stop_loss_levels': {
                    'tight': round(current_price - (atr * 1.5), 2),
                    'conservative': round(current_price - (atr * 2), 2),
                    'wide': round(current_price - (atr * 3), 2)
                },
                'profit_targets': {
                    'target_1': round(current_price + (atr * 2), 2),
                    'target_2': round(current_price + (atr * 4), 2),
                    'target_3': round(current_price + (atr * 6), 2)
                },
                'position_size': position_size,
                'max_risk_per_trade': self._get_max_risk_per_trade(risk_level),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(atr, current_price),
                'risk_warnings': self._generate_risk_warnings_enhanced(market_conditions)
            }
            
            return {
                'signal': signal,
                'confidence': confidence,
                'action': action,
                'current_price': round(current_price, 2),
                'risk_level': risk_level,
                'total_strength': total_strength,
                'signal_breakdown': self._get_signal_breakdown(signals),
                'risk_management': risk_management,
                'market_conditions': market_conditions,
                'warnings': self._generate_warnings_enhanced(market_conditions)
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {
                'signal': 'Hold',
                'confidence': 'Low',
                'action': 'خطأ في التحليل',
                'error': str(e)
            }
    
    def _calculate_atr_enhanced(self, data, period=14):
        """حساب ATR محسن"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(period).mean().iloc[-1]
        except:
            return data['Close'].iloc[-1] * 0.02
    
    def _get_max_risk_per_trade(self, risk_level):
        """الحصول على الحد الأقصى للمخاطرة"""
        risk_map = {
            'very_high': '0.5%',
            'high': '1%',
            'medium': '2%',
            'low': '3%'
        }
        return risk_map.get(risk_level, '2%')
    
    def _calculate_risk_reward_ratio(self, atr, current_price):
        """حساب نسبة المخاطرة إلى المكافأة"""
        try:
            stop_loss = current_price - (atr * 2)
            target = current_price + (atr * 4)
            risk = current_price - stop_loss
            reward = target - current_price
            return round(reward / risk, 2) if risk > 0 else 0
        except:
            return 2.0
    
    def _get_signal_breakdown(self, signals):
        """تحليل تفصيلي للإشارات"""
        breakdown = {
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'total_strength': 0
        }
        
        for signal, strength in signals:
            if signal == 'buy':
                breakdown['buy_signals'] += 1
            elif signal == 'sell':
                breakdown['sell_signals'] += 1
            else:
                breakdown['hold_signals'] += 1
            breakdown['total_strength'] += strength
        
        return breakdown
    
    def _generate_risk_warnings_enhanced(self, conditions):
        """توليد تحذيرات المخاطر المحسنة"""
        warnings = []
        
        for condition_name, condition in conditions.items():
            if condition_name == 'overall_risk':
                continue
                
            if condition.get('risk') == 'very_high':
                warnings.append(f"🚨 {condition_name}: {condition['message']}")
            elif condition.get('risk') == 'high':
                warnings.append(f"⚠️ {condition_name}: {condition['message']}")
        
        return warnings
    
    def _generate_warnings_enhanced(self, conditions):
        """توليد تحذيرات عامة محسنة"""
        warnings = []
        
        rsi_cond = conditions.get('rsi_condition', {})
        if rsi_cond.get('status') in ['extreme_overbought', 'overbought']:
            warnings.append("تجنب الشراء في هذه المنطقة")
        
        trend_cond = conditions.get('trend_condition', {})
        if trend_cond.get('status') == 'sideways':
            warnings.append("السوق عرضي - انتظار اتجاه واضح")
        
        volume_cond = conditions.get('volume_condition', {})
        if volume_cond.get('status') == 'extremely_high':
            warnings.append("حجم استثنائي - احتمال حركة قوية")
        
        return warnings
    
    def generate_report_v5(self, analysis_result):
        """توليد تقرير V5 محسن"""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 تقرير التحليل الاحترافي للذهب - الإصدار 5.0")
            report.append("=" * 80)
            report.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # الإشارة الرئيسية
            if 'signal' in analysis_result:
                report.append("🎯 الإشارة الرئيسية:")
                report.append(f"  • الإشارة: {analysis_result['signal']}")
                report.append(f"  • الثقة: {analysis_result['confidence']}")
                report.append(f"  • التوصية: {analysis_result['action']}")
                report.append(f"  • السعر الحالي: ${analysis_result['current_price']}")
                report.append(f"  • مستوى المخاطر: {analysis_result['risk_level']}")
                report.append(f"  • قوة الإشارة الإجمالية: {analysis_result.get('total_strength', 0)}")
                report.append("")
            
            # تحليل الإشارات التفصيلي
            if 'signal_breakdown' in analysis_result:
                breakdown = analysis_result['signal_breakdown']
                report.append("📈 تحليل الإشارات التفصيلي:")
                report.append(f"  • إشارات الشراء: {breakdown['buy_signals']}")
                report.append(f"  • إشارات البيع: {breakdown['sell_signals']}")
                report.append(f"  • إشارات الانتظار: {breakdown['hold_signals']}")
                report.append(f"  • القوة الإجمالية: {breakdown['total_strength']}")
                report.append("")
            
            # تحليل حالة السوق
            if 'market_conditions' in analysis_result:
                mc = analysis_result['market_conditions']
                report.append("📊 تحليل حالة السوق:")
                
                for condition_name, condition in mc.items():
                    if condition_name != 'overall_risk':
                        report.append(f"  • {condition_name.upper()}: {condition['message']}")
                
                report.append("")
            
            # إدارة المخاطر
            if 'risk_management' in analysis_result:
                rm = analysis_result['risk_management']
                report.append("⚠️ إدارة المخاطر:")
                report.append(f"  • حجم المركز: {rm['position_size']}")
                report.append(f"  • وقف الخسارة المحافظ: ${rm['stop_loss_levels']['conservative']}")
                report.append(f"  • الهدف الأول: ${rm['profit_targets']['target_1']}")
                report.append(f"  • المخاطرة القصوى: {rm['max_risk_per_trade']}")
                report.append(f"  • نسبة المخاطرة/المكافأة: {rm['risk_reward_ratio']}")
                report.append("")
            
            # التحذيرات
            if 'warnings' in analysis_result and analysis_result['warnings']:
                report.append("🚨 التحذيرات:")
                for warning in analysis_result['warnings']:
                    report.append(f"  • {warning}")
                report.append("")
            
            # تحذيرات المخاطر
            if 'risk_management' in analysis_result and 'risk_warnings' in analysis_result['risk_management']:
                risk_warnings = analysis_result['risk_management']['risk_warnings']
                if risk_warnings:
                    report.append("⚠️ تحذيرات المخاطر:")
                    for warning in risk_warnings:
                        report.append(f"  • {warning}")
                    report.append("")
            
            report.append("=" * 80)
            report.append("انتهى التقرير - الإصدار 5.0")
            report.append("تم تطوير: منطق محسن | إدارة مخاطر شاملة | تحليل تفصيلي")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"
    
    def run_analysis_v5(self):
        """تشغيل التحليل الاحترافي V5"""
        print("🚀 بدء التحليل الاحترافي للذهب - الإصدار 5.0...")
        print("=" * 80)
        
        try:
            # 1. جلب البيانات
            market_data = self.fetch_multi_timeframe_data()
            if market_data is None:
                raise ValueError("فشل في جلب بيانات السوق")
            
            # 2. استخراج بيانات الذهب
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None:
                raise ValueError("فشل في استخراج بيانات الذهب")
            
            # 3. حساب المؤشرات الفنية
            technical_data = self.calculate_professional_indicators(gold_data)
            
            # 4. تحليل حالة السوق المحسن
            market_conditions = self.analyze_market_conditions_enhanced(technical_data)
            
            # 5. توليد الإشارات الاحترافية V5
            signals = self.generate_professional_signals_v5(market_conditions, technical_data)
            
            # 6. تجميع النتائج
            final_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'version': '5.0',
                **signals
            }
            
            # 7. حفظ النتائج
            self.save_results_v5(final_result)
            
            # 8. توليد التقرير
            report = self.generate_report_v5(final_result)
            print(report)
            
            print("\n✅ تم إتمام التحليل الاحترافي V5.0 بنجاح!")
            return final_result
            
        except Exception as e:
            error_message = f"❌ فشل التحليل: {e}"
            print(error_message)
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'version': '5.0',
                'error': str(e)
            }
            self.save_results_v5(error_result)
            return error_result
    
    def save_results_v5(self, results):
        """حفظ النتائج V5"""
        try:
            filename = "gold_analysis_professional_v5.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل في: {filename}")
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

# المكونات الإضافية (سيتم إضافتها في الجزء التالي)
class MLPredictor:
    """نظام التنبؤ بالتعلم الآلي - محسن"""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "gold_ml_model_v5.pkl"
        self.scaler_path = "gold_scaler_v5.pkl"

class MultiTimeframeAnalyzer:
    """محلل متعدد الأطر الزمنية - محسن"""
    def __init__(self):
        self.timeframes = {
            '1h': {'period': '5d', 'weight': 0.2},
            '4h': {'period': '1mo', 'weight': 0.3},
            '1d': {'period': '3mo', 'weight': 0.5}
        }

class AdvancedNewsAnalyzer:
    """محلل أخبار متقدم - محسن"""
    def __init__(self, api_key):
        self.api_key = api_key

class DatabaseManager:
    """مدير قاعدة البيانات - محسن"""
    def __init__(self, db_path="analysis_history_v5.db"):
        self.db_path = db_path

class ProfessionalBacktester:
    """نظام اختبار خلفي احترافي - محسن"""
    def __init__(self, analyzer):
        self.analyzer = analyzer

def main():
    """الدالة الرئيسية"""
    analyzer = ProfessionalGoldAnalyzerV5()
    analyzer.run_analysis_v5()

if __name__ == "__main__":
    main()
