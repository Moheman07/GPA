#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class ProfessionalGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'treasury': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY',
            'usdeur': 'EURUSD=X'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")

    def fetch_multi_timeframe_data(self):
        """جلب بيانات متعددة الأطر الزمنية"""
        print("📊 جلب بيانات متعددة الأطر الزمنية...")
        try:
            symbols_list = list(self.symbols.values())
            daily_data = yf.download(symbols_list, period="1y", interval="1d", group_by='ticker', progress=False)
            if daily_data.empty:
                raise ValueError("فشل جلب البيانات")
            print("✅ تم جلب البيانات")
            return {'daily': daily_data}
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب مع تحسينات"""
        try:
            daily_data = market_data['daily']
            if hasattr(daily_data.columns, 'levels') and len(daily_data.columns.levels) > 1:
                available_symbols = daily_data.columns.get_level_values(0).unique()
                gold_symbol_to_use = None
                if self.symbols['gold'] in available_symbols and not daily_data[self.symbols['gold']].dropna().empty:
                    gold_symbol_to_use = self.symbols['gold']
                elif self.symbols['gold_etf'] in available_symbols and not daily_data[self.symbols['gold_etf']].dropna().empty:
                    gold_symbol_to_use = self.symbols['gold_etf']
                
                if gold_symbol_to_use:
                    gold_daily = daily_data[gold_symbol_to_use].copy()
                    print(f"✅ تم استخدام بيانات {gold_symbol_to_use}")
                else:
                    raise ValueError("لا يمكن العثور على بيانات الذهب (GC=F or GLD)")
            else:
                gold_daily = daily_data.copy()
            
            gold_daily = gold_daily.dropna(subset=['Close'])
            if len(gold_daily) < 200:
                raise ValueError("بيانات غير كافية لحساب المتوسطات طويلة الأمد")
            print(f"✅ بيانات يومية نظيفة: {len(gold_daily)} يوم")
            return gold_daily
        except Exception as e:
            print(f"❌ خطأ في استخراج بيانات الذهب: {e}")
            return None

    def calculate_professional_indicators(self, gold_data):
        """حساب مؤشرات احترافية متقدمة"""
        try:
            print("📊 حساب المؤشرات الاحترافية...")
            df = gold_data.copy()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA_20'] + (std * 2)
            df['BB_Lower'] = df['SMA_20'] - (std * 2)
            df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']) * 100
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['ROC'] = ((df['Close'] - df['Close'].shift(14)) / df['Close'].shift(14)) * 100
            df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                                     (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
            print("✅ تم حساب المؤشرات الاحترافية")
            return df.dropna()
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return gold_data

    def calculate_fibonacci_levels(self, data, periods=50):
        try:
            recent_data = data.tail(periods)
            high, low = recent_data['High'].max(), recent_data['Low'].min()
            diff = high - low
            return {
                'high': round(high, 2), 'low': round(low, 2),
                'fib_23_6': round(high - (diff * 0.236), 2), 'fib_38_2': round(high - (diff * 0.382), 2),
                'fib_50_0': round(high - (diff * 0.500), 2), 'fib_61_8': round(high - (diff * 0.618), 2),
            }
        except Exception as e:
            print(f"❌ خطأ في حساب فيبوناتشي: {e}")
            return {}

    def analyze_volume_profile(self, data):
        try:
            latest = data.iloc[-1]
            return {
                'current_volume': int(latest.get('Volume', 0)),
                'avg_volume_20': int(data.tail(20)['Volume'].mean()),
                'volume_ratio': round(latest.get('Volume_Ratio', 1), 2),
                'volume_strength': 'قوي' if latest.get('Volume_Ratio', 1) > 1.5 else ('ضعيف' if latest.get('Volume_Ratio', 1) < 0.7 else 'طبيعي')
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل الحجم: {e}")
            return {}

    def analyze_correlations(self, market_data):
        try:
            print("📊 تحليل الارتباطات المتقدم...")
            daily_data = market_data['daily']
            correlations, strength = {}, {}
            if hasattr(daily_data.columns, 'levels'):
                available_symbols = daily_data.columns.get_level_values(0).unique()
                gold_symbol = self.symbols['gold'] if self.symbols['gold'] in available_symbols else self.symbols['gold_etf']
                if gold_symbol in available_symbols:
                    gold_prices = daily_data[gold_symbol]['Close'].dropna()
                    for name, symbol in self.symbols.items():
                        if name not in ['gold', 'gold_etf'] and symbol in available_symbols:
                            asset_prices = daily_data[symbol]['Close'].dropna()
                            common_index = gold_prices.index.intersection(asset_prices.index)
                            if len(common_index) > 30:
                                corr = gold_prices.loc[common_index].corr(asset_prices.loc[common_index])
                                if pd.notna(corr):
                                    correlations[name] = round(corr, 3)
                                    if abs(corr) > 0.7: strength[name] = 'قوي جداً'
                                    elif abs(corr) > 0.5: strength[name] = 'قوي'
                                    elif abs(corr) > 0.3: strength[name] = 'متوسط'
                                    else: strength[name] = 'ضعيف'
            return {'correlations': correlations, 'strength_analysis': strength}
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {'correlations': {}, 'strength_analysis': {}}

    def fetch_news(self):
        print("📰 جلب أخبار الذهب المتخصصة...")
        if not self.news_api_key: return {"status": "no_api_key"}
        try:
            keywords = "gold OR XAU OR \"federal reserve\" OR inflation OR \"interest rate\""
            url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&pageSize=20&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            high_impact_kw = ['federal reserve', 'fed', 'interest rate', 'inflation']
            high, medium = [], []
            for article in articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                news_item = {'title': article.get('title'), 'source': article.get('source', {}).get('name')}
                if any(kw in content for kw in high_impact_kw): high.append(news_item)
                else: medium.append(news_item)
            return {"status": "success", "high_impact_news": high[:3], "medium_impact_news": medium[:3]}
        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return {"status": "error"}
    
    def generate_professional_signals(self, tech_data, correlations, volume, fib_levels):
        print("🎯 توليد إشارات احترافية...")
        try:
            latest, prev = tech_data.iloc[-1], tech_data.iloc[-2]
            trend_score, momentum_score, volume_score, fib_score, corr_score = 0, 0, 0, 0, 0
            
            # Trend
            if latest['Close'] > latest['SMA_200']: trend_score = 3 if latest['Close'] > latest['SMA_50'] else 2
            else: trend_score = -3 if latest['Close'] < latest['SMA_50'] else -2
            
            # Momentum
            if latest['MACD'] > latest['MACD_Signal']: momentum_score = 2 if latest['MACD_Histogram'] > prev['MACD_Histogram'] else 1
            else: momentum_score = -1
            if 40 <= latest['RSI'] <= 60: momentum_score += 1
            elif latest['RSI'] < 30: momentum_score += 2
            
            # Volume
            if volume.get('volume_strength') == 'قوي': volume_score = 1
            
            # Fibonacci
            if fib_levels and latest['Close'] > fib_levels.get('fib_61_8', 0): fib_score = 1
            
            # Correlation
            dxy_corr = correlations.get('correlations', {}).get('dxy', 0)
            if dxy_corr < -0.7: corr_score = 2
            elif dxy_corr < -0.5: corr_score = 1
            
            total_score = (trend_score * 0.30 + momentum_score * 0.25 + volume_score * 0.15 + fib_score * 0.15 + corr_score * 0.15)
            
            if total_score >= 1.5: signal, confidence = "Buy", "High"
            elif total_score >= 0.5: signal, confidence = "Weak Buy", "Medium"
            elif total_score <= -1.5: signal, confidence = "Sell", "High"
            elif total_score <= -0.5: signal, confidence = "Weak Sell", "Medium"
            else: signal, confidence = "Hold", "Low"
            
            atr = latest.get('ATR', latest['Close'] * 0.02)
            return {
                'signal': signal, 'confidence': confidence, 'total_score': round(total_score, 2),
                'component_scores': {'trend': trend_score, 'momentum': momentum_score, 'volume': volume_score, 'fibonacci': fib_score, 'correlation': corr_score},
                'current_price': round(latest['Close'], 2),
                'risk_management': {
                    'stop_loss_moderate': round(latest['Close'] - (atr * 2.0), 2),
                    'take_profit_target_1': round(latest['Close'] + (atr * 3.0), 2)
                }
            }
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}

    def run_analysis(self):
        print("🚀 بدء التحليل الاحترافي للذهب...")
        try:
            market_data = self.fetch_multi_timeframe_data()
            if market_data is None: raise ValueError("فشل في جلب بيانات السوق")
            
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None: raise ValueError("فشل في استخراج بيانات الذهب")
            
            technical_data = self.calculate_professional_indicators(gold_data)
            fibonacci_levels = self.calculate_fibonacci_levels(technical_data)
            volume_analysis = self.analyze_volume_profile(technical_data)
            correlations = self.analyze_correlations(market_data)
            news_data = self.fetch_news()
            signals = self.generate_professional_signals(technical_data, correlations, volume_analysis, fibonacci_levels)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'market_status': {'ny_market_status': 'Closed'}, # Simplified
                'gold_analysis': signals,
                'market_correlations': correlations,
                'news_analysis': news_data,
            }
            
            self.save_single_result(results)
            print("✅ تم إتمام التحليل الاحترافي بنجاح!")
            return results
        except Exception as e:
            print(f"❌ فشل التحليل الاحترافي: {e}")
            error_result = {'status': 'error', 'error': str(e)}
            self.save_single_result(error_result)
            return error_result

    def save_single_result(self, results):
        try:
            filename = "gold_analysis.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم تحديث الملف: {filename}")
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    print("=" * 60)
    print("🏆 محلل الذهب الاحترافي المتطور")
    print("=" * 60)
    analyzer = ProfessionalGoldAnalyzer()
    analyzer.run_analysis()
    print("=" * 60)
    print("🔔 انتهى التحليل الاحترافي")
    print("=" * 60)

if __name__ == "__main__":
    main()
