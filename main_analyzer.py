#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sqlite3
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import asyncio
import aiohttp
from textblob import TextBlob
import backtrader as bt

warnings.filterwarnings('ignore')

class GoldAnalyzerV4:
    """محلل الذهب الاحترافي - نسخة محسنة ومختصرة"""
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'dxy': 'DX-Y.NYB', 
            'vix': '^VIX',
            'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.db_path = "gold_analysis.db"
        self.init_database()
        
    def init_database(self):
        """تهيئة قاعدة البيانات"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    price REAL,
                    signal TEXT,
                    score REAL,
                    indicators TEXT,
                    success BOOLEAN
                )
            ''')
            conn.commit()
    
    def fetch_data(self):
        """جلب البيانات الأساسية"""
        print("📊 جلب البيانات...")
        try:
            # جلب بيانات متعددة بشكل آمن
            data = {}
            for name, symbol in self.symbols.items():
                try:
                    df = yf.download(symbol, period="6mo", interval="1d", progress=False)
                    if not df.empty:
                        data[name] = df
                except:
                    print(f"⚠️ تعذر جلب {symbol}")
                    
            if 'gold' not in data or data['gold'].empty:
                raise ValueError("فشل جلب بيانات الذهب")
                
            return data
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
    def calculate_indicators(self, df):
        """حساب المؤشرات الفنية الأساسية"""
        try:
            # المتوسطات المتحركة
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['SMA_20'] + (std * 2)
            df['BB_Lower'] = df['SMA_20'] - (std * 2)
            
            # Volume
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            return df.dropna()
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return df
    
    def analyze_multi_timeframe(self, symbol):
        """تحليل متعدد الأطر الزمنية - محسّن"""
        try:
            timeframes = {
                '1d': {'period': '1mo', 'weight': 0.5},
                '1wk': {'period': '3mo', 'weight': 0.3},
                '1mo': {'period': '1y', 'weight': 0.2}
            }
            
            total_score = 0
            results = {}
            
            for tf_name, tf_config in timeframes.items():
                try:
                    data = yf.download(symbol, period=tf_config['period'], 
                                     interval=tf_name, progress=False)
                    if not data.empty and len(data) > 20:
                        data = self.calculate_indicators(data)
                        if not data.empty:
                            latest = data.iloc[-1]
                            
                            # تحليل بسيط
                            score = 0
                            if latest['Close'] > latest.get('SMA_20', latest['Close']):
                                score += 1
                            if latest.get('RSI', 50) > 50:
                                score += 0.5
                            if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
                                score += 0.5
                                
                            results[tf_name] = {
                                'score': score,
                                'trend': 'صاعد' if score > 1 else 'هابط'
                            }
                            total_score += score * tf_config['weight']
                except:
                    continue
                    
            return total_score, results
        except Exception as e:
            print(f"خطأ في تحليل الأطر الزمنية: {e}")
            return 0, {}
    
    async def fetch_news(self):
        """جلب وتحليل الأخبار"""
        if not self.news_api_key:
            return {'sentiment': 0, 'count': 0}
            
        try:
            url = f"https://newsapi.org/v2/everything?q=gold+price&language=en&sortBy=publishedAt&pageSize=10&apiKey={self.news_api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    data = await response.json()
                    
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                sentiments = []
                
                for article in articles:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                return {
                    'sentiment': avg_sentiment,
                    'count': len(articles),
                    'impact': 'إيجابي' if avg_sentiment > 0.1 else 'سلبي' if avg_sentiment < -0.1 else 'محايد'
                }
        except:
            return {'sentiment': 0, 'count': 0}
    
    def calculate_correlations(self, market_data):
        """حساب الارتباطات"""
        try:
            if 'gold' not in market_data or 'dxy' not in market_data:
                return {}
                
            gold_returns = market_data['gold']['Close'].pct_change().dropna()
            correlations = {}
            
            for name, data in market_data.items():
                if name != 'gold' and not data.empty:
                    asset_returns = data['Close'].pct_change().dropna()
                    common_idx = gold_returns.index.intersection(asset_returns.index)
                    if len(common_idx) > 30:
                        corr = gold_returns.loc[common_idx].corr(asset_returns.loc[common_idx])
                        correlations[name] = round(corr, 3)
                        
            return correlations
        except:
            return {}
    
    def generate_signals(self, gold_data, mtf_score, news_sentiment, correlations):
        """توليد الإشارات النهائية"""
        try:
            latest = gold_data.iloc[-1]
            scores = {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'mtf': mtf_score,
                'news': news_sentiment.get('sentiment', 0) * 2,
                'correlation': 0
            }
            
            # تحليل الاتجاه
            if latest['Close'] > latest.get('SMA_200', latest['Close']):
                scores['trend'] += 2
            if latest['Close'] > latest.get('SMA_50', latest['Close']):
                scores['trend'] += 1
            if latest['Close'] > latest.get('SMA_20', latest['Close']):
                scores['trend'] += 1
                
            # تحليل الزخم
            if latest.get('RSI', 50) > 30 and latest.get('RSI', 50) < 70:
                if latest['RSI'] > 50:
                    scores['momentum'] += 1
                else:
                    scores['momentum'] -= 1
            elif latest.get('RSI', 50) < 30:
                scores['momentum'] += 2  # ذروة بيع
            else:
                scores['momentum'] -= 2  # ذروة شراء
                
            if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
                scores['momentum'] += 1
                
            # تحليل الحجم
            if latest.get('Volume_Ratio', 1) > 1.5:
                scores['volume'] = 2
            elif latest.get('Volume_Ratio', 1) > 1:
                scores['volume'] = 1
                
            # تحليل الارتباطات
            dxy_corr = correlations.get('dxy', 0)
            if dxy_corr < -0.5:
                scores['correlation'] = 2
            elif dxy_corr < -0.3:
                scores['correlation'] = 1
                
            # حساب النتيجة النهائية
            total_score = sum(scores.values())
            
            # تحديد الإشارة
            if total_score >= 6:
                signal = "Strong Buy"
                confidence = "عالية جداً"
            elif total_score >= 3:
                signal = "Buy"
                confidence = "عالية"
            elif total_score <= -6:
                signal = "Strong Sell"
                confidence = "عالية جداً"
            elif total_score <= -3:
                signal = "Sell"
                confidence = "عالية"
            else:
                signal = "Hold"
                confidence = "منخفضة"
                
            # إدارة المخاطر
            atr = latest.get('ATR', latest['Close'] * 0.02)
            price = latest['Close']
            
            risk_management = {
                'stop_loss': round(price - (atr * 2), 2),
                'take_profit_1': round(price + (atr * 2), 2),
                'take_profit_2': round(price + (atr * 4), 2),
                'position_size': self._get_position_size(confidence)
            }
            
            return {
                'signal': signal,
                'confidence': confidence,
                'total_score': round(total_score, 2),
                'scores': scores,
                'price': round(price, 2),
                'risk_management': risk_management,
                'technical_levels': {
                    'sma_20': round(latest.get('SMA_20', 0), 2),
                    'sma_50': round(latest.get('SMA_50', 0), 2),
                    'sma_200': round(latest.get('SMA_200', 0), 2),
                    'rsi': round(latest.get('RSI', 0), 1),
                    'volume_ratio': round(latest.get('Volume_Ratio', 1), 2)
                }
            }
        except Exception as e:
            print(f"خطأ في توليد الإشارات: {e}")
            return {'error': str(e)}
    
    def _get_position_size(self, confidence):
        """تحديد حجم المركز"""
        sizes = {
            'عالية جداً': '50-75% من رأس المال',
            'عالية': '25-50%',
            'متوسطة': '10-25%',
            'منخفضة': '5-10% أو عدم الدخول'
        }
        return sizes.get(confidence, '5-10%')
    
    def run_simple_backtest(self, data, signals_func):
        """اختبار خلفي مبسط"""
        try:
            initial_capital = 10000
            capital = initial_capital
            position = 0
            trades = []
            
            for i in range(100, len(data)):
                current_data = data.iloc[:i+1]
                signal = signals_func(current_data)
                
                if signal.get('signal') in ['Buy', 'Strong Buy'] and position == 0:
                    # شراء
                    position = capital / data.iloc[i]['Close']
                    entry_price = data.iloc[i]['Close']
                    
                elif signal.get('signal') in ['Sell', 'Strong Sell'] and position > 0:
                    # بيع
                    exit_price = data.iloc[i]['Close']
                    profit = (exit_price - entry_price) * position
                    capital += profit
                    trades.append({
                        'profit': profit,
                        'return': (exit_price - entry_price) / entry_price
                    })
                    position = 0
            
            # إغلاق المركز المفتوح في نهاية الفترة
            if position > 0:
                final_price = data.iloc[-1]['Close']
                profit = (final_price - entry_price) * position
                capital += profit
                trades.append({
                    'profit': profit,
                    'return': (final_price - entry_price) / entry_price
                })
            
            # حساب الإحصائيات
            total_return = ((capital - initial_capital) / initial_capital) * 100
            winning_trades = [t for t in trades if t['profit'] > 0]
            losing_trades = [t for t in trades if t['profit'] <= 0]
            
            return {
                'initial_capital': initial_capital,
                'final_capital': round(capital, 2),
                'total_return': round(total_return, 2),
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(len(winning_trades) / max(len(trades), 1) * 100, 2),
                'avg_win': round(np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0, 2),
                'avg_loss': round(np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0, 2)
            }
        except Exception as e:
            print(f"خطأ في الاختبار الخلفي: {e}")
            return None
    
    def save_to_database(self, analysis):
        """حفظ التحليل في قاعدة البيانات"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO analysis_history (price, signal, score, indicators)
                    VALUES (?, ?, ?, ?)
                ''', (
                    analysis.get('price'),
                    analysis.get('signal'),
                    analysis.get('total_score'),
                    json.dumps(analysis.get('technical_levels', {}))
                ))
                conn.commit()
        except Exception as e:
            print(f"خطأ في حفظ البيانات: {e}")
    
    def generate_report(self, analysis):
        """توليد تقرير مختصر"""
        report = []
        report.append("=" * 60)
        report.append("📊 تقرير تحليل الذهب - النسخة المحسنة")
        report.append("=" * 60)
        report.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'error' in analysis:
            report.append(f"❌ خطأ: {analysis['error']}")
        else:
            # الإشارة الرئيسية
            report.append("🎯 الإشارة الرئيسية:")
            report.append(f"  • الإشارة: {analysis.get('signal', 'N/A')}")
            report.append(f"  • الثقة: {analysis.get('confidence', 'N/A')}")
            report.append(f"  • السعر الحالي: ${analysis.get('price', 'N/A')}")
            report.append(f"  • النقاط الإجمالية: {analysis.get('total_score', 'N/A')}")
            report.append("")
            
            # تفاصيل النقاط
            if 'scores' in analysis:
                report.append("📈 تحليل المكونات:")
                for component, score in analysis['scores'].items():
                    report.append(f"  • {component}: {score}")
                report.append("")
            
            # إدارة المخاطر
            if 'risk_management' in analysis:
                rm = analysis['risk_management']
                report.append("⚠️ إدارة المخاطر:")
                report.append(f"  • وقف الخسارة: ${rm.get('stop_loss', 'N/A')}")
                report.append(f"  • الهدف الأول: ${rm.get('take_profit_1', 'N/A')}")
                report.append(f"  • الهدف الثاني: ${rm.get('take_profit_2', 'N/A')}")
                report.append(f"  • حجم المركز: {rm.get('position_size', 'N/A')}")
                report.append("")
            
            # المستويات الفنية
            if 'technical_levels' in analysis:
                tl = analysis['technical_levels']
                report.append("📊 المستويات الفنية:")
                report.append(f"  • SMA 20: ${tl.get('sma_20', 'N/A')}")
                report.append(f"  • SMA 50: ${tl.get('sma_50', 'N/A')}")
                report.append(f"  • SMA 200: ${tl.get('sma_200', 'N/A')}")
                report.append(f"  • RSI: {tl.get('rsi', 'N/A')}")
                report.append(f"  • نسبة الحجم: {tl.get('volume_ratio', 'N/A')}")
                report.append("")
            
            # تحليل الأطر الزمنية
            if 'mtf_analysis' in analysis:
                report.append("⏰ تحليل الأطر الزمنية:")
                for tf, data in analysis['mtf_analysis'].items():
                    report.append(f"  • {tf}: {data.get('trend', 'N/A')} (نقاط: {data.get('score', 0)})")
                report.append("")
            
            # تحليل الأخبار
            if 'news_analysis' in analysis:
                na = analysis['news_analysis']
                report.append("📰 تحليل الأخبار:")
                report.append(f"  • المشاعر: {na.get('impact', 'N/A')}")
                report.append(f"  • عدد المقالات: {na.get('count', 0)}")
                report.append("")
            
            # الارتباطات
            if 'correlations' in analysis:
                report.append("🔗 الارتباطات:")
                for asset, corr in analysis['correlations'].items():
                    impact = "إيجابي" if (asset == 'dxy' and corr < -0.3) else "سلبي" if (asset == 'dxy' and corr > 0.3) else "محايد"
                    report.append(f"  • {asset.upper()}: {corr} ({impact})")
                report.append("")
            
            # نتائج الاختبار الخلفي
            if 'backtest' in analysis and analysis['backtest']:
                bt = analysis['backtest']
                report.append("🔄 نتائج الاختبار الخلفي:")
                report.append(f"  • العائد الإجمالي: {bt.get('total_return', 0)}%")
                report.append(f"  • معدل الفوز: {bt.get('win_rate', 0)}%")
                report.append(f"  • عدد الصفقات: {bt.get('total_trades', 0)}")
                report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    async def run_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء تحليل الذهب المحسّن...")
        print("=" * 60)
        
        try:
            # 1. جلب البيانات
            market_data = self.fetch_data()
            if not market_data:
                raise ValueError("فشل جلب البيانات")
            
            # 2. حساب المؤشرات
            gold_data = self.calculate_indicators(market_data['gold'])
            
            # 3. تحليل متعدد الأطر الزمنية
            print("⏰ تحليل الأطر الزمنية...")
            mtf_score, mtf_results = self.analyze_multi_timeframe(self.symbols['gold'])
            
            # 4. جلب وتحليل الأخبار
            print("📰 تحليل الأخبار...")
            news_sentiment = await self.fetch_news()
            
            # 5. حساب الارتباطات
            print("🔗 حساب الارتباطات...")
            correlations = self.calculate_correlations(market_data)
            
            # 6. توليد الإشارات
            print("🎯 توليد الإشارات...")
            signals = self.generate_signals(gold_data, mtf_score, news_sentiment, correlations)
            
            # 7. اختبار خلفي بسيط
            print("🔄 تشغيل الاختبار الخلفي...")
            backtest_results = self.run_simple_backtest(
                gold_data, 
                lambda data: self.generate_signals(data, 0, {'sentiment': 0}, {})
            )
            
            # تجميع النتائج
            final_analysis = {
                **signals,
                'mtf_analysis': mtf_results,
                'news_analysis': news_sentiment,
                'correlations': correlations,
                'backtest': backtest_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # حفظ في قاعدة البيانات
            self.save_to_database(final_analysis)
            
            # حفظ في ملف JSON
            with open('gold_analysis_v4.json', 'w', encoding='utf-8') as f:
                json.dump(final_analysis, f, ensure_ascii=False, indent=2, default=str)
            
            # توليد وطباعة التقرير
            report = self.generate_report(final_analysis)
            print(report)
            
            print("\n✅ تم إتمام التحليل بنجاح!")
            return final_analysis
            
        except Exception as e:
            error_msg = f"❌ فشل التحليل: {e}"
            print(error_msg)
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

def main():
    """الدالة الرئيسية"""
    analyzer = GoldAnalyzerV4()
    asyncio.run(analyzer.run_analysis())

if __name__ == "__main__":
    main()
