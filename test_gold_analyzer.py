#!/usr/bin/env python3
"""
محلل الذهب المحسن لـ GitHub Actions
نسخة محسنة للعمل بشكل مثالي في بيئة GitHub Actions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime, timedelta

# Optional imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available - ML features disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not available - news analysis disabled")

warnings.filterwarnings('ignore')

class GitHubGoldAnalyzer:
    """محلل الذهب المحسن للعمل على GitHub Actions"""
    
    def __init__(self):
        """تهيئة المحلل"""
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'spy': 'SPY',
            'oil': 'CL=F'
        }
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
        
        print("🚀 تم تهيئة محلل الذهب المحسن لـ GitHub")
    
    def fetch_market_data(self):
        """جلب بيانات السوق"""
        print("📊 جلب بيانات السوق...")
        market_data = {}
        
        for name, symbol in self.symbols.items():
            try:
                print(f"📈 جلب {name}...")
                data = yf.download(symbol, period="6mo", interval="1d", progress=False)
                if not data.empty:
                    market_data[name] = data
                    print(f"✅ {name}: {len(data)} نقطة")
                else:
                    print(f"⚠️ لا توجد بيانات لـ {name}")
            except Exception as e:
                print(f"❌ خطأ في {name}: {e}")
        
        return market_data
    
    def calculate_indicators(self, data):
        """حساب المؤشرات الفنية"""
        print("📈 حساب المؤشرات...")
        df = data.copy()
        
        try:
            # المتوسطات المتحركة
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume Analysis
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Support/Resistance
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            
            print("✅ تم حساب المؤشرات الفنية")
            return df.dropna()
            
        except Exception as e:
            print(f"❌ خطأ في المؤشرات: {e}")
            return df
    
    def calculate_correlations(self, market_data):
        """حساب الارتباطات"""
        print("🔗 تحليل الارتباطات...")
        correlations = {}
        
        try:
            if 'gold' in market_data:
                gold_prices = market_data['gold']['Close'].dropna()
                
                for asset, data in market_data.items():
                    if asset != 'gold' and not data.empty:
                        asset_prices = data['Close'].dropna()
                        common_dates = gold_prices.index.intersection(asset_prices.index)
                        
                        if len(common_dates) > 30:
                            corr = gold_prices.loc[common_dates].corr(asset_prices.loc[common_dates])
                            correlations[asset] = round(corr, 3)
            
            print(f"✅ تم حساب {len(correlations)} ارتباط")
            return correlations
            
        except Exception as e:
            print(f"❌ خطأ في الارتباطات: {e}")
            return {}
    
    def analyze_news_simple(self):
        """تحليل الأخبار المبسط"""
        print("📰 تحليل الأخبار...")
        
        if not self.news_api_key or not REQUESTS_AVAILABLE:
            return {
                'status': 'simulated',
                'sentiment': 0.1,
                'impact': 'إيجابي معتدل للذهب'
            }
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'gold OR "federal reserve" OR inflation',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(days=3)).isoformat()
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # تحليل بسيط للعناوين
                positive_words = ['rise', 'gain', 'up', 'bull', 'strong', 'growth']
                negative_words = ['fall', 'drop', 'down', 'bear', 'weak', 'decline']
                
                sentiment_score = 0
                for article in articles[:10]:
                    title = article.get('title', '').lower()
                    for word in positive_words:
                        if word in title:
                            sentiment_score += 1
                    for word in negative_words:
                        if word in title:
                            sentiment_score -= 1
                
                normalized_sentiment = sentiment_score / max(len(articles), 1)
                
                return {
                    'status': 'success',
                    'articles_count': len(articles),
                    'sentiment': round(normalized_sentiment, 3),
                    'impact': self._get_impact_text(normalized_sentiment)
                }
            else:
                return self._get_simulated_news()
                
        except Exception as e:
            print(f"❌ خطأ في الأخبار: {e}")
            return self._get_simulated_news()
    
    def _get_simulated_news(self):
        """أخبار محاكاة"""
        return {
            'status': 'simulated',
            'sentiment': 0.15,
            'impact': 'إيجابي معتدل - مخاوف التضخم'
        }
    
    def _get_impact_text(self, sentiment):
        """تحويل النقاط إلى نص"""
        if sentiment > 0.3:
            return "إيجابي قوي للذهب"
        elif sentiment > 0.1:
            return "إيجابي معتدل للذهب"
        elif sentiment < -0.3:
            return "سلبي قوي للذهب"
        elif sentiment < -0.1:
            return "سلبي معتدل للذهب"
        else:
            return "محايد - لا تأثير واضح"
    
    def simple_ml_prediction(self, technical_data):
        """تنبؤ التعلم الآلي المبسط"""
        print("🤖 تحليل ML...")
        
        if not SKLEARN_AVAILABLE or len(technical_data) < 50:
            return {
                'prediction': 'صعود',
                'confidence': 0.65,
                'method': 'rule_based'
            }
        
        try:
            # إعداد المتغيرات
            features = ['RSI', 'MACD', 'Volume_Ratio']
            available_features = [f for f in features if f in technical_data.columns]
            
            if len(available_features) < 2:
                return {
                    'prediction': 'صعود',
                    'confidence': 0.6,
                    'method': 'insufficient_features'
                }
            
            X = technical_data[available_features].fillna(0)
            
            # إنشاء target بسيط
            price_change = technical_data['Close'].pct_change(3).shift(-3)
            y = (price_change > 0.005).astype(int)
            
            # تدريب سريع
            X_train = X[:-10].fillna(0)
            y_train = y[:-10].fillna(0)
            
            if len(X_train) > 20:
                self.model.fit(X_train, y_train)
                
                # تنبؤ للنقطة الحالية
                current = X.iloc[-1:].fillna(0)
                pred = self.model.predict(current)[0]
                prob = self.model.predict_proba(current)[0].max()
                
                return {
                    'prediction': 'صعود' if pred == 1 else 'هبوط',
                    'confidence': round(prob, 3),
                    'method': 'random_forest'
                }
            
            return {
                'prediction': 'صعود',
                'confidence': 0.6,
                'method': 'insufficient_data'
            }
            
        except Exception as e:
            print(f"❌ خطأ في ML: {e}")
            return {
                'prediction': 'صعود',
                'confidence': 0.6,
                'method': 'error_fallback'
            }
    
    def generate_signals(self, technical_data, correlations, ml_result, news_analysis):
        """توليد الإشارات النهائية"""
        print("🎯 توليد الإشارات...")
        
        try:
            latest = technical_data.iloc[-1]
            score = 0
            signals = {}
            
            # التحليل الفني
            if latest['Close'] > latest['SMA_20']:
                score += 1
                signals['trend'] = 'صاعد'
            else:
                score -= 1
                signals['trend'] = 'هابط'
            
            # RSI
            rsi = latest['RSI']
            if rsi < 30:
                score += 2
                signals['rsi'] = 'ذروة بيع'
            elif rsi > 70:
                score -= 2
                signals['rsi'] = 'ذروة شراء'
            else:
                signals['rsi'] = f'متوازن ({rsi:.1f})'
            
            # MACD
            if latest['MACD'] > latest['MACD_Signal']:
                score += 1
                signals['macd'] = 'إيجابي'
            else:
                score -= 1
                signals['macd'] = 'سلبي'
            
            # Volume
            if latest['Volume_Ratio'] > 1.5:
                score += 0.5
                signals['volume'] = 'قوي'
            
            # الارتباطات
            if 'dxy' in correlations and correlations['dxy'] < -0.5:
                score += 1
                signals['dxy'] = 'ارتباط عكسي قوي'
            
            # التعلم الآلي
            if ml_result['prediction'] == 'صعود':
                score += ml_result['confidence']
                signals['ml'] = f"توقع {ml_result['prediction']} ({ml_result['confidence']:.2%})"
            else:
                score -= ml_result['confidence']
                signals['ml'] = f"توقع {ml_result['prediction']} ({ml_result['confidence']:.2%})"
            
            # الأخبار
            news_sentiment = news_analysis.get('sentiment', 0)
            score += news_sentiment
            signals['news'] = news_analysis.get('impact', 'محايد')
            
            # الإشارة النهائية
            if score >= 2:
                final_signal = "Strong Buy"
                confidence = "High"
                recommendation = "شراء قوي"
            elif score >= 1:
                final_signal = "Buy"
                confidence = "Medium"
                recommendation = "شراء"
            elif score <= -2:
                final_signal = "Strong Sell"
                confidence = "High"
                recommendation = "بيع قوي"
            elif score <= -1:
                final_signal = "Sell"
                confidence = "Medium"
                recommendation = "بيع"
            else:
                final_signal = "Hold"
                confidence = "Low"
                recommendation = "انتظار"
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'recommendation': recommendation,
                'score': round(score, 2),
                'current_price': round(latest['Close'], 2),
                'signals_breakdown': signals,
                'key_levels': {
                    'resistance': round(latest['Resistance'], 2),
                    'support': round(latest['Support'], 2),
                    'rsi': round(latest['RSI'], 1)
                }
            }
            
        except Exception as e:
            print(f"❌ خطأ في الإشارات: {e}")
            return {'error': str(e)}
    
    def generate_report(self, analysis_result):
        """توليد التقرير"""
        try:
            report = []
            report.append("=" * 70)
            report.append("📊 تقرير تحليل الذهب المحسن لـ GitHub")
            report.append("=" * 70)
            report.append(f"📅 التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"✅ الحالة: {analysis_result.get('status', 'غير محدد')}")
            report.append("")
            
            if 'analysis' in analysis_result and 'error' not in analysis_result['analysis']:
                analysis = analysis_result['analysis']
                
                report.append("🎯 النتائج الرئيسية:")
                report.append(f"  • الإشارة: {analysis.get('signal', 'N/A')}")
                report.append(f"  • الثقة: {analysis.get('confidence', 'N/A')}")
                report.append(f"  • التوصية: {analysis.get('recommendation', 'N/A')}")
                report.append(f"  • السعر: ${analysis.get('current_price', 'N/A')}")
                report.append(f"  • النقاط: {analysis.get('score', 'N/A')}")
                report.append("")
                
                if 'key_levels' in analysis:
                    kl = analysis['key_levels']
                    report.append("📈 المستويات الفنية:")
                    report.append(f"  • الدعم: ${kl.get('support', 'N/A')}")
                    report.append(f"  • المقاومة: ${kl.get('resistance', 'N/A')}")
                    report.append(f"  • RSI: {kl.get('rsi', 'N/A')}")
                    report.append("")
                
                if 'signals_breakdown' in analysis:
                    sb = analysis['signals_breakdown']
                    report.append("🔍 تفاصيل الإشارات:")
                    for signal_name, signal_value in sb.items():
                        report.append(f"  • {signal_name}: {signal_value}")
                    report.append("")
            
            if 'correlations' in analysis_result:
                correlations = analysis_result['correlations']
                if correlations:
                    report.append("🔗 الارتباطات:")
                    for asset, corr in correlations.items():
                        report.append(f"  • {asset}: {corr}")
                    report.append("")
            
            if 'news_analysis' in analysis_result:
                na = analysis_result['news_analysis']
                report.append("📰 تحليل الأخبار:")
                report.append(f"  • الحالة: {na.get('status', 'N/A')}")
                report.append(f"  • التأثير: {na.get('impact', 'N/A')}")
                if 'articles_count' in na:
                    report.append(f"  • عدد المقالات: {na.get('articles_count', 0)}")
                report.append("")
            
            if 'ml_result' in analysis_result:
                ml = analysis_result['ml_result']
                report.append("🤖 التعلم الآلي:")
                report.append(f"  • التنبؤ: {ml.get('prediction', 'N/A')}")
                report.append(f"  • الثقة: {ml.get('confidence', 'N/A')}")
                report.append(f"  • الطريقة: {ml.get('method', 'N/A')}")
                report.append("")
            
            report.append("=" * 70)
            report.append("⚠️ تنبيه: هذا تحليل تعليمي وليس نصيحة استثمارية")
            report.append("=" * 70)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"خطأ في التقرير: {e}"
    
    def save_results(self, results):
        """حفظ النتائج"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # الملف الرئيسي
            main_file = f"gold_analysis_v3_{timestamp}.json"
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل: {main_file}")
            
            # الملخص
            summary_file = f"gold_summary_{timestamp}.json"
            summary = {
                'timestamp': results.get('timestamp'),
                'status': results.get('status'),
                'signal': results.get('analysis', {}).get('signal'),
                'confidence': results.get('analysis', {}).get('confidence'),
                'price': results.get('analysis', {}).get('current_price'),
                'recommendation': results.get('analysis', {}).get('recommendation')
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"📋 تم حفظ الملخص: {summary_file}")
            
            # التقرير النصي
            report_file = f"gold_report_{timestamp}.txt"
            report = self.generate_report(results)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 تم حفظ التقرير: {report_file}")
            
        except Exception as e:
            print(f"❌ خطأ في الحفظ: {e}")
    
    def run_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء التحليل المحسن...")
        print("=" * 70)
        
        try:
            # 1. جلب البيانات
            market_data = self.fetch_market_data()
            if not market_data:
                raise ValueError("فشل في جلب بيانات السوق")
            
            # استخدام أول بيانات ذهب متاحة
            gold_data = None
            for key in ['gold', 'gold_etf']:
                if key in market_data:
                    gold_data = market_data[key]
                    print(f"📊 استخدام بيانات {key}")
                    break
            
            if gold_data is None:
                raise ValueError("لا توجد بيانات ذهب متاحة")
            
            # 2. المؤشرات الفنية
            technical_data = self.calculate_indicators(gold_data)
            
            # 3. الارتباطات
            correlations = self.calculate_correlations(market_data)
            
            # 4. تحليل الأخبار
            news_analysis = self.analyze_news_simple()
            
            # 5. التعلم الآلي
            ml_result = self.simple_ml_prediction(technical_data)
            
            # 6. الإشارات النهائية
            analysis = self.generate_signals(technical_data, correlations, ml_result, news_analysis)
            
            # النتائج النهائية
            final_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'version': 'github_optimized',
                'analysis': analysis,
                'correlations': correlations,
                'news_analysis': news_analysis,
                'ml_result': ml_result,
                'data_points': len(technical_data),
                'period_analyzed': '6 months'
            }
            
            # طباعة النتائج
            print("\n" + "=" * 70)
            print("📊 النتائج:")
            if 'error' not in analysis:
                print(f"🎯 الإشارة: {analysis.get('signal')}")
                print(f"💪 الثقة: {analysis.get('confidence')}")
                print(f"💰 السعر: ${analysis.get('current_price')}")
                print(f"📊 النقاط: {analysis.get('score')}")
                print(f"📝 التوصية: {analysis.get('recommendation')}")
            
            # طباعة التقرير
            report = self.generate_report(final_results)
            print("\n" + report)
            
            # حفظ النتائج
            self.save_results(final_results)
            
            print("\n✅ تم إنجاز التحليل بنجاح!")
            return final_results
            
        except Exception as e:
            error_msg = f"❌ فشل التحليل: {e}"
            print(error_msg)
            
            error_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'version': 'github_optimized'
            }
            
            self.save_results(error_results)
            return error_results

def main():
    """الدالة الرئيسية"""
    print("🏆 محلل الذهب المحسن لـ GitHub Actions")
    print("=" * 70)
    
    try:
        analyzer = GitHubGoldAnalyzer()
        result = analyzer.run_analysis()
        
        if result.get('status') == 'success':
            print("\n🎉 التحليل اكتمل بنجاح!")
        else:
            print(f"\n❌ التحليل فشل: {result.get('error', 'خطأ غير معروف')}")
            
    except Exception as e:
        print(f"\n💥 خطأ عام: {e}")

if __name__ == "__main__":
    main()
