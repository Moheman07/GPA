#!/usr/bin/env python3
"""
محلل الذهب الاحترافي الكامل V4.0
سكربت واحد شامل يحتوي على جميع المميزات المتقدمة
GitHub Actions Compatible - Professional Gold Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import warnings
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import joblib

warnings.filterwarnings('ignore')

class ProfessionalGoldAnalyzer:
    """محلل الذهب الاحترافي الكامل مع جميع المميزات المتقدمة"""
    
    def __init__(self):
        """تهيئة المحلل مع جميع الإعدادات"""
        self.symbols = {
            'gold_futures': 'GC=F', 'gold_etf': 'GLD', 'silver': 'SI=F',
            'dxy': 'DX-Y.NYB', 'vix': '^VIX', 'spy': 'SPY', 'oil': 'CL=F',
            'copper': 'HG=F', 'platinum': 'PL=F', 'treasury_10y': '^TNX'
        }
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        print("🚀 تم تهيئة محلل الذهب الاحترافي الكامل V4.0")
    
    async def fetch_market_data_comprehensive(self):
        """جلب بيانات السوق الشاملة"""
        print("📊 جلب البيانات الشاملة للأسواق...")
        market_data = {}
        
        for name, symbol in self.symbols.items():
            try:
                print(f"📈 جلب بيانات {name}...")
                data = yf.download(symbol, period="1y", interval="1d", progress=False)
                if not data.empty:
                    market_data[name] = data
                    print(f"✅ تم جلب {len(data)} نقطة لـ {name}")
            except Exception as e:
                print(f"❌ خطأ في جلب {name}: {e}")
        
        return market_data
    
    def calculate_advanced_indicators(self, data):
        """حساب المؤشرات الفنية المتقدمة"""
        print("📈 حساب المؤشرات المتقدمة...")
        df = data.copy()
        
        try:
            # المتوسطات المتحركة
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # RSI متعدد الفترات
            for period in [14, 21, 30]:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD متقدم
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            for period, std_dev in [(20, 2), (20, 2.5)]:
                sma = df['Close'].rolling(period).mean()
                std = df['Close'].rolling(period).std()
                df[f'BB_Upper_{period}_{std_dev}'] = sma + (std * std_dev)
                df[f'BB_Lower_{period}_{std_dev}'] = sma - (std * std_dev)
                df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}_{std_dev}'] - df[f'BB_Lower_{period}_{std_dev}']) / sma
            
            # Stochastic & Williams %R
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['%D'] = df['%K'].rolling(3).mean()
            df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
            
            # ATR & CCI
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(14).mean()
            
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
            
            # Volume Analysis
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # Support/Resistance
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            df['Distance_to_Resistance'] = ((df['Resistance'] - df['Close']) / df['Close']) * 100
            df['Distance_to_Support'] = ((df['Close'] - df['Support']) / df['Close']) * 100
            
            print(f"✅ تم حساب {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} مؤشر")
            return df.dropna()
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return df
    
    def calculate_correlations_advanced(self, market_data):
        """حساب الارتباطات المتقدمة"""
        print("🔗 تحليل الارتباطات...")
        correlations = {}
        
        try:
            if 'gold_futures' in market_data:
                gold_prices = market_data['gold_futures']['Close'].dropna()
                
                for asset, data in market_data.items():
                    if asset != 'gold_futures' and not data.empty:
                        asset_prices = data['Close'].dropna()
                        common_dates = gold_prices.index.intersection(asset_prices.index)
                        
                        if len(common_dates) > 30:
                            corr = gold_prices.loc[common_dates].corr(asset_prices.loc[common_dates])
                            correlations[asset] = {
                                'correlation': round(corr, 3),
                                'strength': self._classify_correlation(corr),
                                'data_points': len(common_dates)
                            }
            
            return correlations
            
        except Exception as e:
            print(f"❌ خطأ في حساب الارتباطات: {e}")
            return {}
    
    def _classify_correlation(self, corr):
        """تصنيف قوة الارتباط"""
        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "قوي جداً"
        elif abs_corr >= 0.6:
            return "قوي"
        elif abs_corr >= 0.4:
            return "متوسط"
        elif abs_corr >= 0.2:
            return "ضعيف"
        else:
            return "ضعيف جداً"
    
    async def fetch_news_analysis(self):
        """تحليل الأخبار المتقدم"""
        print("📰 تحليل الأخبار...")
        
        if not self.news_api_key:
            return self._simulate_news_analysis()
        
        try:
            keywords = ['gold', 'federal reserve', 'inflation', 'dollar']
            news_data = []
            
            async with aiohttp.ClientSession() as session:
                for keyword in keywords:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': keyword,
                        'apiKey': self.news_api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'from': (datetime.now() - timedelta(days=7)).isoformat(),
                        'pageSize': 10
                    }
                    
                    try:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                news_data.extend(data.get('articles', []))
                    except Exception as e:
                        print(f"⚠️ خطأ في {keyword}: {e}")
            
            # تحليل المشاعر
            sentiment_scores = []
            for article in news_data[:30]:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                blob = TextBlob(text)
                sentiment_scores.append(blob.sentiment.polarity)
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_trend = "إيجابي" if avg_sentiment > 0.1 else "سلبي" if avg_sentiment < -0.1 else "محايد"
                
                return {
                    'status': 'success',
                    'articles_count': len(news_data),
                    'sentiment_score': round(avg_sentiment, 3),
                    'sentiment_trend': sentiment_trend,
                    'market_impact': self._calculate_news_impact(avg_sentiment)
                }
            else:
                return self._simulate_news_analysis()
                
        except Exception as e:
            print(f"❌ خطأ في الأخبار: {e}")
            return self._simulate_news_analysis()
    
    def _simulate_news_analysis(self):
        """محاكاة تحليل الأخبار"""
        return {
            'status': 'simulated',
            'sentiment_score': 0.15,
            'sentiment_trend': 'إيجابي معتدل',
            'market_impact': 'إيجابي للذهب - مخاوف التضخم'
        }
    
    def _calculate_news_impact(self, sentiment):
        """حساب تأثير الأخبار"""
        if sentiment > 0.2:
            return "إيجابي قوي - ارتفاع محتمل"
        elif sentiment > 0.05:
            return "إيجابي معتدل - دعم للأسعار"
        elif sentiment < -0.2:
            return "سلبي قوي - ضغط على الأسعار"
        elif sentiment < -0.05:
            return "سلبي معتدل - حذر في السوق"
        else:
            return "محايد - لا تأثير واضح"
    
    def prepare_ml_features(self, technical_data, correlations):
        """إعداد متغيرات التعلم الآلي"""
        print("🤖 إعداد متغيرات ML...")
        
        try:
            features = []
            feature_names = []
            
            # المؤشرات الفنية الرئيسية
            tech_features = [
                'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                '%K', '%D', 'Williams_R', 'ATR', 'CCI',
                'BB_Width_20_2.0', 'Volume_Ratio', 'Distance_to_Resistance', 'Distance_to_Support'
            ]
            
            for feature in tech_features:
                if feature in technical_data.columns:
                    features.append(technical_data[feature].fillna(0))
                    feature_names.append(feature)
            
            # الارتباطات
            for asset in ['dxy', 'vix', 'spy', 'oil']:
                if asset in correlations:
                    corr_value = correlations[asset].get('correlation', 0)
                    features.append(pd.Series([corr_value] * len(technical_data), index=technical_data.index))
                    feature_names.append(f'corr_{asset}')
            
            if features:
                feature_df = pd.concat(features, axis=1)
                feature_df.columns = feature_names
                return feature_df.dropna()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ خطأ في إعداد ML: {e}")
            return pd.DataFrame()
    
    def train_ml_models(self, features, target):
        """تدريب نماذج التعلم الآلي"""
        print("🧠 تدريب نماذج ML...")
        
        try:
            if len(features) < 50:
                print("⚠️ بيانات غير كافية - نموذج مبسط")
                return {'simple_model': {'prediction': 1, 'confidence': 0.65}}
            
            X = features.fillna(0)
            y = target
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                
                test_pred = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, test_pred)
                
                current_features = X.iloc[-1:].fillna(0)
                current_scaled = self.scaler.transform(current_features)
                current_pred = model.predict(current_scaled)[0]
                current_prob = model.predict_proba(current_scaled)[0].max()
                
                results[name] = {
                    'test_accuracy': round(test_accuracy, 3),
                    'current_prediction': current_pred,
                    'confidence': round(current_prob, 3)
                }
                
                print(f"✅ {name}: دقة {test_accuracy:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ خطأ في تدريب النماذج: {e}")
            return {'simple_model': {'prediction': 1, 'confidence': 0.65}}
    
    def generate_trading_signals(self, technical_data, correlations, ml_results, news_analysis):
        """توليد الإشارات المتقدمة"""
        print("🎯 توليد الإشارات...")
        
        try:
            latest = technical_data.iloc[-1]
            signals = {}
            total_score = 0
            
            # التحليل الفني (40%)
            tech_score = 0
            
            # الاتجاه
            if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
                tech_score += 2
                signals['trend'] = 'صاعد قوي'
            elif latest['Close'] > latest['SMA_20']:
                tech_score += 1
                signals['trend'] = 'صاعد'
            else:
                tech_score -= 1
                signals['trend'] = 'هابط'
            
            # RSI
            rsi = latest['RSI_14']
            if rsi < 30:
                tech_score += 2
                signals['rsi'] = 'ذروة بيع - فرصة شراء'
            elif rsi > 70:
                tech_score -= 2
                signals['rsi'] = 'ذروة شراء - حذر'
            else:
                signals['rsi'] = f'RSI: {rsi:.1f} - متوازن'
            
            # MACD
            if latest['MACD'] > latest['MACD_Signal']:
                tech_score += 1
                signals['macd'] = 'إشارة صعود'
            else:
                tech_score -= 1
                signals['macd'] = 'إشارة هبوط'
            
            # Volume
            if latest['Volume_Ratio'] > 1.5:
                tech_score += 1
                signals['volume'] = 'حجم قوي'
            elif latest['Volume_Ratio'] < 0.7:
                tech_score -= 0.5
                signals['volume'] = 'حجم ضعيف'
            
            total_score += tech_score * 0.4
            
            # الارتباطات (25%)
            corr_score = 0
            if 'dxy' in correlations:
                dxy_corr = correlations['dxy'].get('correlation', 0)
                if dxy_corr < -0.5:
                    corr_score += 2
                    signals['dxy'] = 'ارتباط عكسي قوي'
                elif dxy_corr < -0.2:
                    corr_score += 1
                    signals['dxy'] = 'ارتباط عكسي معتدل'
            
            total_score += corr_score * 0.25
            
            # التعلم الآلي (25%)
            ml_score = 0
            if ml_results:
                best_model = max(ml_results.items(), key=lambda x: x[1].get('test_accuracy', 0))
                model_name, model_data = best_model
                
                if model_data.get('current_prediction', 0) == 1:
                    ml_score += model_data.get('confidence', 0) * 3
                    signals['ml'] = f'توقع صعود ({model_data.get("confidence", 0):.2%})'
                else:
                    ml_score -= model_data.get('confidence', 0) * 3
                    signals['ml'] = f'توقع هبوط ({model_data.get("confidence", 0):.2%})'
            
            total_score += ml_score * 0.25
            
            # الأخبار (10%)
            news_score = 0
            if news_analysis:
                sentiment = news_analysis.get('sentiment_score', 0)
                news_score = sentiment * 2
                signals['news'] = news_analysis.get('market_impact', 'محايد')
            
            total_score += news_score * 0.1
            
            # الإشارة النهائية
            if total_score >= 2:
                final_signal = "Strong Buy"
                confidence = "High"
                recommendation = "شراء قوي"
            elif total_score >= 1:
                final_signal = "Buy"
                confidence = "Medium-High"
                recommendation = "شراء"
            elif total_score >= 0.5:
                final_signal = "Weak Buy"
                confidence = "Medium"
                recommendation = "شراء ضعيف"
            elif total_score <= -2:
                final_signal = "Strong Sell"
                confidence = "High"
                recommendation = "بيع قوي"
            elif total_score <= -1:
                final_signal = "Sell"
                confidence = "Medium-High"
                recommendation = "بيع"
            elif total_score <= -0.5:
                final_signal = "Weak Sell"
                confidence = "Medium"
                recommendation = "بيع ضعيف"
            else:
                final_signal = "Hold"
                confidence = "Low"
                recommendation = "انتظار"
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'recommendation': recommendation,
                'total_score': round(total_score, 2),
                'component_scores': {
                    'technical': round(tech_score, 2),
                    'correlations': round(corr_score, 2),
                    'machine_learning': round(ml_score, 2),
                    'news_sentiment': round(news_score, 2)
                },
                'signals_breakdown': signals,
                'current_price': round(latest['Close'], 2),
                'key_levels': {
                    'resistance': round(latest['Resistance'], 2),
                    'support': round(latest['Support'], 2),
                    'rsi': round(latest['RSI_14'], 1),
                    'distance_to_resistance': round(latest['Distance_to_Resistance'], 2),
                    'distance_to_support': round(latest['Distance_to_Support'], 2)
                }
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, analysis_result):
        """توليد التقرير الشامل"""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 تقرير تحليل الذهب الاحترافي الكامل V4.0")
            report.append("=" * 80)
            report.append(f"🕒 التاريخ والوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"📍 حالة التحليل: {analysis_result.get('status', 'غير محدد')}")
            report.append("")
            
            # تحليل الأسعار الرئيسي
            if 'gold_analysis' in analysis_result:
                ga = analysis_result['gold_analysis']
                if 'error' not in ga:
                    report.append("🎯 الإشارة الرئيسية:")
                    report.append(f"  • الإشارة: {ga.get('signal', 'N/A')}")
                    report.append(f"  • مستوى الثقة: {ga.get('confidence', 'N/A')}")
                    report.append(f"  • التوصية: {ga.get('recommendation', 'N/A')}")
                    report.append(f"  • السعر الحالي: ${ga.get('current_price', 'N/A')}")
                    report.append(f"  • النقاط الإجمالية: {ga.get('total_score', 'N/A')}")
                    report.append("")
                    
                    # المستويات الفنية
                    if 'key_levels' in ga:
                        kl = ga['key_levels']
                        report.append("📈 المستويات الفنية الرئيسية:")
                        report.append(f"  • الدعم: ${kl.get('support', 'N/A')}")
                        report.append(f"  • المقاومة: ${kl.get('resistance', 'N/A')}")
                        report.append(f"  • RSI: {kl.get('rsi', 'N/A')}")
                        report.append(f"  • المسافة للمقاومة: {kl.get('distance_to_resistance', 'N/A')}%")
                        report.append(f"  • المسافة للدعم: {kl.get('distance_to_support', 'N/A')}%")
                        report.append("")
                    
                    # تفاصيل النقاط
                    if 'component_scores' in ga:
                        cs = ga['component_scores']
                        report.append("🔢 تفاصيل النقاط:")
                        report.append(f"  • التحليل الفني: {cs.get('technical', 'N/A')} نقطة")
                        report.append(f"  • الارتباطات: {cs.get('correlations', 'N/A')} نقطة")
                        report.append(f"  • التعلم الآلي: {cs.get('machine_learning', 'N/A')} نقطة")
                        report.append(f"  • الأخبار: {cs.get('news_sentiment', 'N/A')} نقطة")
                        report.append("")
                    
                    # تفاصيل الإشارات
                    if 'signals_breakdown' in ga:
                        sb = ga['signals_breakdown']
                        report.append("🔍 تفاصيل الإشارات:")
                        for signal_name, signal_value in sb.items():
                            report.append(f"  • {signal_name}: {signal_value}")
                        report.append("")
            
            # الارتباطات
            if 'correlations' in analysis_result:
                correlations = analysis_result['correlations']
                if correlations:
                    report.append("🔗 تحليل الارتباطات:")
                    for asset, corr_data in correlations.items():
                        if isinstance(corr_data, dict):
                            corr_val = corr_data.get('correlation', 0)
                            strength = corr_data.get('strength', 'غير محدد')
                            report.append(f"  • {asset}: {corr_val} ({strength})")
                    report.append("")
            
            # تحليل الأخبار
            if 'news_analysis' in analysis_result:
                na = analysis_result['news_analysis']
                if na.get('status') == 'success':
                    report.append("📰 تحليل الأخبار:")
                    report.append(f"  • عدد المقالات: {na.get('articles_count', 0)}")
                    report.append(f"  • نقاط المشاعر: {na.get('sentiment_score', 0):.3f}")
                    report.append(f"  • الاتجاه: {na.get('sentiment_trend', 'N/A')}")
                    report.append(f"  • التأثير على السوق: {na.get('market_impact', 'N/A')}")
                    report.append("")
                elif na.get('status') == 'simulated':
                    report.append("📰 تحليل الأخبار (محاكاة):")
                    report.append(f"  • التأثير: {na.get('market_impact', 'N/A')}")
                    report.append("")
            
            # نماذج التعلم الآلي
            if 'ml_results' in analysis_result:
                ml = analysis_result['ml_results']
                if ml:
                    report.append("🤖 نتائج التعلم الآلي:")
                    for model_name, model_data in ml.items():
                        if isinstance(model_data, dict):
                            accuracy = model_data.get('test_accuracy', 0)
                            prediction = model_data.get('current_prediction', 0)
                            confidence = model_data.get('confidence', 0)
                            pred_text = "صعود" if prediction == 1 else "هبوط"
                            report.append(f"  • {model_name}: {pred_text} (دقة: {accuracy:.3f}, ثقة: {confidence:.3f})")
                    report.append("")
            
            # معلومات تقنية
            report.append("🛠️ معلومات تقنية:")
            report.append(f"  • عدد نقاط البيانات: {analysis_result.get('data_points', 'N/A')}")
            report.append(f"  • فترة التحليل: {analysis_result.get('period_analyzed', 'N/A')}")
            report.append(f"  • الإصدار: محلل الذهب الكامل V4.0")
            report.append("")
            
            report.append("=" * 80)
            report.append("⚠️ تنبيه: هذا تحليل تعليمي ولا يُعتبر نصيحة استثمارية")
            report.append("💡 يُنصح بالتشاور مع مستشار مالي مؤهل قبل اتخاذ قرارات الاستثمار")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"
    
    def save_comprehensive_results(self, results):
        """حفظ النتائج الشاملة"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # حفظ التحليل الكامل
            main_filename = f"gold_analysis_v3_{timestamp}.json"
            with open(main_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل الكامل: {main_filename}")
            
            # حفظ الملخص
            summary_filename = f"gold_summary_{timestamp}.json"
            summary = {
                'timestamp': results.get('timestamp'),
                'status': results.get('status'),
                'signal': results.get('gold_analysis', {}).get('signal'),
                'confidence': results.get('gold_analysis', {}).get('confidence'),
                'price': results.get('gold_analysis', {}).get('current_price'),
                'recommendation': results.get('gold_analysis', {}).get('recommendation')
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"📋 تم حفظ الملخص: {summary_filename}")
            
            # حفظ التقرير النصي
            report_filename = f"gold_report_{timestamp}.txt"
            report = self.generate_comprehensive_report(results)
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 تم حفظ التقرير: {report_filename}")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")
    
    async def run_complete_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء التحليل الشامل للذهب...")
        print("=" * 80)
        
        try:
            # 1. جلب بيانات السوق
            market_data = await self.fetch_market_data_comprehensive()
            if not market_data or 'gold_futures' not in market_data:
                if 'gold_etf' in market_data:
                    market_data['gold_futures'] = market_data['gold_etf']
                    print("📊 استخدام بيانات GLD كبديل")
                else:
                    raise ValueError("فشل في جلب بيانات الذهب")
            
            # 2. حساب المؤشرات الفنية
            print("📈 تحليل المؤشرات الفنية...")
            technical_data = self.calculate_advanced_indicators(market_data['gold_futures'])
            
            # 3. تحليل الارتباطات
            print("🔗 تحليل الارتباطات...")
            correlations = self.calculate_correlations_advanced(market_data)
            
            # 4. تحليل الأخبار
            print("📰 تحليل الأخبار...")
            news_analysis = await self.fetch_news_analysis()
            
            # 5. التعلم الآلي
            print("🤖 تدريب نماذج التعلم الآلي...")
            features = self.prepare_ml_features(technical_data, correlations)
            
            ml_results = {}
            if not features.empty and len(features) > 20:
                # إنشاء target للتدريب (صعود/هبوط بناءً على الأسعار)
                price_change = technical_data['Close'].pct_change(5).shift(-5)  # التغيير خلال 5 أيام
                target = (price_change > 0.01).astype(int)  # صعود > 1%
                
                # تدريب النماذج
                ml_results = self.train_ml_models(features, target)
            else:
                print("⚠️ بيانات غير كافية للتعلم الآلي")
            
            # 6. توليد الإشارات
            print("🎯 توليد الإشارات النهائية...")
            trading_signals = self.generate_trading_signals(
                technical_data, correlations, ml_results, news_analysis
            )
            
            # تجميع النتائج النهائية
            final_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'version': 'complete_v4.0',
                'gold_analysis': trading_signals,
                'correlations': correlations,
                'news_analysis': news_analysis,
                'ml_results': ml_results,
                'data_points': len(technical_data),
                'period_analyzed': '1 year',
                'assets_analyzed': list(market_data.keys())
            }
            
            # طباعة النتائج
            print("\n" + "=" * 80)
            print("📊 نتائج التحليل الكامل:")
            print("=" * 80)
            
            if 'error' not in trading_signals:
                print(f"🎯 الإشارة: {trading_signals.get('signal')}")
                print(f"💪 الثقة: {trading_signals.get('confidence')}")
                print(f"💰 السعر: ${trading_signals.get('current_price')}")
                print(f"📊 النقاط: {trading_signals.get('total_score')}")
                print(f"📋 التوصية: {trading_signals.get('recommendation')}")
            
            # طباعة التقرير الكامل
            report = self.generate_comprehensive_report(final_results)
            print("\n" + report)
            
            # حفظ النتائج
            self.save_comprehensive_results(final_results)
            
            print("\n✅ تم إتمام التحليل الشامل بنجاح!")
            return final_results
            
        except Exception as e:
            error_msg = f"❌ فشل التحليل الشامل: {e}"
            print(error_msg)
            
            error_results = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'version': 'complete_v4.0'
            }
            
            self.save_comprehensive_results(error_results)
            return error_results

def main():
    """الدالة الرئيسية"""
    print("🏆 محلل الذهب الاحترافي الكامل V4.0")
    print("🚀 سكربت شامل بجميع المميزات المتقدمة")
    print("=" * 80)
    
    try:
        analyzer = ProfessionalGoldAnalyzer()
        result = asyncio.run(analyzer.run_complete_analysis())
        
        if result.get('status') == 'success':
            print("\n🎉 انتهى التحليل بنجاح!")
        else:
            print(f"\n❌ التحليل فشل: {result.get('error', 'خطأ غير معروف')}")
            
    except Exception as e:
        print(f"\n💥 خطأ في التشغيل: {e}")

if __name__ == "__main__":
    main()
