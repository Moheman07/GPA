import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import json
from datetime import datetime, timedelta
import pytz
import os
import warnings
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings('ignore')

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalGoldAnalyzer:
    def __init__(self):
        self.setup_config()
        self.ensure_directories()
        
    def setup_config(self):
        """إعداد المتغيرات والرموز"""
        self.SYMBOLS = {
            'gold_futures': 'GC=F',
            'gold_spot': 'XAUUSD=X', 
            'gold_etf': 'GLD',
            'silver': 'SI=F',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'tnx': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY',
            'eur_usd': 'EURUSD=X',
            'jpy_usd': 'JPYUSD=X',
            'btc': 'BTC-USD'
        }
        
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        
        self.GOLD_KEYWORDS = [
            "gold", "XAU", "federal reserve", "Fed", "inflation", 
            "interest rate", "dollar", "safe haven", "precious metals",
            "monetary policy", "central bank", "NFP", "CPI"
        ]
        
    def ensure_directories(self):
        """إنشاء المجلدات المطلوبة"""
        for directory in ['results', 'logs', 'data']:
            os.makedirs(directory, exist_ok=True)

    def fetch_market_data(self) -> Optional[Dict]:
        """جلب بيانات السوق"""
        logger.info("🔄 جاري جلب بيانات السوق...")
        
        try:
            symbols = list(self.SYMBOLS.values())
            
            # جلب بيانات متعددة الفترات
            data_daily = yf.download(symbols, period="1y", interval="1d", group_by='ticker')
            data_hourly = yf.download([self.SYMBOLS['gold_futures']], period="30d", interval="1h")
            
            if data_daily.empty:
                raise ValueError("فشل في جلب البيانات اليومية")
            
            logger.info("✅ تم جلب بيانات السوق بنجاح")
            return {
                'daily': data_daily,
                'hourly': data_hourly,
                'last_update': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def calculate_technical_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """حساب المؤشرات الفنية المتقدمة"""
        try:
            if len(data.columns.levels) > 1:
                df = data[symbol].copy()
            else:
                df = data.copy()
            
            # التأكد من وجود الأعمدة المطلوبة
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"عمود {col} غير موجود")
                    return df
            
            # المؤشرات الأساسية
            df.ta.sma(length=[20, 50, 200], append=True)
            df.ta.ema(length=[12, 26], append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(append=True)
            df.ta.atr(append=True)
            df.ta.obv(append=True)
            df.ta.adx(append=True)
            
            # مؤشرات مخصصة للذهب
            df['Gold_Momentum'] = df['Close'].pct_change(10) * 100
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
            
            # مؤشرات الاتجاه
            df['Trend_Strength'] = np.where(df['Close'] > df['SMA_200'], 1, 
                                  np.where(df['Close'] < df['SMA_200'], -1, 0))
            
            # مستويات الدعم والمقاومة
            df['Support'] = df['Low'].rolling(window=20, center=True).min()
            df['Resistance'] = df['High'].rolling(window=20, center=True).max()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب المؤشرات: {e}")
            return data[symbol] if len(data.columns.levels) > 1 else data

    def analyze_market_correlations(self, data: Dict) -> Dict:
        """تحليل الارتباطات مع الأصول الأخرى"""
        logger.info("📊 تحليل الارتباطات...")
        
        try:
            daily_data = data['daily']
            
            # استخراج أسعار الإغلاق
            prices = {}
            for asset, symbol in self.SYMBOLS.items():
                try:
                    if len(daily_data.columns.levels) > 1 and symbol in daily_data.columns.levels[0]:
                        prices[asset] = daily_data[symbol]['Close'].dropna()
                    elif 'Close' in daily_data.columns:
                        prices[asset] = daily_data['Close'].dropna()
                except:
                    continue
            
            # حساب الارتباطات
            correlations = {}
            if 'gold_futures' in prices:
                gold_prices = prices['gold_futures']
                for asset, asset_prices in prices.items():
                    if asset != 'gold_futures' and len(asset_prices) > 50:
                        try:
                            # محاذاة البيانات
                            common_dates = gold_prices.index.intersection(asset_prices.index)
                            if len(common_dates) > 30:
                                corr = gold_prices.loc[common_dates].corr(asset_prices.loc[common_dates])
                                correlations[asset] = round(corr, 3)
                        except:
                            continue
            
            # تصنيف الارتباطات
            strong_negative = {k: v for k, v in correlations.items() if v < -0.5}
            strong_positive = {k: v for k, v in correlations.items() if v > 0.5}
            moderate = {k: v for k, v in correlations.items() if -0.5 <= v <= 0.5}
            
            return {
                'all_correlations': correlations,
                'strong_negative': strong_negative,
                'strong_positive': strong_positive,
                'moderate': moderate,
                'interpretation': self.interpret_correlations(correlations)
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {}

    def interpret_correlations(self, correlations: Dict) -> Dict:
        """تفسير الارتباطات"""
        interpretations = {}
        
        # تفسير الارتباط مع الدولار
        dxy_corr = correlations.get('dxy', 0)
        if dxy_corr < -0.5:
            interpretations['dxy'] = "ارتباط سلبي قوي - تعزز الدولار يضر الذهب"
        elif dxy_corr > 0.3:
            interpretations['dxy'] = "ارتباط إيجابي غير معتاد - قد يشير لعوامل جيوسياسية"
        else:
            interpretations['dxy'] = "ارتباط معتدل"
        
        # تفسير VIX
        vix_corr = correlations.get('vix', 0)
        if vix_corr > 0.3:
            interpretations['vix'] = "الذهب يعمل كملاذ آمن في أوقات الخوف"
        else:
            interpretations['vix'] = "علاقة ضعيفة مع مؤشر الخوف"
        
        return interpretations

    def fetch_gold_news(self) -> Dict:
        """جلب وتحليل الأخبار المتعلقة بالذهب"""
        logger.info("📰 جلب وتحليل الأخبار...")
        
        if not self.NEWS_API_KEY:
            return {"error": "مفتاح API للأخبار غير متوفر"}
        
        try:
            # بناء استعلام البحث
            keywords = " OR ".join([f'"{keyword}"' for keyword in self.GOLD_KEYWORDS[:5]])
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={keywords}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"pageSize=50&"
                f"from={datetime.now().date() - timedelta(days=3)}&"
                f"apiKey={self.NEWS_API_KEY}"
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            
            # تصفية الأخبار
            filtered_articles = self.filter_relevant_news(articles)
            
            # تحليل المشاعر
            sentiment_analysis = self.analyze_news_sentiment(filtered_articles)
            
            return {
                'total_articles': len(articles),
                'relevant_articles': len(filtered_articles),
                'sentiment_analysis': sentiment_analysis,
                'key_headlines': filtered_articles[:5],
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الأخبار: {e}")
            return {"error": str(e)}

    def filter_relevant_news(self, articles: List) -> List:
        """تصفية الأخبار ذات الصلة بالذهب"""
        relevant_articles = []
        
        for article in articles:
            title = (article.get('title', '') or '').lower()
            description = (article.get('description', '') or '').lower()
            content = f"{title} {description}"
            
            # حساب درجة الصلة
            relevance_score = 0
            for keyword in self.GOLD_KEYWORDS:
                if keyword.lower() in content:
                    if keyword.lower() in ['gold', 'xau']:
                        relevance_score += 3
                    elif keyword.lower() in ['federal reserve', 'fed', 'inflation']:
                        relevance_score += 2
                    else:
                        relevance_score += 1
            
            if relevance_score >= 2:
                article['relevance_score'] = relevance_score
                relevant_articles.append(article)
        
        return sorted(relevant_articles, key=lambda x: x['relevance_score'], reverse=True)

    def analyze_news_sentiment(self, articles: List) -> Dict:
        """تحليل مشاعر الأخبار"""
        if not articles:
            return {'overall_sentiment': 'محايد', 'confidence': 0}
        
        try:
            from textblob import TextBlob
            
            sentiment_scores = []
            analyzed_articles = []
            
            for article in articles[:10]:  # تحليل أفضل 10 مقالات
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title}. {description}"
                
                if text.strip():
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity
                    
                    sentiment_scores.append(sentiment)
                    
                    # تحديد التأثير على الذهب
                    if sentiment > 0.1:
                        gold_impact = "إيجابي"
                    elif sentiment < -0.1:
                        gold_impact = "سلبي"
                    else:
                        gold_impact = "محايد"
                    
                    analyzed_articles.append({
                        'title': title,
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'sentiment_score': round(sentiment, 3),
                        'gold_impact': gold_impact,
                        'relevance': article.get('relevance_score', 0)
                    })
            
            # حساب المشاعر الإجمالية
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            overall_label = "محايد"
            if avg_sentiment > 0.2:
                overall_label = "إيجابي للذهب"
            elif avg_sentiment < -0.2:
                overall_label = "سلبي للذهب"
            
            return {
                'overall_sentiment': overall_label,
                'average_score': round(avg_sentiment, 3),
                'confidence': round(abs(avg_sentiment), 2),
                'analyzed_articles': analyzed_articles
            }
            
        except ImportError:
            logger.warning("مكتبة TextBlob غير متوفرة - تخطي تحليل المشاعر")
            return {'overall_sentiment': 'غير متوفر', 'confidence': 0}
        except Exception as e:
            logger.error(f"خطأ في تحليل المشاعر: {e}")
            return {'overall_sentiment': 'خطأ', 'confidence': 0}

    def generate_trading_signals(self, technical_data: pd.DataFrame, correlations: Dict, news_data: Dict) -> Dict:
        """توليد إشارات التداول المتقدمة"""
        logger.info("🎯 توليد إشارات التداول...")
        
        try:
            latest = technical_data.iloc[-1]
            prev = technical_data.iloc[-2]
            
            # إشارات التحليل الفني
            tech_signals = self.calculate_technical_signals(latest, prev)
            
            # إشارات التحليل الأساسي
            fundamental_signals = self.calculate_fundamental_signals(correlations)
            
            # إشارات الأخبار
            news_signals = self.calculate_news_signals(news_data)
            
            # حساب الإشارة المركبة
            total_signal = self.combine_signals(tech_signals, fundamental_signals, news_signals)
            
            # إدارة المخاطر
            risk_management = self.calculate_risk_management(latest, technical_data)
            
            return {
                'final_signal': total_signal,
                'component_signals': {
                    'technical': tech_signals,
                    'fundamental': fundamental_signals,
                    'news': news_signals
                },
                'risk_management': risk_management,
                'market_context': {
                    'current_price': round(latest['Close'], 2),
                    'daily_change_pct': round(((latest['Close'] / prev['Close']) - 1) * 100, 2),
                    'trend_direction': "صاعد" if latest['Close'] > latest.get('SMA_50', latest['Close']) else "هابط",
                    'volatility': self.calculate_volatility(technical_data['Close'])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}

    def calculate_technical_signals(self, latest: pd.Series, prev: pd.Series) -> Dict:
        """حساب الإشارات الفنية"""
        signals = {}
        score = 0
        
        # اتجاه السوق
        if latest['Close'] > latest.get('SMA_200', latest['Close']):
            signals['long_term_trend'] = "إيجابي"
            score += 2
        else:
            signals['long_term_trend'] = "سلبي"
            score -= 2
        
        # الزخم
        if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', 0):
            signals['momentum'] = "إيجابي"
            score += 1
        else:
            signals['momentum'] = "سلبي"
            score -= 1
        
        # RSI
        rsi = latest.get('RSI_14', 50)
        if 40 < rsi < 60:
            signals['rsi_signal'] = "محايد"
        elif rsi > 70:
            signals['rsi_signal'] = "ذروة شراء"
            score -= 1
        elif rsi < 30:
            signals['rsi_signal'] = "ذروة بيع"
            score += 1
        else:
            signals['rsi_signal'] = "طبيعي"
        
        # الحجم
        volume_ratio = latest.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:
            signals['volume'] = "حجم مرتفع"
            score += 0.5
        elif volume_ratio < 0.7:
            signals['volume'] = "حجم منخفض"
            score -= 0.5
        
        signals['technical_score'] = score
        return signals

    def calculate_fundamental_signals(self, correlations: Dict) -> Dict:
        """حساب الإشارات الأساسية"""
        signals = {}
        score = 0
        
        # الارتباط مع الدولار
        dxy_corr = correlations.get('all_correlations', {}).get('dxy', 0)
        if dxy_corr < -0.5:
            signals['dxy_relationship'] = "سلبي قوي - مفيد للذهب"
            score += 1
        elif dxy_corr > 0.3:
            signals['dxy_relationship'] = "إيجابي - غير طبيعي"
            score -= 0.5
        
        # مؤشر الخوف
        vix_corr = correlations.get('all_correlations', {}).get('vix', 0)
        if vix_corr > 0.3:
            signals['safe_haven_status'] = "نشط"
            score += 0.5
        
        signals['fundamental_score'] = score
        return signals

    def calculate_news_signals(self, news_data: Dict) -> Dict:
        """حساب إشارات الأخبار"""
        signals = {}
        score = 0
        
        if 'sentiment_analysis' in news_data:
            sentiment = news_data['sentiment_analysis']
            avg_score = sentiment.get('average_score', 0)
            confidence = sentiment.get('confidence', 0)
            
            if confidence > 0.3:  # ثقة عالية
                if avg_score > 0.1:
                    signals['news_sentiment'] = "إيجابي"
                    score += confidence * 2
                elif avg_score < -0.1:
                    signals['news_sentiment'] = "سلبي"
                    score -= confidence * 2
                else:
                    signals['news_sentiment'] = "محايد"
            else:
                signals['news_sentiment'] = "غير واضح"
        
        signals['news_score'] = score
        return signals

    def combine_signals(self, tech: Dict, fundamental: Dict, news: Dict) -> Dict:
        """دمج جميع الإشارات"""
        tech_weight = 0.5
        fundamental_weight = 0.3
        news_weight = 0.2
        
        total_score = (
            tech.get('technical_score', 0) * tech_weight +
            fundamental.get('fundamental_score', 0) * fundamental_weight +
            news.get('news_score', 0) * news_weight
        )
        
        # تحديد الإشارة النهائية
        if total_score >= 2:
            signal = "شراء قوي"
            confidence = "عالي"
        elif total_score >= 1:
            signal = "شراء"
            confidence = "متوسط"
        elif total_score <= -2:
            signal = "بيع قوي"
            confidence = "عالي"
        elif total_score <= -1:
            signal = "بيع"
            confidence = "متوسط"
        else:
            signal = "انتظار"
            confidence = "منخفض"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'total_score': round(total_score, 2),
            'recommendation': self.get_trading_recommendation(signal, confidence)
        }

    def get_trading_recommendation(self, signal: str, confidence: str) -> str:
        """الحصول على توصية التداول"""
        recommendations = {
            "شراء قوي": "افتح صفقة شراء بحجم كامل",
            "شراء": "افتح صفقة شراء بحجم متوسط",
            "بيع قوي": "افتح صفقة بيع بحجم كامل",
            "بيع": "افتح صفقة بيع بحجم متوسط",
            "انتظار": "ابق خارج السوق وانتظر إشارة أوضح"
        }
        
        return recommendations.get(signal, "لا توجد توصية واضحة")

    def calculate_risk_management(self, latest: pd.Series, data: pd.DataFrame) -> Dict:
        """حساب إدارة المخاطر"""
        current_price = latest['Close']
        atr = latest.get('ATRr_14', current_price * 0.02)
        
        return {
            'stop_loss_buy': round(current_price - (atr * 2), 2),
            'take_profit_buy': round(current_price + (atr * 3), 2),
            'stop_loss_sell': round(current_price + (atr * 2), 2),
            'take_profit_sell': round(current_price - (atr * 3), 2),
            'position_size_recommendation': self.calculate_position_size(data),
            'risk_reward_ratio': 1.5
        }

    def calculate_position_size(self, data: pd.DataFrame) -> str:
        """حساب حجم الصفقة المناسب"""
        volatility = self.calculate_volatility(data['Close'])
        
        if volatility > 25:
            return "صغير (تقلبات عالية)"
        elif volatility > 15:
            return "متوسط"
        else:
            return "كبير (تقلبات منخفضة)"

    def calculate_volatility(self, prices: pd.Series) -> float:
        """حساب التقلبات"""
        try:
            returns = prices.pct_change().dropna()
            volatility = returns.rolling(window=20).std() * np.sqrt(252) * 100
            return round(volatility.iloc[-1], 2)
        except:
            return 20.0  # قيمة افتراضية

    def generate_report(self, analysis_results: Dict) -> str:
        """إنشاء التقرير النهائي"""
        timestamp = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S EST')
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           📈 تحليل الذهب الشامل                              ║
║                          {timestamp}                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🎯 الإشارة النهائية: {analysis_results['trading_signals']['final_signal']['signal']}
🔍 مستوى الثقة: {analysis_results['trading_signals']['final_signal']['confidence']}
💰 السعر الحالي: ${analysis_results['trading_signals']['market_context']['current_price']}
📊 التغير اليومي: {analysis_results['trading_signals']['market_context']['daily_change_pct']}%

════════════════════════════════════════════════════════════════════════════════
                                    📊 التحليل الفني
════════════════════════════════════════════════════════════════════════════════
🔄 الاتجاه: {analysis_results['trading_signals']['market_context']['trend_direction']}
⚡ التقلبات: {analysis_results['trading_signals']['market_context']['volatility']}%

════════════════════════════════════════════════════════════════════════════════
                                   💼 إدارة المخاطر
════════════════════════════════════════════════════════════════════════════════
🛑 وقف الخسارة (شراء): ${analysis_results['trading_signals']['risk_management']['stop_loss_buy']}
🎯 الهدف (شراء): ${analysis_results['trading_signals']['risk_management']['take_profit_buy']}
📏 حجم الصفقة المقترح: {analysis_results['trading_signals']['risk_management']['position_size_recommendation']}

════════════════════════════════════════════════════════════════════════════════
                                    📰 تحليل الأخبار
════════════════════════════════════════════════════════════════════════════════
📑 إجمالي الأخبار: {analysis_results['news_analysis'].get('total_articles', 0)}
🔍 الأخبار ذات الصلة: {analysis_results['news_analysis'].get('relevant_articles', 0)}
💭 المشاعر العامة: {analysis_results['news_analysis'].get('sentiment_analysis', {}).get('overall_sentiment', 'غير متوفر')}

════════════════════════════════════════════════════════════════════════════════
                                   🔗 تحليل الارتباطات
════════════════════════════════════════════════════════════════════════════════
"""
        
        # إضافة الارتباطات
        correlations = analysis_results.get('correlations', {})
        if correlations.get('all_correlations'):
            report += "📊 الارتباطات الرئيسية:\n"
            for asset, corr in list(correlations['all_correlations'].items())[:5]:
                report += f"   • {asset}: {corr}\n"
        
        report += f"""
════════════════════════════════════════════════════════════════════════════════
                                     📝 التوصية
════════════════════════════════════════════════════════════════════════════════
{
