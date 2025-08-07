#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import json
import os
import logging
from datetime import datetime, timedelta
import pytz
import warnings

warnings.filterwarnings('ignore')

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gold_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GoldAnalyzer:
    def __init__(self):
        self.setup_config()
        self.ensure_directories()
        
    def setup_config(self):
        """إعداد المتغيرات"""
        self.SYMBOLS = {
            'gold': 'GC=F',        # Gold Futures
            'gold_etf': 'GLD',     # Gold ETF
            'dxy': 'DX-Y.NYB',     # Dollar Index
            'vix': '^VIX',         # Fear Index
            'tnx': '^TNX',         # 10-Year Treasury
            'oil': 'CL=F',         # Oil
            'spy': 'SPY',          # S&P 500
            'btc': 'BTC-USD'       # Bitcoin
        }
        
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        
    def ensure_directories(self):
        """إنشاء المجلدات"""
        for directory in ['results', 'logs', 'data']:
            os.makedirs(directory, exist_ok=True)

    def fetch_market_data(self):
        """جلب بيانات السوق"""
        logger.info("🔄 جاري جلب بيانات السوق...")
        
        try:
            symbols = list(self.SYMBOLS.values())
            logger.info(f"Fetching data for symbols: {symbols}")
            
            # جلب البيانات
            data = yf.download(
                symbols, 
                period="1y", 
                interval="1d",
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if data.empty:
                raise ValueError("فشل في جلب البيانات - البيانات فارغة")
            
            logger.info(f"✅ تم جلب البيانات بنجاح. الشكل: {data.shape}")
            logger.info(f"الأعمدة: {data.columns.names}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات: {str(e)}")
            return None

    def calculate_technical_indicators(self, data, symbol_key):
        """حساب المؤشرات الفنية"""
        try:
            symbol = self.SYMBOLS[symbol_key]
            logger.info(f"حساب المؤشرات لـ {symbol}")
            
            # استخراج البيانات
            if len(data.columns.levels) > 1:
                # Multi-level columns
                df = data[symbol].copy()
            else:
                # Single level columns
                df = data.copy()
            
            # التأكد من وجود البيانات
            if df.empty:
                logger.warning(f"لا توجد بيانات لـ {symbol}")
                return None
                
            # حذف القيم المفقودة
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"بيانات غير كافية لـ {symbol}")
                return df
            
            # المؤشرات الأساسية
            try:
                df.ta.sma(length=20, append=True)
                df.ta.sma(length=50, append=True) 
                df.ta.sma(length=200, append=True)
                df.ta.rsi(length=14, append=True)
                df.ta.macd(append=True)
                df.ta.bbands(append=True)
                df.ta.atr(append=True)
                
                # مؤشرات مخصصة
                df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
                df['Volume_SMA'] = df['Volume'].rolling(20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                
                # مؤشر القوة العامة
                df['Strength_Index'] = (
                    (df['RSI_14'] - 50) * 0.4 + 
                    ((df['Close'] / df['SMA_50'] - 1) * 100) * 0.6
                )
                
                logger.info(f"✅ تم حساب المؤشرات بنجاح لـ {symbol}")
                
            except Exception as e:
                logger.warning(f"خطأ في حساب بعض المؤشرات: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب المؤشرات: {str(e)}")
            return None

    def analyze_correlations(self, data):
        """تحليل الارتباطات"""
        logger.info("📊 تحليل الارتباطات...")
        
        try:
            correlations = {}
            
            # استخراج أسعار الإغلاق
            prices = {}
            for key, symbol in self.SYMBOLS.items():
                try:
                    if len(data.columns.levels) > 1 and symbol in data.columns.levels[0]:
                        close_prices = data[symbol]['Close'].dropna()
                        if len(close_prices) > 100:  # التأكد من وجود بيانات كافية
                            prices[key] = close_prices
                            logger.info(f"تم استخراج {len(close_prices)} سعر لـ {key}")
                except Exception as e:
                    logger.warning(f"تخطي {key}: {str(e)}")
                    continue
            
            # حساب الارتباطات مع الذهب
            if 'gold' in prices:
                gold_prices = prices['gold']
                for asset, asset_prices in prices.items():
                    if asset != 'gold':
                        try:
                            # محاذاة التواريخ
                            common_index = gold_prices.index.intersection(asset_prices.index)
                            if len(common_index) > 50:
                                corr = gold_prices.loc[common_index].corr(
                                    asset_prices.loc[common_index]
                                )
                                correlations[asset] = round(corr, 3)
                        except Exception as e:
                            logger.warning(f"فشل حساب الارتباط مع {asset}: {str(e)}")
            
            logger.info(f"✅ تم حساب {len(correlations)} ارتباط")
            return correlations
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل الارتباطات: {str(e)}")
            return {}

    def fetch_gold_news(self):
        """جلب الأخبار"""
        logger.info("📰 جلب أخبار الذهب...")
        
        if not self.NEWS_API_KEY:
            logger.warning("مفتاح الأخبار غير متوفر")
            return {"status": "no_api_key", "articles": []}
        
        try:
            keywords = "gold OR XAU OR \"federal reserve\" OR inflation OR \"interest rate\""
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={keywords}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"pageSize=20&"
                f"from={(datetime.now() - timedelta(days=2)).date()}&"
                f"apiKey={self.NEWS_API_KEY}"
            )
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            # تصفية الأخبار المهمة
            filtered_articles = []
            for article in articles:
                title = (article.get('title', '') or '').lower()
                description = (article.get('description', '') or '').lower()
                
                if any(keyword in f"{title} {description}" for keyword in 
                       ['gold', 'xau', 'federal reserve', 'fed', 'inflation']):
                    filtered_articles.append({
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'url': article.get('url', '')
                    })
            
            logger.info(f"✅ تم جلب {len(filtered_articles)} خبر مهم")
            
            return {
                "status": "success",
                "total_articles": len(articles),
                "relevant_articles": len(filtered_articles),
                "articles": filtered_articles[:5]
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الأخبار: {str(e)}")
            return {"status": "error", "error": str(e), "articles": []}

    def generate_trading_signals(self, technical_data, correlations, news_data):
        """توليد إشارات التداول"""
        logger.info("🎯 توليد إشارات التداول...")
        
        try:
            if technical_data is None or technical_data.empty:
                raise ValueError("لا توجد بيانات فنية")
                
            latest = technical_data.iloc[-1]
            
            # حساب الإشارات
            signals = {}
            score = 0
            
            # الاتجاه العام
            if pd.notna(latest.get('SMA_200')):
                if latest['Close'] > latest['SMA_200']:
                    signals['trend'] = "صاعد"
                    score += 2
                else:
                    signals['trend'] = "هابط"
                    score -= 2
            else:
                signals['trend'] = "غير محدد"
            
            # الزخم
            if pd.notna(latest.get('MACD_12_26_9')) and pd.notna(latest.get('MACDs_12_26_9')):
                if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    signals['momentum'] = "إيجابي"
                    score += 1
                else:
                    signals['momentum'] = "سلبي"
                    score -= 1
            else:
                signals['momentum'] = "غير محدد"
            
            # RSI
            if pd.notna(latest.get('RSI_14')):
                rsi = latest['RSI_14']
                if rsi > 70:
                    signals['rsi'] = "ذروة شراء"
                    score -= 1
                elif rsi < 30:
                    signals['rsi'] = "ذروة بيع"
                    score += 1
                else:
                    signals['rsi'] = f"عادي ({rsi:.1f})"
            else:
                signals['rsi'] = "غير محدد"
            
            # تحديد الإشارة النهائية
            if score >= 2:
                final_signal = "شراء"
                confidence = "عالي" if score >= 3 else "متوسط"
            elif score <= -2:
                final_signal = "بيع"
                confidence = "عالي" if score <= -3 else "متوسط"
            else:
                final_signal = "انتظار"
                confidence = "منخفض"
            
            # إدارة المخاطر
            current_price = latest['Close']
            atr = latest.get('ATRr_14', current_price * 0.02)
            
            risk_management = {
                'current_price': round(current_price, 2),
                'stop_loss': round(current_price - (atr * 2), 2),
                'take_profit': round(current_price + (atr * 3), 2),
                'risk_reward_ratio': 1.5
            }
            
            result = {
                'final_signal': final_signal,
                'confidence': confidence,
                'score': score,
                'components': signals,
                'risk_management': risk_management,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ الإشارة النهائية: {final_signal} ({confidence})")
            return result
            
        except Exception as e:
            logger.error(f"❌ خطأ في توليد الإشارات: {str(e)}")
            return {"error": str(e)}

    def create_summary_report(self, analysis_results):
        """إنشاء تقرير الملخص"""
        try:
            timestamp = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S EST')
            
            # استخراج البيانات
            signals = analysis_results.get('signals', {})
            correlations = analysis_results.get('correlations', {})
            news = analysis_results.get('news', {})
            
            # إنشاء التقرير
            report = f"""
═══════════════════════════════════════════════════════════════════
                       📈 تحليل الذهب الشامل
                         {timestamp}
═══════════════════════════════════════════════════════════════════

🎯 الإشارة النهائية: {signals.get('final_signal', 'غير محدد')}
🔍 مستوى الثقة: {signals.get('confidence', 'غير محدد')}
💰 السعر الحالي: ${signals.get('risk_management', {}).get('current_price', 'N/A')}

═══════════════════════════════════════════════════════════════════
                           📊 التحليل الفني
═══════════════════════════════════════════════════════════════════
📈 الاتجاه العام: {signals.get('components', {}).get('trend', 'غير محدد')}
⚡ الزخم: {signals.get('components', {}).get('momentum', 'غير محدد')}
📊 مؤشر القوة النسبية: {signals.get('components', {}).get('rsi', 'غير محدد')}

═══════════════════════════════════════════════════════════════════
                          💼 إدارة المخاطر
═══════════════════════════════════════════════════════════════════
🛑 وقف الخسارة: ${signals.get('risk_management', {}).get('stop_loss', 'N/A')}
🎯 جني الأرباح: ${signals.get('risk_management', {}).get('take_profit', 'N/A')}

═══════════════════════════════════════════════════════════════════
                           🔗 الارتباطات
═══════════════════════════════════════════════════════════════════
"""
            
            # إضافة الارتباطات
            if correlations:
                for asset, corr in list(correlations.items())[:5]:
                    report += f"• {asset}: {corr}\n"
            else:
                report += "• لا توجد بيانات ارتباط متاحة\n"
            
            report += f"""
═══════════════════════════════════════════════════════════════════
                            📰 الأخبار
═══════════════════════════════════════════════════════════════════
📑 حالة الأخبار: {news.get('status', 'غير متوفر')}
🔍 عدد الأخبار المهمة: {news.get('relevant_articles', 0)}

"""
            
            # إضافة العناوين الرئيسية
            if news.get('articles'):
                report += "📰 العناوين الرئيسية:\n"
                for i, article in enumerate(news['articles'][:3], 1):
                    report += f"{i}. {article.get('title', 'بدون عنوان')}\n"
            
            report += f"""
═══════════════════════════════════════════════════════════════════
                            📝 الخلاصة
═══════════════════════════════════════════════════════════════════
النتيجة النهائية: {signals.get('final_signal', 'غير محدد')}
درجة التقييم: {signals.get('score', 0)}/5

تم إنتاج هذا التقرير تلقائياً بواسطة نظام تحليل الذهب المتطور
═══════════════════════════════════════════════════════════════════
"""
            
            return report
            
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء التقرير: {str(e)}")
            return f"خطأ في إنشاء التقرير: {str(e)}"

    def run_analysis(self):
        """تشغيل التحليل الشامل"""
        logger.info("🚀 بدء تحليل الذهب الشامل...")
        
        try:
            # 1. جلب بيانات السوق
            market_data = self.fetch_market_data()
            if market_data is None:
                raise ValueError("فشل في جلب بيانات السوق")
            
            # 2. حساب المؤشرات الفنية
            technical_data = self.calculate_technical_indicators(market_data, 'gold')
            
            # 3. تحليل الارتباطات
            correlations = self.analyze_correlations(market_data)
            
            # 4. جلب الأخبار
            news_data = self.fetch_gold_news()
            
            # 5. توليد الإشارات
            signals = self.generate_trading_signals(technical_data, correlations, news_data)
            
            # تجميع النتائج
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'signals': signals,
                'correlations': correlations,
                'news': news_data,
                'status': 'completed'
            }
            
            # 6. حفظ النتائج
            self.save_results(analysis_results)
            
            logger.info("✅ تم إتمام التحليل بنجاح")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ فشل في التحليل: {str(e)}")
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            self.save_results(error_result)
            return error_result

    def save_results(self, results):
        """حفظ النتائج"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # حفظ JSON
            json_file = f"results/gold_analysis_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            # حفظ ملخص نصي
            summary = self.create_summary_report(results)
            txt_file = f"results/gold_analysis_summary.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # حفظ آخر تحليل
            latest_file = "results/latest_analysis.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"✅ تم حفظ النتائج في {json_file}")
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ النتائج: {str(e)}")

def main():
    """الدالة الرئيسية"""
    try:
        print("🏁 بدء تشغيل محلل الذهب...")
        
        analyzer = GoldAnalyzer()
        results = analyzer.run_analysis()
        
        if results.get('status') == 'completed':
            print("✅ تم إتمام التحليل بنجاح!")
            
            # طباعة ملخص سريع
            signals = results.get('signals', {})
            print(f"\n📊 النتيجة: {signals.get('final_signal', 'غير محدد')}")
            print(f"🔍 الثقة: {signals.get('confidence', 'غير محدد')}")
            
            if 'risk_management' in signals:
                rm = signals['risk_management']
                print(f"💰 السعر: ${rm.get('current_price', 'N/A')}")
                
        else:
            print(f"❌ فشل التحليل: {results.get('error', 'خطأ غير محدد')}")
            
    except Exception as e:
        print(f"💥 خطأ حرج: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
