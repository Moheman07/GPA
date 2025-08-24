#!/usr/bin/env python3
"""
محلل الذهب الاحترافي المحسّن للعمل في GitHub Actions
النسخة 4.0 - محسّنة ومُبسّطة

المميزات الجديدة:
- معالجة محسّنة للأخطاء
- دعم متغيرات البيئة
- إشعارات تيليجرام
- تحليل الأخبار المبسط
- حفظ النتائج في ملفات
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

# إعداد متغيرات البيئة
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

class GoldAnalyzer:
    """محلل الذهب المحسّن"""
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'oil': 'CL=F'
        }
        
    def fetch_gold_data(self):
        """جلب بيانات الذهب مع معالجة الأخطاء"""
        print("📊 جلب بيانات الذهب...")
        
        for symbol_name, symbol in [('gold', self.symbols['gold']), ('gold_etf', self.symbols['gold_etf'])]:
            try:
                print(f"  • محاولة جلب {symbol_name} ({symbol})")
                data = yf.download(symbol, period="1y", interval="1d", progress=False)
                
                if not data.empty and len(data) > 50:
                    print(f"  ✅ نجح جلب {len(data)} يوم من البيانات")
                    return data
                else:
                    print(f"  ⚠️ بيانات غير كافية لـ {symbol_name}")
                    
            except Exception as e:
                print(f"  ❌ خطأ في جلب {symbol_name}: {e}")
                continue
        
        raise ValueError("فشل في جلب بيانات الذهب من جميع المصادر")
    
    def calculate_technical_indicators(self, data):
        """حساب المؤشرات الفنية"""
        print("📊 حساب المؤشرات الفنية...")
        
        try:
            df = data.copy()
            
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
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Volume indicators
            if 'Volume' in df.columns:
                df['Volume_SMA'] = df['Volume'].rolling(20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            print("✅ تم حساب المؤشرات بنجاح")
            return df.dropna()
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return data
    
    def generate_signals(self, data):
        """توليد إشارات التداول"""
        print("🎯 توليد إشارات التداول...")
        
        try:
            if len(data) < 10:
                raise ValueError("بيانات غير كافية للتحليل")
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            # حساب النقاط
            scores = {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'volatility': 0
            }
            
            # نقاط الاتجاه
            if latest['Close'] > latest.get('SMA_200', 0):
                scores['trend'] += 2
            if latest['Close'] > latest.get('SMA_50', 0):
                scores['trend'] += 1
            if latest['Close'] > latest.get('SMA_20', 0):
                scores['trend'] += 1
            scores['trend'] -= 2  # تطبيع
            
            # نقاط الزخم
            rsi = latest.get('RSI', 50)
            if 30 <= rsi <= 70:
                scores['momentum'] += 1
            elif rsi < 30:
                scores['momentum'] += 2  # ذروة بيع
            elif rsi > 70:
                scores['momentum'] -= 2  # ذروة شراء
            
            if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
                scores['momentum'] += 1
            else:
                scores['momentum'] -= 1
            
            # نقاط الحجم
            volume_ratio = latest.get('Volume_Ratio', 1)
            if volume_ratio > 1.5:
                scores['volume'] = 2
            elif volume_ratio > 1.2:
                scores['volume'] = 1
            elif volume_ratio < 0.8:
                scores['volume'] = -1
            
            # نقاط التقلب
            bb_position = latest.get('BB_Position', 0.5)
            if bb_position < 0.2:
                scores['volatility'] = 2  # قرب الحد السفلي
            elif bb_position > 0.8:
                scores['volatility'] = -2  # قرب الحد العلوي
            
            # حساب النقاط الإجمالية
            weights = {'trend': 0.4, 'momentum': 0.3, 'volume': 0.15, 'volatility': 0.15}
            total_score = sum(scores[key] * weights[key] for key in scores)
            
            # تحديد الإشارة
            if total_score >= 1.5:
                signal = "Strong Buy"
                confidence = "High"
                action = "شراء قوي - دخول بحجم كبير"
            elif total_score >= 0.5:
                signal = "Buy"
                confidence = "Medium"
                action = "شراء - دخول بحجم متوسط"
            elif total_score <= -1.5:
                signal = "Strong Sell"
                confidence = "High"
                action = "بيع قوي - تجنب أو خروج"
            elif total_score <= -0.5:
                signal = "Sell"
                confidence = "Medium"
                action = "بيع - تقليل المراكز"
            else:
                signal = "Hold"
                confidence = "Low"
                action = "انتظار - لا توجد إشارة واضحة"
            
            # إدارة المخاطر
            price = latest['Close']
            atr = latest.get('ATR', price * 0.02)
            
            risk_management = {
                'stop_loss': round(price - (atr * 2), 2),
                'take_profit_1': round(price + (atr * 2), 2),
                'take_profit_2': round(price + (atr * 4), 2),
                'position_size': self._get_position_size(confidence),
                'risk_reward_ratio': 2.0
            }
            
            return {
                'signal': signal,
                'confidence': confidence,
                'action': action,
                'total_score': round(total_score, 2),
                'component_scores': scores,
                'current_price': round(price, 2),
                'risk_management': risk_management,
                'technical_summary': {
                    'rsi': round(rsi, 1),
                    'macd_signal': 'positive' if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else 'negative',
                    'bb_position': round(bb_position, 2),
                    'volume_ratio': round(volume_ratio, 2)
                }
            }
            
        except Exception as e:
            return {'error': f'خطأ في توليد الإشارات: {e}'}
    
    def _get_position_size(self, confidence):
        """تحديد حجم المركز"""
        if confidence == "High":
            return "كبير (3-5% من رأس المال)"
        elif confidence == "Medium":
            return "متوسط (1-2% من رأس المال)"
        else:
            return "صغير (0.5-1% من رأس المال)"
    
    async def fetch_news_sentiment(self):
        """جلب وتحليل الأخبار"""
        print("📰 تحليل الأخبار...")
        
        if not NEWS_API_KEY:
            return {
                'status': 'no_api_key',
                'sentiment': 'neutral',
                'summary': 'يتطلب مفتاح API للأخبار'
            }
        
        try:
            url = f"https://newsapi.org/v2/everything?q=gold+price&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        return self._analyze_news_sentiment(articles)
                    else:
                        return {
                            'status': 'api_error',
                            'sentiment': 'neutral',
                            'summary': f'خطأ API: {response.status}'
                        }
        except Exception as e:
            return {
                'status': 'error',
                'sentiment': 'neutral',
                'summary': f'خطأ في جلب الأخبار: {e}'
            }
    
    def _analyze_news_sentiment(self, articles):
        """تحليل مشاعر الأخبار"""
        if not articles:
            return {
                'status': 'no_articles',
                'sentiment': 'neutral',
                'summary': 'لا توجد أخبار متاحة'
            }
        
        positive_words = ['surge', 'rally', 'gain', 'rise', 'bullish', 'strong', 'up']
        negative_words = ['fall', 'drop', 'decline', 'bearish', 'weak', 'down']
        
        positive_count = 0
        negative_count = 0
        
        for article in articles[:5]:
            title = article.get('title', '').lower()
            positive_count += sum(1 for word in positive_words if word in title)
            negative_count += sum(1 for word in negative_words if word in title)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            summary = f'أخبار إيجابية للذهب ({positive_count} مؤشر إيجابي)'
        elif negative_count > positive_count:
            sentiment = 'negative'
            summary = f'أخبار سلبية للذهب ({negative_count} مؤشر سلبي)'
        else:
            sentiment = 'neutral'
            summary = 'أخبار محايدة أو مختلطة'
        
        return {
            'status': 'success',
            'sentiment': sentiment,
            'summary': summary,
            'article_count': len(articles)
        }
    

    def save_results(self, result):
        """حفظ النتائج في ملفات"""
        try:
            # إنشاء مجلد النتائج
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # حفظ JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = results_dir / f"gold_analysis_{timestamp}.json"
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            # حفظ التقرير النصي
            text_report = self.generate_text_report(result)
            text_file = results_dir / f"report_{timestamp}.txt"
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            print(f"💾 تم حفظ النتائج:")
            print(f"  • JSON: {json_file}")
            print(f"  • تقرير: {text_file}")
            
            return str(json_file), str(text_file)
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")
            return None, None
    
    def generate_text_report(self, result):
        """توليد تقرير نصي"""
        report = []
        report.append("=" * 60)
        report.append("📊 تقرير تحليل الذهب الاحترافي")
        report.append("=" * 60)
        report.append(f"⏰ التوقيت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'error' in result:
            report.append(f"❌ خطأ: {result['error']}")
            return "\n".join(report)
        
        # الإشارة الرئيسية
        report.append("🎯 الإشارة الرئيسية:")
        report.append(f"  • الإشارة: {result.get('signal', 'غير محدد')}")
        report.append(f"  • مستوى الثقة: {result.get('confidence', 'غير محدد')}")
        report.append(f"  • التوصية: {result.get('action', 'غير محدد')}")
        report.append(f"  • السعر الحالي: ${result.get('current_price', 0):.2f}")
        report.append(f"  • النقاط الإجمالية: {result.get('total_score', 0)}")
        report.append("")
        
        # تفاصيل النقاط
        if 'component_scores' in result:
            report.append("📊 تحليل المكونات:")
            for component, score in result['component_scores'].items():
                report.append(f"  • {component}: {score}")
            report.append("")
        
        # إدارة المخاطر
        if 'risk_management' in result:
            rm = result['risk_management']
            report.append("⚠️ إدارة المخاطر:")
            report.append(f"  • وقف الخسارة: ${rm.get('stop_loss', 0):.2f}")
            report.append(f"  • الهدف الأول: ${rm.get('take_profit_1', 0):.2f}")
            report.append(f"  • الهدف الثاني: ${rm.get('take_profit_2', 0):.2f}")
            report.append(f"  • حجم المركز المقترح: {rm.get('position_size', 'غير محدد')}")
            report.append("")
        
        # تحليل الأخبار
        if 'news_analysis' in result:
            news = result['news_analysis']
            report.append("📰 تحليل الأخبار:")
            report.append(f"  • المشاعر: {news.get('sentiment', 'محايد')}")
            report.append(f"  • الملخص: {news.get('summary', 'غير متاح')}")
            report.append("")
        
        report.append("=" * 60)
        report.append("تم إنتاج التقرير بواسطة محلل الذهب الاحترافي V4.0")
        
        return "\n".join(report)
    
    async def run_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء التحليل الشامل للذهب...")
        print("=" * 60)
        
        try:
            # 1. جلب البيانات
            gold_data = self.fetch_gold_data()
            
            # 2. حساب المؤشرات الفنية
            technical_data = self.calculate_technical_indicators(gold_data)
            
            # 3. توليد الإشارات
            signals = self.generate_signals(technical_data)
            
            if 'error' in signals:
                raise ValueError(signals['error'])
            
            # 4. تحليل الأخبار (بشكل متوازي)
            news_analysis = await self.fetch_news_sentiment()
            
            # تجميع النتائج النهائية
            final_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'version': '4.0_github_optimized',
                'data_points': len(technical_data),
                **signals,
                'news_analysis': news_analysis
            }
            
            # 5. حفظ النتائج
            self.save_results(final_result)
            
            # 6. طباعة التقرير
            report = self.generate_text_report(final_result)
            print(report)
            
            print("\n✅ تم إنهاء التحليل بنجاح!")
            return final_result
            
        except Exception as e:
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'version': '4.0_github_optimized'
            }
            
            print(f"❌ فشل في التحليل: {e}")
            self.save_results(error_result)
            
            return error_result

def setup_environment():
    """إعداد البيئة"""
    print("🔧 إعداد البيئة...")
    
    # إنشاء المجلدات المطلوبة
    Path("results").mkdir(exist_ok=True)
    
    # التحقق من متغيرات البيئة
    if NEWS_API_KEY:
        print("✅ مفتاح API للأخبار متوفر")
    else:
        print("⚠️ مفتاح API للأخبار غير متوفر - سيتم تخطي تحليل الأخبار")

async def main():
    """الدالة الرئيسية"""
    try:
        # إعداد البيئة
        setup_environment()
        
        # إنشاء المحلل وتشغيله
        analyzer = GoldAnalyzer()
        result = await analyzer.run_analysis()
        
        # التحقق من النتيجة
        if result.get('status') == 'success':
            print("\n🎉 تم التحليل بنجاح!")
            exit_code = 0
        else:
            print(f"\n❌ فشل التحليل: {result.get('error', 'خطأ غير معروف')}")
            exit_code = 1
        
        # إنهاء البرنامج مع كود الخروج المناسب
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف التحليل بواسطة المستخدم")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 خطأ فادح: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # تشغيل التحليل
    asyncio.run(main())
