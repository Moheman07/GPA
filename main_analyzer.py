#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

class ProfessionalGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GLD', 'dxy': 'DX-Y.NYB', 'vix': '^VIX',
            'treasury': '^TNX', 'oil': 'CL=F', 'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        # ✅ تحميل نموذج تحليل المشاعر مرة واحدة فقط عند بدء التشغيل للكفاءة
        try:
            print("🧠 تحميل نموذج تحليل المشاعر المالي (قد يستغرق بعض الوقت أول مرة)...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            print("✅ نموذج تحليل المشاعر جاهز.")
        except Exception as e:
            print(f"⚠️ تحذير: فشل تحميل نموذج المشاعر. سيتم تخطي تحليل الأخبار. الخطأ: {e}")
            self.sentiment_pipeline = None

    def analyze_news(self):
        """
        الطبقة الكاملة لتحليل الأخبار: جلب، فلترة بالأهمية، وتحليل المشاعر.
        """
        print("\n📰 بدء محرك تحليل الأخبار...")
        if not self.news_api_key or not self.sentiment_pipeline:
            return {"status": "skipped", "news_score": 0, "headlines": []}

        # 1. الفلترة المسبقة عبر استعلام API ذكي
        query = ('(gold OR XAU OR bullion) OR ("interest rates" AND ("federal reserve" OR fed)) OR '
                 '(inflation AND (CPI OR "jobs report")) OR (geopolitical AND (risk OR tension))')
        
        url = (f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt"
               f"&pageSize=100&from={(datetime.now() - timedelta(days=2)).date()}&apiKey={self.news_api_key}")

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not articles: raise ValueError("لم يتم العثور على مقالات.")
            print(f"🔍 تم جلب {len(articles)} مقالاً أولياً.")

            # 2. الفلترة اللاحقة عبر نظام نقاط الأهمية
            keyword_scores = {
                'gold': 3, 'xau': 3, 'bullion': 3,
                'federal reserve': 2, 'fed': 2, 'interest rate': 2, 'inflation': 2, 'cpi': 2, 'nfp': 2,
                'dollar': 1, 'dxy': 1, 'geopolitical': 1, 'risk': 1, 'tension': 1, 'war': 1
            }
            
            scored_articles = []
            for article in articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                score = sum(points for keyword, points in keyword_scores.items() if keyword in content)
                
                if score >= 3: # حد القبول (يمكن تعديله)
                    article['relevance_score'] = score
                    scored_articles.append(article)
            
            if not scored_articles: raise ValueError("لم يتم العثور على مقالات ذات أهمية كافية بعد الفلترة.")
            print(f"🎯 تم تحديد {len(scored_articles)} مقالاً ذا أهمية عالية.")

            # 3. تحليل المشاعر للمقالات المفلترة فقط
            total_sentiment = 0
            for article in scored_articles:
                text_to_analyze = article['description'] or article['title']
                result = self.sentiment_pipeline(text_to_analyze)[0]
                if result['label'] == 'positive':
                    total_sentiment += result['score']
                elif result['label'] == 'negative':
                    total_sentiment -= result['score']

            # 4. حساب النتيجة النهائية للأخبار
            news_score = round(total_sentiment / len(scored_articles), 2)
            print(f"⚖️ نتيجة مشاعر الأخبار النهائية: {news_score}")
            
            # فرز المقالات حسب الأهمية لعرضها
            sorted_articles = sorted(scored_articles, key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                "status": "success",
                "news_score": news_score, # النتيجة الرقمية التي ستدخل في التحليل الفني
                "headlines": [{'title': a['title'], 'source': a.get('source', {}).get('name')} for a in sorted_articles[:5]]
            }

        except Exception as e:
            print(f"❌ خطأ في تحليل الأخبار: {e}")
            return {"status": "error", "news_score": 0, "headlines": []}

    # ... باقي الدوال تبقى كما هي (fetch_data, calculate_indicators, etc.) ...
    # ... لكن سنقوم بتعديل دالة `generate_professional_signals` لتستقبل نتيجة الأخبار ...

    def generate_professional_signals(self, technical_data, correlations, volume_analysis, fibonacci_levels, news_analysis):
        """توليد إشارات احترافية مدمجة (فني + أساسي)"""
        try:
            print("🎯 توليد إشارات مدمجة...")
            latest = technical_data.iloc[-1]
            # ... (كل منطق حساب النقاط الفنية يبقى كما هو) ...
            trend_score, momentum_score, volume_score, fib_score, correlation_score = 0,0,0,0,0
            # (هنا يتم حساب النقاط الفنية... لم أكرر الكود للاختصار)
            
            # --- ✅ دمج نتيجة الأخبار ---
            news_score = news_analysis.get('news_score', 0)
            
            # حساب النتيجة النهائية المرجحة (مع إضافة وزن للأخبار)
            total_score = (
                trend_score * 0.30 +
                momentum_score * 0.25 +
                volume_score * 0.10 +
                fib_score * 0.10 +
                correlation_score * 0.15 +
                news_score * 0.10  # <-- تم إضافة وزن 10% للأخبار
            )
            
            # ... (باقي منطق تحديد الإشارة النهائية وإدارة المخاطر يبقى كما هو) ...
            
            # (هذا مجرد مثال مختصر، يجب استخدام الكود الكامل من السكربت السابق لهذه الدالة)
            final_signal = "Hold"
            if total_score >= 1.5: final_signal = "Buy"
            elif total_score <= -1.5: final_signal = "Sell"
            
            result = {
                'signal': final_signal,
                'total_score': round(total_score, 2),
                'component_scores': {
                    'trend': trend_score, 'momentum': momentum_score, 'volume': volume_score,
                    'fibonacci': fib_score, 'correlation': correlation_score,
                    'news': news_score # <-- إضافة نتيجة الأخبار للمكونات
                },
                # ... باقي المخرجات
            }
            return result

        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات المدمجة: {e}")
            return {"error": str(e)}

    def run_analysis(self):
        """تشغيل التحليل الاحترافي الشامل"""
        # ...
        # 7. جلب وتحليل الأخبار
        news_analysis_result = self.analyze_news()
        # ...
        # 8. توليد الإشارات الاحترافية المدمجة
        signals = self.generate_professional_signals(
            technical_data, correlations, volume_analysis, fibonacci_levels, news_analysis_result
        )
        # ...
        # 9. تجميع النتائج النهائية (مع إضافة نتيجة الأخبار)
        results = {
            # ...
            'news_analysis': {
                'status': news_analysis_result.get('status'),
                'news_sentiment_score': news_analysis_result.get('news_score'),
                'headlines': news_analysis_result.get('headlines')
            },
            # ...
        }
        # ...
        return results

    # يجب نسخ باقي الدوال من السكربت السابق هنا...
    # (main, save_single_result, get_market_status, calculate_indicators, etc.)
