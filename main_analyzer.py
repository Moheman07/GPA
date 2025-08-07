#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from transformers import pipeline
import pytz
import warnings

warnings.filterwarnings('ignore')

class ProfessionalGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GLD', 'dxy': 'DX-Y.NYB', 'vix': '^VIX',
            'treasury': '^TNX', 'oil': 'CL=F', 'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        try:
            print("🧠 تحميل نموذج تحليل المشاعر المالي...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            print("✅ نموذج تحليل المشاعر جاهز.")
        except Exception as e:
            print(f"⚠️ تحذير: فشل تحميل نموذج المشاعر. الخطأ: {e}")

    # --- ✅ تم تعديل هذه الدالة بالكامل ---
    def analyze_news(self):
        """
        الطبقة الكاملة لتحليل الأخبار: جلب، فلترة بالأهمية، وتحليل المشاعر.
        """
        print("\n📰 بدء محرك تحليل الأخبار...")
        if not self.news_api_key or not self.sentiment_pipeline:
            return {"status": "skipped", "news_score": 0, "headlines": []}

        try:
            query = ('(gold OR XAU OR bullion OR "precious metal") OR ("interest rate" OR fed OR inflation OR CPI OR NFP OR tariff OR geopolitical)')
            
            url = (f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt"
                   f"&pageSize=100&from={(datetime.now() - timedelta(days=2)).date()}&apiKey={self.news_api_key}")

            response = requests.get(url, timeout=20)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not articles: raise ValueError("لم يتم العثور على مقالات.")
            print(f"🔍 تم جلب {len(articles)} مقالاً أولياً.")

            keyword_scores = {
                'gold': 3, 'xau': 3, 'bullion': 3, 'precious metal': 3,
                'federal reserve': 2, 'fed': 2, 'interest rate': 2, 'inflation': 2, 'cpi': 2, 'nfp': 2,
                'dollar': 1, 'dxy': 1, 'geopolitical': 1, 'risk': 1, 'tension': 1, 'war': 1, 'tariff': 1
            }
            
            scored_articles = []
            for article in articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                score = sum(points for keyword, points in keyword_scores.items() if keyword in content)
                
                # --- ✅ تم خفض حد القبول ---
                if score >= 2: # الآن نقبل المقالات التي تحصل على نقطتين أو أكثر
                    article['relevance_score'] = score
                    scored_articles.append(article)
            
            if not scored_articles: raise ValueError("لم يتم العثور على مقالات ذات أهمية كافية بعد الفلترة.")
            print(f"🎯 تم تحديد {len(scored_articles)} مقالاً ذا أهمية عالية.")

            total_sentiment = sum(
                (res['score'] if res['label'] == 'positive' else -res['score'])
                for art in scored_articles
                if (res := self.sentiment_pipeline(art['description'] or art['title'])[0])
            )

            news_score = round(total_sentiment / len(scored_articles), 2)
            print(f"⚖️ نتيجة مشاعر الأخبار النهائية: {news_score}")
            
            sorted_articles = sorted(scored_articles, key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                "status": "success", "news_score": news_score,
                "headlines": [{'title': a['title'], 'source': a.get('source', {}).get('name')} for a in sorted_articles[:5]]
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل الأخبار: {e}")
            return {"status": "error", "news_score": 0, "headlines": []}

    # ... باقي الدوال تبقى كما هي ...
    # (fetch_data, run_full_analysis, main, etc.)
    # (تأكد من استخدام النسخة الكاملة من السكربت السابق، وقم فقط باستبدال دالة analyze_news بهذه النسخة الجديدة)
