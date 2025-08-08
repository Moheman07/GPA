#!/usr/bin/env python3
"""
🏆 محلل الذهب الاحترافي المتطور
Professional Gold Analyzer with Advanced Features

التحسينات المضمنة:
✅ تحسينات الأداء (asyncio, threading, caching)
✅ تحليل أخبار محسن ومتخصص
✅ مؤشرات فنية متخصصة بالذهب
✅ نظام Backtesting شامل
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sqlite3
import logging
import warnings
from datetime import datetime, timedelta
from transformers import pipeline
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
from typing import Dict, List, Tuple, Optional, Union
import threading
from dataclasses import dataclass
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# 📋 إعداد نظام التسجيل المحسن
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analysis_advanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# 🔧 تحسينات الأداء والسرعة
# =============================================================================

@dataclass
class PerformanceConfig:
    """إعدادات الأداء"""
    max_workers: int = 4
    request_timeout: int = 15
    cache_duration: int = 300  # 5 دقائق
    max_articles_per_query: int = 30
    sentiment_batch_size: int = 5

class AsyncDataFetcher:
    """جلب البيانات بشكل غير متزامن لتحسين الأداء"""
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_news_async(self, url: str) -> Optional[Dict]:
        """جلب الأخبار بشكل غير متزامن"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.warning(f"⚠️ خطأ في جلب الأخبار: {e}")
        return None

class DataCache:
    """نظام تخزين مؤقت للبيانات"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str, max_age_seconds: int = 300) -> Optional[any]:
        """استرجاع من التخزين المؤقت"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # فحص عمر الملف
        if (time.time() - cache_path.stat().st_mtime) > max_age_seconds:
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, key: str, value: any):
        """حفظ في التخزين المؤقت"""
        with self._lock:
            try:
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"⚠️ فشل حفظ التخزين المؤقت: {e}")

# =============================================================================
# 📰 تحليل الأخبار المحسن والمتخصص
# =============================================================================

class EnhancedNewsAnalyzer:
    """محلل أخبار متطور ومتخصص للذهب"""
    
    def __init__(self, api_key: str, sentiment_pipeline):
        self.api_key = api_key
        self.sentiment_pipeline = sentiment_pipeline
        self.cache = DataCache("news_cache")
        
        # كلمات مفتاحية متخصصة مع أوزان محسنة
        self.gold_keywords = {
            # الذهب المباشر - وزن عالي جداً
            'gold': 8, 'xau/usd': 8, 'xauusd': 8, 'bullion': 7, 'precious metal': 7,
            'gold price': 8, 'gold futures': 7, 'gold etf': 6, 'gld': 5,
            
            # السياسة النقدية - وزن عالي
            'federal reserve': 6, 'fed': 6, 'jerome powell': 6, 'fomc': 6,
            'interest rate': 6, 'rate cut': 7, 'rate hike': 7, 'monetary policy': 6,
            'quantitative easing': 6, 'tapering': 5, 'dovish': 5, 'hawkish': 5,
            
            # التضخم والاقتصاد - وزن متوسط عالي
            'inflation': 6, 'cpi': 6, 'pce': 5, 'consumer price': 6,
            'core inflation': 6, 'deflation': 5, 'stagflation': 6,
            'economic data': 4, 'gdp': 4, 'unemployment': 4, 'nfp': 5,
            
            # الدولار والعملات - وزن متوسط
            'dollar': 4, 'dxy': 5, 'dollar index': 5, 'usd': 3,
            'dollar strength': 5, 'dollar weakness': 6, 'currency': 3,
            
            # الجيوسياسية والأزمات - وزن متوسط عالي  
            'geopolitical': 5, 'safe haven': 7, 'safe-haven': 7, 'risk-off': 6,
            'war': 5, 'conflict': 5, 'tension': 4, 'sanctions': 4,
            'trade war': 4, 'tariff': 4, 'crisis': 5, 'recession': 6,
            
            # أسواق أخرى مؤثرة - وزن منخفض
            'stock market': 2, 'bonds': 3, 'treasury': 3, 'yield': 3,
            'oil': 2, 'commodities': 3, 'mining': 3, 'central bank': 4
        }
        
        # مصادر أخبار موثوقة مع أوزان
        self.trusted_sources = {
            'Reuters': 1.2, 'Bloomberg': 1.2, 'MarketWatch': 1.1,
            'CNBC': 1.1, 'Financial Times': 1.2, 'Wall Street Journal': 1.2,
            'Yahoo Finance': 1.0, 'Investing.com': 1.0, 'Kitco': 1.3,
            'GoldSeek': 1.3, 'BullionVault': 1.2
        }
    
    async def fetch_multi_source_news(self) -> List[Dict]:
        """جلب الأخبار من مصادر متعددة بالتوازي"""
        news_queries = [
            'gold OR XAU OR bullion OR "precious metals"',
            '"interest rates" OR "federal reserve" OR "jerome powell" OR FOMC',
            'inflation OR CPI OR "consumer prices" OR "monetary policy"',
            '"dollar index" OR DXY OR "dollar strength" OR USD',
            'geopolitical OR "safe haven" OR "risk off" OR crisis',
            '"gold price" OR "gold futures" OR "gold mining" OR GLD'
        ]
        
        all_articles = []
        
        async with AsyncDataFetcher(timeout=20) as fetcher:
            tasks = []
            
            for query in news_queries:
                url = (
                    f"https://newsapi.org/v2/everything?"
                    f"q={query}&language=en&sortBy=publishedAt&"
                    f"pageSize=25&"
                    f"from={(datetime.now() - timedelta(days=2)).date()}&"
                    f"apiKey={self.api_key}"
                )
                tasks.append(fetcher.fetch_news_async(url))
            
            # تنفيذ جميع الطلبات بالتوازي
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and 'articles' in result:
                    all_articles.extend(result['articles'])
        
        logger.info(f"🔍 تم جلب {len(all_articles)} مقالاً من جميع المصادر")
        return all_articles
    
    def calculate_relevance_score(self, article: Dict) -> Tuple[float, List[str]]:
        """حساب نقاط الصلة بالذهب مع الكلمات المطابقة"""
        title = (article.get('title', '') or '').lower()
        description = (article.get('description', '') or '').lower()
        content = f"{title} {description}"
        
        score = 0
        matched_keywords = []
        
        for keyword, weight in self.gold_keywords.items():
            if keyword in content:
                score += weight
                matched_keywords.append(keyword)
        
        # مكافأة إضافية للمصادر الموثوقة
        source_name = article.get('source', {}).get('name', '')
        if source_name in self.trusted_sources:
            score *= self.trusted_sources[source_name]
        
        return score, matched_keywords[:5]  # أهم 5 كلمات
    
    def batch_sentiment_analysis(self, texts: List[str]) -> List[Dict]:
        """تحليل المشاعر بالدفعات لتحسين الأداء"""
        results = []
        batch_size = PerformanceConfig.sentiment_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # معالجة الدفعة
                batch_results = []
                for text in batch:
                    if len(text.strip()) < 10:
                        batch_results.append({'positive': 0, 'negative': 0, 'neutral': 1})
                        continue
                    
                    # تحليل المشاعر
                    sentiment_output = self.sentiment_pipeline(text[:300])  # قطع النص
                    
                    # تحويل النتيجة إلى format موحد
                    sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
                    for item in sentiment_output[0]:
                        sentiment_scores[item['label']] = item['score']
                    
                    batch_results.append(sentiment_scores)
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"⚠️ خطأ في تحليل دفعة المشاعر: {e}")
                # إضافة نتائج محايدة للدفعة الفاشلة
                results.extend([{'positive': 0, 'negative': 0, 'neutral': 1}] * len(batch))
        
        return results
    
    async def run_enhanced_analysis(self) -> Dict:
        """تشغيل التحليل المحسن للأخبار"""
        logger.info("📰 بدء تحليل الأخبار المتطور...")
        
        if not self.api_key or not self.sentiment_pipeline:
            return self._get_default_result("missing_requirements")
        
        try:
            # فحص التخزين المؤقت
            cache_key = f"news_analysis_{datetime.now().strftime('%Y%m%d_%H')}"
            cached_result = self.cache.get(cache_key, max_age_seconds=1800)  # 30 دقيقة
            if cached_result:
                logger.info("📦 استخدام نتائج مخزنة مؤقتاً")
                return cached_result
            
            # جلب الأخبار
            all_articles = await self.fetch_multi_source_news()
            
            if not all_articles:
                return self._get_default_result("no_articles")
            
            # إزالة المكرر وفلترة المقالات
            unique_articles = self._remove_duplicates(all_articles)
            logger.info(f"🎯 تم الحصول على {len(unique_articles)} مقالاً فريداً")
            
            # تقييم الصلة بالذهب
            relevant_articles = []
            for article in unique_articles:
                relevance_score, matched_keywords = self.calculate_relevance_score(article)
                
                # قبول المقالات ذات الصلة العالية
                if relevance_score >= 4:  # حد مقبول
                    article['relevance_score'] = relevance_score
                    article['matched_keywords'] = matched_keywords
                    relevant_articles.append(article)
            
            if not relevant_articles:
                return self._get_default_result("no_relevant")
            
            # ترتيب حسب الصلة والحداثة
            relevant_articles.sort(key=lambda x: (x['relevance_score'], x.get('publishedAt', '')), reverse=True)
            top_articles = relevant_articles[:40]  # أفضل 40 مقال
            
            logger.info(f"🔥 تم اختيار {len(top_articles)} مقالاً عالي الصلة للتحليل")
            
            # تحضير النصوص لتحليل المشاعر
            texts_for_analysis = []
            for article in top_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                texts_for_analysis.append(text)
            
            # تحليل المشاعر بالدفعات
            sentiment_results = self.batch_sentiment_analysis(texts_for_analysis)
            
            # حساب النتائج النهائية
            weighted_sentiments = []
            processed_articles = []
            
            for i, (article, sentiment) in enumerate(zip(top_articles, sentiment_results)):
                try:
                    # حساب نتيجة المشاعر النهائية
                    sentiment_score = sentiment['positive'] - sentiment['negative']
                    
                    # وزن النتيجة حسب صلة المقال بالذهب
                    relevance_weight = min(article['relevance_score'] / 10, 1.5)
                    weighted_sentiment = sentiment_score * relevance_weight
                    
                    weighted_sentiments.append(weighted_sentiment)
                    
                    # حفظ المقال المعالج
                    processed_articles.append({
                        'title': article['title'],
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'sentiment_score': round(sentiment_score, 3),
                        'weighted_sentiment': round(weighted_sentiment, 3),
                        'relevance_score': round(article['relevance_score'], 1),
                        'matched_keywords': article['matched_keywords'],
                        'published_at': article.get('publishedAt', ''),
                        'confidence': round(max(sentiment.values()), 3)
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ خطأ في معالجة مقال: {e}")
                    continue
            
            if not weighted_sentiments:
                return self._get_default_result("analysis_failed")
            
            # حساب الإحصائيات النهائية
            final_sentiment = np.mean(weighted_sentiments)
            sentiment_volatility = np.std(weighted_sentiments)
            confidence_level = 1 - (sentiment_volatility / (abs(final_sentiment) + 0.1))
            
            # ترتيب المقالات للعرض
            processed_articles.sort(key=lambda x: (x['relevance_score'], abs(x['weighted_sentiment'])), reverse=True)
            
            # إعداد النتيجة النهائية
            result = {
                "status": "success",
                "news_score": round(final_sentiment, 3),
                "confidence_level": round(max(0, min(1, confidence_level)), 3),
                "headlines": processed_articles[:10],  # أهم 10 مقالات
                "analysis_details": {
                    'total_articles_analyzed': len(processed_articles),
                    'average_sentiment': round(final_sentiment, 3),
                    'sentiment_volatility': round(sentiment_volatility, 3),
                    'positive_articles': len([a for a in processed_articles if a['weighted_sentiment'] > 0.1]),
                    'negative_articles': len([a for a in processed_articles if a['weighted_sentiment'] < -0.1]),
                    'neutral_articles': len([a for a in processed_articles if abs(a['weighted_sentiment']) <= 0.1]),
                    'high_relevance_articles': len([a for a in processed_articles if a['relevance_score'] > 8]),
                    'confidence_distribution': {
                        'high_confidence': len([a for a in processed_articles if a['confidence'] > 0.8]),
                        'medium_confidence': len([a for a in processed_articles if 0.6 < a['confidence'] <= 0.8]),
                        'low_confidence': len([a for a in processed_articles if a['confidence'] <= 0.6])
                    }
                }
            }
            
            # حفظ في التخزين المؤقت
            self.cache.set(cache_key, result)
            
            logger.info(f"📊 تحليل الأخبار مكتمل: النتيجة {final_sentiment:.3f} (ثقة: {confidence_level:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل الأخبار: {e}")
            return self._get_default_result("error", str(e))
    
    def _remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """إزالة المقالات المكررة"""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title = (article.get('title', '') or '').lower().strip()
            if title and title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _get_default_result(self, status: str, error_msg: str = "") -> Dict:
        """إرجاع نتيجة افتراضية في حالة الأخطاء"""
        return {
            "status": status,
            "news_score": 0,
            "confidence_level": 0,
            "headlines": [],
            "analysis_details": {"error": error_msg} if error_msg else {}
        }

# =============================================================================
# 📈 مؤشرات فنية متخصصة بالذهب
# =============================================================================

class GoldSpecificIndicators:
    """مؤشرات فنية متخصصة بالذهب"""
    
    @staticmethod
    def gold_silver_ratio(gold_prices: pd.Series, silver_prices: pd.Series) -> pd.Series:
        """نسبة الذهب إلى الفضة - مؤشر مهم للمعادن الثمينة"""
        return gold_prices / silver_prices
    
    @staticmethod
    def gold_oil_ratio(gold_prices: pd.Series, oil_prices: pd.Series) -> pd.Series:
        """نسبة الذهب إلى النفط - مؤشر للتضخم"""
        return gold_prices / oil_prices
    
    @staticmethod
    def seasonal_strength(prices: pd.Series) -> Dict:
        """تحليل الأنماط الموسمية للذهب"""
        df = pd.DataFrame({'price': prices, 'date': prices.index})
        df['month'] = df['date'].dt.month
        df['returns'] = df['price'].pct_change()
        
        monthly_performance = df.groupby('month')['returns'].agg(['mean', 'std', 'count'])
        
        # أشهر قوة الذهب تاريخياً
        strong_months = [1, 2, 8, 9, 12]  # يناير، فبراير، أغسطس، سبتمبر، ديسمبر
        current_month = datetime.now().month
        
        seasonal_score = 1 if current_month in strong_months else -1
        
        return {
            'seasonal_score': seasonal_score,
            'current_month_historical_return': monthly_performance.loc[current_month, 'mean'] if current_month in monthly_performance.index else 0,
            'is_strong_season': current_month in strong_months,
            'monthly_stats': monthly_performance.to_dict('index')
        }
    
    @staticmethod
    def support_resistance_levels(prices: pd.Series, window: int = 20) -> Dict:
        """حساب مستويات الدعم والمقاومة للذهب"""
        high_prices = prices.rolling(window=window, center=True).max()
        low_prices = prices.rolling(window=window, center=True).min()
        
        current_price = prices.iloc[-1]
        
        # العثور على مستويات المقاومة (القمم المحلية)
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(prices) - window):
            if prices.iloc[i] == high_prices.iloc[i]:
                resistance_levels.append(prices.iloc[i])
            if prices.iloc[i] == low_prices.iloc[i]:
                support_levels.append(prices.iloc[i])
        
        # أقرب مستويات دعم ومقاومة
        resistance_above = [r for r in resistance_levels if r > current_price]
        support_below = [s for s in support_levels if s < current_price]
        
        nearest_resistance = min(resistance_above) if resistance_above else None
        nearest_support = max(support_below) if support_below else None
        
        return {
            'current_price': current_price,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'resistance_distance': (nearest_resistance - current_price) / current_price * 100 if nearest_resistance else None,
            'support_distance': (current_price - nearest_support) / current_price * 100 if nearest_support else None,
            'total_resistance_levels': len(resistance_levels),
            'total_support_levels': len(support_levels)
        }
    
    @staticmethod
    def gold_volatility_regime(prices: pd.Series, short_window: int = 10, long_window: int = 30) -> Dict:
        """تحديد نظام التقلبات الحالي للذهب"""
        returns = prices.pct_change().dropna()
        
        short_vol = returns.rolling(short_window).std() * np.sqrt(252)  # سنوي
        long_vol = returns.rolling(long_window).std() * np.sqrt(252)
        
        current_short_vol = short_vol.iloc[-1]
        current_long_vol = long_vol.iloc[-1]
        
        # تصنيف التقلبات
        if current_short_vol > 0.25:
            regime = "high_volatility"
            regime_score = 1  # التقلبات العالية تفيد الذهب
        elif current_short_vol < 0.15:
            regime = "low_volatility" 
            regime_score = -0.5
        else:
            regime = "normal_volatility"
            regime_score = 0
        
        # اتجاه التقلبات
        vol_trend = "increasing" if current_short_vol > current_long_vol else "decreasing"
        
        return {
            'current_volatility': round(current_short_vol, 3),
            'average_volatility': round(current_long_vol, 3),
            'volatility_regime': regime,
            'regime_score': regime_score,
            'volatility_trend': vol_trend,
            'volatility_percentile': round(
                (returns.rolling(252).std().iloc[-1] > returns.rolling(252).std().quantile(0.75)) * 100, 1
            )
        }
    
    @staticmethod
    def cot_simulation(prices: pd.Series) -> Dict:
        """محاكاة تقرير التزامات التجار (COT) للذهب"""
        # تحليل مبسط يحاكي سلوك COT
        returns = prices.pct_change().dropna()
        
        # تحليل الاتجاهات طويلة المدى (تجار تجاريون)
        long_term_trend = prices.rolling(50).mean().pct_change().iloc[-1]
        
        # تحليل المضاربة (صناديق التحوط)
        short_term_momentum = returns.rolling(10).mean().iloc[-1]
        
        # تحليل المتداولين الصغار
        price_vs_ma = (prices.iloc[-1] - prices.rolling(20).mean().iloc[-1]) / prices.iloc[-1] * 100
        
        # تقدير مراكز COT
        commercial_position = -long_term_trend * 10  # التجاريون عادة عكس الاتجاه
        hedge_fund_position = short_term_momentum * 10  # صناديق التحوط مع الزخم
        retail_position = price_vs_ma / 10  # الصغار يتبعون السعر
        
        return {
            'commercial_net_position': round(commercial_position, 2),
            'hedge_fund_net_position': round(hedge_fund_position, 2),
            'retail_net_position': round(retail_position, 2),
            'market_sentiment': 'bullish' if hedge_fund_position > 0 else 'bearish',
            'commercial_signal': 'buy' if commercial_position < -0.5 else 'sell' if commercial_position > 0.5 else 'neutral'
        }

# =============================================================================
# 🔬 نظام Backtesting الشامل
# =============================================================================

class GoldBacktestEngine:
    """نظام اختبار تاريخي شامل للاستراتيجية"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% عمولة
        
    def prepare_backtest_data(self, market_data: pd.DataFrame, signals_data: pd.DataFrame) -> pd.DataFrame:
        """تحضير البيانات للاختبار التاريخي"""
        # دمج البيانات
        backtest_df = market_data.copy()
        
        # إضافة الإشارات (محاكاة)
        backtest_df['signal'] = self._simulate_historical_signals(backtest_df)
        backtest_df['position'] = backtest_df['signal'].shift(1)  # تأخير يوم واحد للواقعية
        
        return backtest_df.dropna()
    
    def _simulate_historical_signals(self, data: pd.DataFrame) -> pd.Series:
        """محاكاة الإشارات التاريخية"""
        signals = []
        
        for i in range(len(data)):
            if i < 200:  # نحتاج بيانات كافية للمؤشرات
                signals.append(0)
                continue
            
            # حساب مؤشرات مبسطة للمحاكاة
            current_slice = data.iloc[max(0, i-200):i+1]
            price = current_slice['Close'].iloc[-1]
            sma_50 = current_slice['Close'].rolling(50).mean().iloc[-1]
            sma_200 = current_slice['Close'].rolling(200).mean().iloc[-1]
            
            # منطق الإشارات المبسط
            if price > sma_200 and price > sma_50:
                signal = 1  # شراء
            elif price < sma_200 and price < sma_50:
                signal = -1  # بيع
            else:
                signal = 0  # انتظار
            
            signals.append(signal)
        
        return pd.Series(signals, index=data.index)
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """تشغيل الاختبار التاريخي"""
        results = []
        capital = self.initial_capital
        position = 0
        entry_price = 0
        
        for i, row in data.iterrows():
            current_price = row['Close']
            current_signal = row.get('position', 0)
            
            if current_signal == 1 and position == 0:  # إشارة شراء
                # دخول صفقة شراء
                position = capital / current_price * (1 - self.commission)
                entry_price = current_price
                capital = 0
                action = 'BUY'
                
            elif current_signal == -1 and position > 0:  # إشارة بيع
                # إغلاق صفقة الشراء
                capital = position * current_price * (1 - self.commission)
                position = 0
                action = 'SELL'
                
            else:
                action = 'HOLD'
            
            # حساب القيمة الحالية للمحفظة
            portfolio_value = capital + (position * current_price if position > 0 else 0)
            
            results.append({
                'date': i,
                'price': current_price,
                'signal': current_signal,
                'action': action,
                'position': position,
                'capital': capital,
                'portfolio_value': portfolio_value,
                'return': (portfolio_value - self.initial_capital) / self.initial_capital * 100
            })
        
        return self._calculate_backtest_metrics(pd.DataFrame(results))
    
    def _calculate_backtest_metrics(self, results_df: pd.DataFrame) -> Dict:
        """حساب مقاييس الأداء"""
        returns = results_df['return'].pct_change().dropna()
        final_return = results_df['return'].iloc[-1]
        
        # مقاييس الأداء الأساسية
        total_return = final_return
        annualized_return = (1 + final_return/100) ** (252/len(results_df)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # أقصى انخفاض (Max Drawdown)
        cumulative = (1 + results_df['return']/100).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # عدد الصفقات
        signals = results_df['signal']
        trades = (signals != signals.shift()).sum() / 2
        
        # معدل الفوز
        trade_returns = []
        in_trade = False
        entry_value = 0
        
        for _, row in results_df.iterrows():
            if row['action'] == 'BUY':
                in_trade = True
                entry_value = row['portfolio_value']
            elif row['action'] == 'SELL' and in_trade:
                trade_return = (row['portfolio_value'] - entry_value) / entry_value
                trade_returns.append(trade_return)
                in_trade = False
        
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        
        return {
            'total_return_percent': round(total_return, 2),
            'annualized_return_percent': round(annualized_return * 100, 2),
            'volatility_percent': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_percent': round(max_drawdown * 100, 2),
            'total_trades': int(trades),
            'win_rate_percent': round(win_rate * 100, 2),
            'final_portfolio_value': round(results_df['portfolio_value'].iloc[-1], 2),
            'best_trade': round(max(trade_returns) * 100, 2) if trade_returns else 0,
            'worst_trade': round(min(trade_returns) * 100, 2) if trade_returns else 0,
            'average_trade': round(np.mean(trade_returns) * 100, 2) if trade_returns else 0
        }

# =============================================================================
# 🏆 المحلل الرئيسي المتقدم
# =============================================================================

class AdvancedGoldAnalyzer:
    """محلل الذهب المتقدم والشامل مع جميع التحسينات"""
    
    def __init__(self):
        # رموز السوق المحسنة
        self.symbols = {
            'gold': 'GC=F',          # Gold Futures
            'gold_etf': 'GLD',       # Gold ETF (backup)
            'silver': 'SI=F',        # Silver Futures
            'dxy': 'DX-Y.NYB',       # Dollar Index
            'vix': '^VIX',           # Volatility Index
            'treasury_10y': '^TNX',   # 10-Year Treasury
            'oil': 'CL=F',           # Oil Futures
            'spy': 'SPY',            # S&P 500 ETF
            'copper': 'HG=F',        # Copper (economic indicator)
        }
        
        # إعداد المكونات
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        self.cache = DataCache("analyzer_cache")
        self.db_path = "gold_analysis_advanced.db"
        
        # تهيئة المكونات
        self._setup_advanced_database()
        self._load_sentiment_model()
        
        # تهيئة المحللين المتخصصين
        self.news_analyzer = None
        if self.sentiment_pipeline:
            self.news_analyzer = EnhancedNewsAnalyzer(self.news_api_key, self.sentiment_pipeline)
        
        self.backtest_engine = GoldBacktestEngine()
        
        logger.info("🚀 محلل الذهب المتقدم جاهز للعمل")
    
    def _setup_advanced_database(self):
        """إعداد قاعدة بيانات متقدمة"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # جدول التحليلات الرئيسي
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS advanced_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    signal_strength TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    confidence_level REAL,
                    gold_price REAL,
                    execution_time_ms INTEGER,
                    
                    -- مكونات النتيجة
                    trend_score REAL,
                    momentum_score REAL,
                    correlation_score REAL,
                    news_score REAL,
                    volatility_score REAL,
                    seasonal_score REAL,
                    support_resistance_score REAL,
                    
                    -- بيانات السوق
                    dxy_value REAL,
                    vix_value REAL,
                    gold_silver_ratio REAL,
                    
                    -- مؤشرات فنية
                    rsi_value REAL,
                    macd_signal TEXT,
                    bb_position TEXT,
                    
                    -- إدارة المخاطر
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    position_size_suggestion REAL,
                    
                    -- تحليل الأخبار
                    news_sentiment REAL,
                    news_confidence REAL,
                    news_articles_count INTEGER,
                    
                    -- باك تيست
                    backtest_total_return REAL,
                    backtest_sharpe_ratio REAL,
                    backtest_max_drawdown REAL,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # جدول الأخبار المفصل
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detailed_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    headline TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    confidence_score REAL,
                    relevance_score REAL,
                    matched_keywords TEXT,
                    published_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES advanced_analysis (id)
                )
            ''')
            
            # جدول نتائج الباك تيست
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    test_period_days INTEGER,
                    total_return REAL,
                    annualized_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES advanced_analysis (id)
                )
            ''')
            
            # إنشاء فهارس للأداء
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON advanced_analysis(timestamp_utc)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal ON advanced_analysis(signal)')
            
            conn.commit()
            conn.close()
            logger.info("✅ قاعدة البيانات المتقدمة جاهزة")
            
        except Exception as e:
            logger.error(f"❌ خطأ في إعداد قاعدة البيانات: {e}")
    
    def _load_sentiment_model(self):
        """تحميل نموذج تحليل المشاعر المتقدم"""
        try:
            logger.info("🧠 تحميل نموذج تحليل المشاعر المتقدم...")
            
            # تجربة تحميل النموذج مع تحسينات الأداء
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                return_all_scores=True,
                device=-1  # استخدام CPU (أكثر استقراراً في البيئات السحابية)
            )
            
            logger.info("✅ نموذج تحليل المشاعر جاهز")
            
        except Exception as e:
            logger.warning(f"⚠️ تحذير: فشل تحميل نموذج المشاعر - {e}")
            self.sentiment_pipeline = None
    
    async def fetch_comprehensive_market_data(self) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق الشاملة بالتحسينات"""
        logger.info("📊 جلب بيانات السوق الشاملة...")
        
        # فحص التخزين المؤقت
        cache_key = f"market_data_{datetime.now().strftime('%Y%m%d_%H')}"
        cached_data = self.cache.get(cache_key, max_age_seconds=3600)  # ساعة واحدة
        
        if cached_data is not None:
            logger.info("📦 استخدام بيانات مخزنة مؤقتاً")
            return cached_data
        
        try:
            # جلب البيانات بالتوازي
            symbols_list = list(self.symbols.values())
            
            # استخدام yfinance مع التحسينات
            data = yf.download(
                symbols_list,
                period="18mo",  # سنة ونصف (توازن بين البيانات الكافية والسرعة)
                interval="1d",
                threads=True,
                progress=False,
                show_errors=False,
                repair=True  # إصلاح البيانات المعطوبة
            )
            
            if data.empty:
                logger.warning("⚠️ البيانات فارغة، محاولة جلب GLD كبديل...")
                # تجربة GLD كبديل
                self.symbols['gold'] = 'GLD'
                symbols_list[0] = 'GLD'
                data = yf.download(symbols_list, period="18mo", interval="1d", threads=True, progress=False)
            
            # تنظيف البيانات
            gold_symbol = self.symbols['gold']
            required_columns = [('Close', gold_symbol), ('High', gold_symbol), ('Low', gold_symbol), ('Volume', gold_symbol)]
            
            # التأكد من وجود الأعمدة المطلوبة
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"❌ أعمدة مفقودة: {missing_columns}")
                return None
            
            # إزالة الصفوف التي تحتوي على NaN في الذهب
            data = data.dropna(subset=[('Close', gold_symbol)])
            
            if len(data) < 100:
                logger.error(f"❌ البيانات غير كافية: {len(data)} صف فقط")
                return None
            
            # حفظ في التخزين المؤقت
            self.cache.set(cache_key, data)
            
            logger.info(f"✅ تم جلب {len(data)} يوم من البيانات ({gold_symbol})")
            return data
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
    def calculate_comprehensive_technical_analysis(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """حساب التحليل الفني الشامل مع المؤشرات المتخصصة"""
        logger.info("📈 حساب التحليل الفني الشامل...")
        
        try:
            gold_symbol = self.symbols['gold']
            
            # إعداد DataFrame للذهب
            gold_data = pd.DataFrame({
                'Open': market_data[('Open', gold_symbol)],
                'High': market_data[('High', gold_symbol)],
                'Low': market_data[('Low', gold_symbol)],
                'Close': market_data[('Close', gold_symbol)],
                'Volume': market_data[('Volume', gold_symbol)]
            }).dropna()
            
            # المؤشرات الأساسية المحسنة
            gold_data['SMA_10'] = ta.sma(gold_data['Close'], length=10)
            gold_data['SMA_20'] = ta.sma(gold_data['Close'], length=20)
            gold_data['SMA_50'] = ta.sma(gold_data['Close'], length=50)
            gold_data['SMA_100'] = ta.sma(gold_data['Close'], length=100)
            gold_data['SMA_200'] = ta.sma(gold_data['Close'], length=200)
            
            # المتوسطات الأسية
            gold_data['EMA_12'] = ta.ema(gold_data['Close'], length=12)
            gold_data['EMA_26'] = ta.ema(gold_data['Close'], length=26)
            gold_data['EMA_50'] = ta.ema(gold_data['Close'], length=50)
            
            # مؤشرات الزخم المتقدمة
            gold_data['RSI'] = ta.rsi(gold_data['Close'], length=14)
            gold_data['RSI_SMA'] = ta.sma(gold_data['RSI'], length=5)  # تنعيم RSI
            
            # MACD المحسن
            macd_data = ta.macd(gold_data['Close'], fast=12, slow=26, signal=9)
            gold_data['MACD'] = macd_data['MACD_12_26_9']
            gold_data['MACD_Signal'] = macd_data['MACDs_12_26_9']
            gold_data['MACD_Histogram'] = macd_data['MACDh_12_26_9']
            
            # Bollinger Bands متعددة المدد
            bb_20 = ta.bbands(gold_data['Close'], length=20, std=2)
            gold_data['BB_Upper_20'] = bb_20['BBU_20_2.0']
            gold_data['BB_Middle_20'] = bb_20['BBM_20_2.0']
            gold_data['BB_Lower_20'] = bb_20['BBL_20_2.0']
            gold_data['BB_Width'] = (gold_data['BB_Upper_20'] - gold_data['BB_Lower_20']) / gold_data['BB_Middle_20'] * 100
            gold_data['BB_Position'] = (gold_data['Close'] - gold_data['BB_Lower_20']) / (gold_data['BB_Upper_20'] - gold_data['BB_Lower_20']) * 100
            
            # مؤشرات التقلبات
            gold_data['ATR'] = ta.atr(gold_data['High'], gold_data['Low'], gold_data['Close'], length=14)
            gold_data['ATR_Percent'] = gold_data['ATR'] / gold_data['Close'] * 100
            
            # مؤشرات الحجم
            gold_data['OBV'] = ta.obv(gold_data['Close'], gold_data['Volume'])
            gold_data['Volume_SMA'] = ta.sma(gold_data['Volume'], length=20)
            gold_data['Volume_Ratio'] = gold_data['Volume'] / gold_data['Volume_SMA']
            
            # مؤشرات متقدمة
            gold_data['Williams_R'] = ta.willr(gold_data['High'], gold_data['Low'], gold_data['Close'], length=14)
            gold_data['CCI'] = ta.cci(gold_data['High'], gold_data['Low'], gold_data['Close'], length=20)
            
            # Stochastic
            stoch = ta.stoch(gold_data['High'], gold_data['Low'], gold_data['Close'])
            gold_data['Stoch_K'] = stoch['STOCHk_14_3_3']
            gold_data['Stoch_D'] = stoch['STOCHd_14_3_3']
            
            # المؤشرات المتخصصة بالذهب
            if self.symbols['silver'] in market_data.columns:
                silver_prices = market_data[('Close', self.symbols['silver'])]
                gold_data['Gold_Silver_Ratio'] = GoldSpecificIndicators.gold_silver_ratio(
                    gold_data['Close'], silver_prices
                )
            
            if self.symbols['oil'] in market_data.columns:
                oil_prices = market_data[('Close', self.symbols['oil'])]
                gold_data['Gold_Oil_Ratio'] = GoldSpecificIndicators.gold_oil_ratio(
                    gold_data['Close'], oil_prices
                )
            
            # تحليل الأنماط الموسمية
            seasonal_analysis = GoldSpecificIndicators.seasonal_strength(gold_data['Close'])
            
            # مستويات الدعم والمقاومة
            support_resistance = GoldSpecificIndicators.support_resistance_levels(gold_data['Close'])
            
            # نظام التقلبات
            volatility_regime = GoldSpecificIndicators.gold_volatility_regime(gold_data['Close'])
            
            # محاكاة COT
            cot_simulation = GoldSpecificIndicators.cot_simulation(gold_data['Close'])
            
            # إضافة البيانات المتخصصة
            gold_data = gold_data.assign(**{
                'Seasonal_Score': seasonal_analysis['seasonal_score'],
                'Volatility_Regime': volatility_regime['regime_score'],
                'Nearest_Resistance': support_resistance.get('nearest_resistance', gold_data['Close'].iloc[-1]),
                'Nearest_Support': support_resistance.get('nearest_support', gold_data['Close'].iloc[-1]),
                'COT_Commercial_Signal': 1 if cot_simulation['commercial_signal'] == 'buy' else -1 if cot_simulation['commercial_signal'] == 'sell' else 0
            })
            
            # تنظيف البيانات النهائي
            gold_data = gold_data.dropna()
            
            logger.info(f"✅ تم حساب {len(gold_data.columns)} مؤشراً فنياً - البيانات النظيفة: {len(gold_data)} صف")
            
            # حفظ التحليل المتخصص في المتغيرات للاستخدام لاحقاً
            self._seasonal_analysis = seasonal_analysis
            self._support_resistance = support_resistance
            self._volatility_regime = volatility_regime
            self._cot_simulation = cot_simulation
            
            return gold_data
            
        except Exception as e:
            logger.error(f"❌ خطأ في التحليل الفني: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_scores(self, gold_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """حساب النقاط المتقدمة لجميع المكونات"""
        logger.info("🎯 حساب النقاط المتقدمة...")
        
        try:
            latest = gold_data.iloc[-1]
            current_price = latest['Close']
            scores = {}
            
            # 1. نقاط الاتجاه المحسنة (وزن: 30%)
            trend_signals = 0
            if current_price > latest['SMA_200']: trend_signals += 3
            if current_price > latest['SMA_50']: trend_signals += 2
            if current_price > latest['SMA_20']: trend_signals += 1
            if latest['SMA_50'] > latest['SMA_200']: trend_signals += 1
            if latest['EMA_12'] > latest['EMA_26']: trend_signals += 1
            
            scores['trend'] = min(trend_signals / 8 * 4, 4) - 2  # تطبيع بين -2 و +2
            
            # 2. نقاط الزخم المتقدمة (وزن: 25%)
            momentum_signals = 0
            
            # MACD
            if latest['MACD'] > latest['MACD_Signal']: momentum_signals += 1
            if latest['MACD_Histogram'] > gold_data['MACD_Histogram'].iloc[-2]: momentum_signals += 1
            
            # RSI
            rsi = latest['RSI']
            if 40 < rsi < 60: momentum_signals += 1  # منطقة محايدة صحية
            elif 30 < rsi < 70: momentum_signals += 0.5
            elif rsi > 70: momentum_signals -= 0.5  # تشبع شراء
            elif rsi < 30: momentum_signals += 1.5  # تشبع بيع (فرصة شراء)
            
            # Williams %R
            if latest['Williams_R'] > -80: momentum_signals += 0.5
            
            # Stochastic
            if latest['Stoch_K'] > latest['Stoch_D']: momentum_signals += 0.5
            
            scores['momentum'] = (momentum_signals / 4.5 * 3) - 1.5  # تطبيع بين -1.5 و +1.5
            
            # 3. نقاط الارتباط والعلاقات (وزن: 20%)
            correlation_signals = 0
            
            # علاقة مع الدولار
            try:
                dxy_current = market_data[('Close', self.symbols['dxy'])].iloc[-1]
                dxy_ma = market_data[('Close', self.symbols['dxy'])].rolling(20).mean().iloc[-1]
                if dxy_current < dxy_ma: correlation_signals += 1  # ضعف الدولار يفيد الذهب
                if dxy_current < 105: correlation_signals += 0.5
            except:
                pass
            
            # علاقة مع VIX
            try:
                vix_current = market_data[('Close', self.symbols['vix'])].iloc[-1]
                if vix_current > 20: correlation_signals += 1  # الخوف يفيد الذهب
                if vix_current > 30: correlation_signals += 1
            except:
                pass
            
            # نسبة الذهب/الفضة
            if 'Gold_Silver_Ratio' in latest:
                gsr = latest['Gold_Silver_Ratio']
                if 70 < gsr < 90: correlation_signals += 0.5  # نطاق طبيعي
                elif gsr > 90: correlation_signals += 1  # الذهب مقوم بأعلى من قيمته
            
            scores['correlation'] = (correlation_signals / 4.5 * 2) - 1  # بين -1 و +1
            
            # 4. نقاط التقلبات والسوق (وزن: 15%)
            volatility_score = latest.get('Volatility_Regime', 0)
            
            # Bollinger Bands
            bb_position = latest.get('BB_Position', 50)
            if bb_position < 20: volatility_score += 1  # قرب الحد السفلي
            elif bb_position > 80: volatility_score -= 0.5  # قرب الحد العلوي
            
            # ATR
            atr_percent = latest.get('ATR_Percent', 2)
            if atr_percent > 3: volatility_score += 0.5  # تقلبات عالية تفيد الذهب
            
            scores['volatility'] = min(max(volatility_score, -1), 1)
            
            # 5. النقاط الموسمية (وزن: 5%)
            scores['seasonal'] = latest.get('Seasonal_Score', 0)
            
            # 6. نقاط الدعم والمقاومة (وزن: 5%)
            try:
                resistance_distance = self._support_resistance.get('resistance_distance', 10)
                support_distance = self._support_resistance.get('support_distance', 10)
                
                if support_distance and support_distance < 2:  # قريب من الدعم
                    support_resistance_score = 1
                elif resistance_distance and resistance_distance < 2:  # قريب من المقاومة
                    support_resistance_score = -0.5
                else:
                    support_resistance_score = 0
                    
                scores['support_resistance'] = support_resistance_score
            except:
                scores['support_resistance'] = 0
            
            logger.info("✅ تم حساب جميع النقاط المتقدمة")
            return scores
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب النقاط: {e}")
            return {
                'trend': 0, 'momentum': 0, 'correlation': 0, 
                'volatility': 0, 'seasonal': 0, 'support_resistance': 0
            }
    
    async def run_ultimate_analysis(self) -> Dict:
        """التحليل النهائي الشامل مع جميع التحسينات"""
        start_time = time.time()
        logger.info("🚀 بدء التحليل النهائي الشامل للذهب...")
        
        try:
            # 1. جلب بيانات السوق
            market_data = await self.fetch_comprehensive_market_data()
            if market_data is None:
                return {"status": "error", "error": "فشل في جلب بيانات السوق"}
            
            # 2. حساب التحليل الفني الشامل
            gold_data = self.calculate_comprehensive_technical_analysis(market_data)
            if gold_data.empty:
                return {"status": "error", "error": "فشل في حساب التحليل الفني"}
            
            # 3. تحليل الأخبار المتطور (بالتوازي)
            news_analysis_task = None
            if self.news_analyzer:
                news_analysis_task = asyncio.create_task(self.news_analyzer.run_enhanced_analysis())
            
            # 4. حساب النقاط المتقدمة
            scores = self.calculate_advanced_scores(gold_data, market_data)
            
            # 5. انتظار تحليل الأخبار
            if news_analysis_task:
                news_analysis = await news_analysis_task
            else:
                news_analysis = {"status": "skipped", "news_score": 0, "confidence_level": 0, "headlines": []}
            
            # 6. تشغيل الباك تيست
            logger.info("🔬 تشغيل اختبار تاريخي...")
            backtest_data = self.backtest_engine.prepare_backtest_data(market_data, gold_data)
            backtest_results = self.backtest_engine.run_backtest(backtest_data)
            
            # 7. حساب الأوزان المتكيفة
            vix_current = market_data[('Close', self.symbols['vix'])].iloc[-1] if ('Close', self.symbols['vix']) in market_data.columns else 20
            
            adaptive_weights = self._calculate_adaptive_weights(vix_current, scores, backtest_results)
            
            # 8. حساب النتيجة النهائية
            final_score = sum(scores[component] * weight for component, weight in adaptive_weights.items() if component in scores)
            
            # إضافة نقاط الأخبار
            news_weight = 0.10
            news_contribution = news_analysis.get('news_score', 0) * news_weight * 2
            final_score += news_contribution
            
            # 9. تحديد الإشارة وقوتها
            signal_info = self._determine_advanced_signal(final_score, backtest_results, news_analysis)
            
            # 10. حساب إدارة المخاطر
            latest = gold_data.iloc[-1]
            current_price = latest['Close']
            risk_management = self._calculate_risk_management(current_price, latest['ATR'], signal_info['signal'], latest)
            
            # 11. إعداد النتيجة النهائية
            execution_time = round((time.time() - start_time) * 1000)  # بالميلي ثانية
            
            comprehensive_result = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "execution_time_ms": execution_time,
                "status": "success",
                
                # الإشارة الرئيسية
                "signal": signal_info['signal'],
                "signal_strength": signal_info['strength'],
                "confidence_level": signal_info['confidence'],
                "total_score": round(final_score, 3),
                
                # مكونات التحليل
                "technical_scores": {k: round(v, 3) for k, v in scores.items()},
                "adaptive_weights": adaptive_weights,
                "news_contribution": round(news_contribution, 3),
                
                # بيانات السوق الحالية
                "market_data": {
                    "gold_price": round(current_price, 2),
                    "dxy": round(market_data[('Close', self.symbols['dxy'])].iloc[-1], 2) if ('Close', self.symbols['dxy']) in market_data.columns else 0,
                    "vix": round(vix_current, 2),
                    "gold_silver_ratio": round(latest.get('Gold_Silver_Ratio', 0), 2),
                    "gold_oil_ratio": round(latest.get('Gold_Oil_Ratio', 0), 2),
                },
                
                # المؤشرات الفنية الرئيسية
                "technical_indicators": {
                    "rsi": round(latest['RSI'], 2),
                    "macd_signal": "bullish" if latest['MACD'] > latest['MACD_Signal'] else "bearish",
                    "bb_position": round(latest.get('BB_Position', 50), 1),
                    "atr_percent": round(latest.get('ATR_Percent', 2), 3),
                    "volume_ratio": round(latest.get('Volume_Ratio', 1), 2),
                    "williams_r": round(latest['Williams_R'], 2),
                    "cci": round(latest['CCI'], 2)
                },
                
                # التحليل المتخصص بالذهب
                "gold_specific_analysis": {
                    "seasonal_analysis": getattr(self, '_seasonal_analysis', {}),
                    "support_resistance": getattr(self, '_support_resistance', {}),
                    "volatility_regime": getattr(self, '_volatility_regime', {}),
                    "cot_simulation": getattr(self, '_cot_simulation', {})
                },
                
                # إدارة المخاطر
                "risk_management": risk_management,
                
                # تحليل الأخبار
                "news_analysis": news_analysis,
                
                # نتائج الباك تيست
                "backtest_results": backtest_results,
                
                # معلومات الأداء
                "performance_info": {
                    "data_points_analyzed": len(gold_data),
                    "indicators_calculated": len(gold_data.columns),
                    "news_articles_processed": len(news_analysis.get('headlines', [])),
                    "backtest_period_days": len(backtest_data),
                    "cache_hits": "market_data" if self.cache.get(f"market_data_{datetime.now().strftime('%Y%m%d_%H')}") else "none"
                }
            }
            
            # 12. حفظ النتائج
            self._save_comprehensive_results(comprehensive_result)
            
            logger.info(f"✅ اكتمل التحليل الشامل في {execution_time}ms")
            logger.info(f"📊 الإشارة النهائية: {signal_info['signal']} ({signal_info['strength']}) - النتيجة: {final_score:.3f}")
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"❌ خطأ في التحليل الشامل: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "execution_time_ms": round((time.time() - start_time) * 1000)
            }
    
    def _calculate_adaptive_weights(self, vix_value: float, scores: Dict, backtest_results: Dict) -> Dict[str, float]:
        """حساب الأوزان المتكيفة بناءً على ظروف السوق والأداء التاريخي"""
        base_weights = {
            'trend': 0.30,
            'momentum': 0.25, 
            'correlation': 0.20,
            'volatility': 0.15,
            'seasonal': 0.05,
            'support_resistance': 0.05
        }
        
        # تعديل الأوزان حسب VIX
        if vix_value > 30:  # سوق عالي التقلب
            base_weights['volatility'] += 0.05
            base_weights['correlation'] += 0.05
            base_weights['trend'] -= 0.05
            base_weights['momentum'] -= 0.05
        elif vix_value < 15:  # سوق هادئ
            base_weights['trend'] += 0.05
            base_weights['momentum'] += 0.05
            base_weights['volatility'] -= 0.05
            base_weights['correlation'] -= 0.05
        
        # تعديل حسب أداء الباك تيست
        if backtest_results.get('sharpe_ratio', 0) > 1:
            # استراتيجية جيدة، زيادة وزن المؤشرات القوية
            for component, score in scores.items():
                if abs(score) > 1 and component in base_weights:
                    base_weights[component] += 0.02
        
        # تطبيع الأوزان
        total_weight = sum(base_weights.values())
        return {k: round(v/total_weight, 3) for k, v in base_weights.items()}
    
    def _determine_advanced_signal(self, final_score: float, backtest_results: Dict, news_analysis: Dict) -> Dict:
        """تحديد الإشارة المتقدمة مع مستوى الثقة"""
        
        # حساب مستوى الثقة
        confidence_factors = []
        
        # ثقة من النتيجة الفنية
        confidence_factors.append(min(abs(final_score) / 2, 1))
        
        # ثقة من الباك تيست
        sharpe = backtest_results.get('sharpe_ratio', 0)
        win_rate = backtest_results.get('win_rate_percent', 50) / 100
        confidence_factors.append(min(max(sharpe, 0) / 2, 1))
        confidence_factors.append(win_rate)
        
        # ثقة من الأخبار
        news_confidence = news_analysis.get('confidence_level', 0)
        confidence_factors.append(news_confidence)
        
        overall_confidence = np.mean(confidence_factors)
        
        # تحديد الإشارة
        if final_score >= 2.0:
            signal, strength = "Buy", "Very Strong Buy"
        elif final_score >= 1.5:
            signal, strength = "Buy", "Strong Buy" 
        elif final_score >= 1.0:
            signal, strength = "Buy", "Buy"
        elif final_score >= 0.5:
            signal, strength = "Buy", "Weak Buy"
        elif final_score <= -2.0:
            signal, strength = "Sell", "Very Strong Sell"
        elif final_score <= -1.5:
            signal, strength = "Sell", "Strong Sell"
        elif final_score <= -1.0:
            signal, strength = "Sell", "Sell"
        elif final_score <= -0.5:
            signal, strength = "Sell", "Weak Sell"
        else:
            signal, strength = "Hold", "Hold"
        
        # تعديل القوة حسب مستوى الثقة
        if overall_confidence < 0.5:
            if "Very Strong" in strength:
                strength = strength.replace("Very Strong", "Strong")
            elif "Strong" in strength and "Very" not in strength:
                strength = strength.replace("Strong", "")
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': round(overall_confidence, 3)
        }
    
    def _calculate_risk_management(self, current_price: float, atr: float, signal: str, latest_data: pd.Series) -> Dict:
        """حساب إدارة المخاطر المتقدمة"""
        
        risk_management = {}
        
        # حساب وقف الخسارة
        if 'buy' in signal.lower():
            stop_loss = current_price - (2.5 * atr)
            take_profit = current_price + (4 * atr)  # نسبة مخاطرة 1:1.6
        elif 'sell' in signal.lower():
            stop_loss = current_price + (2.5 * atr)
            take_profit = current_price - (4 * atr)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        # حساب حجم المركز المقترح (2% مخاطرة)
        risk_amount = 0.02  # 2% من رأس المال
        price_risk = abs(current_price - stop_loss)
        if price_risk > 0:
            position_size_percent = risk_amount / (price_risk / current_price)
        else:
            position_size_percent = 0.1  # افتراضي 10%
        
        risk_management = {
            'stop_loss_price': round(stop_loss, 2),
            'take_profit_price': round(take_profit, 2),
            'position_size_percent': round(min(position_size_percent * 100, 25), 2),  # حد أقصى 25%
            'risk_reward_ratio': round(abs(take_profit - current_price) / abs(current_price - stop_loss), 2) if abs(current_price - stop_loss) > 0 else 0,
            'atr_based_stop': True,
            'volatility_adjustment': round(latest_data.get('ATR_Percent', 2), 3)
        }
        
        return risk_management
    
    def _save_comprehensive_results(self, result: Dict) -> int:
        """حفظ النتائج الشاملة في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # حفظ النتيجة الرئيسية
            cursor.execute('''
                INSERT INTO advanced_analysis (
                    timestamp_utc, signal, signal_strength, total_score, confidence_level,
                    gold_price, execution_time_ms, trend_score, momentum_score,
                    correlation_score, news_score, volatility_score, seasonal_score,
                    support_resistance_score, dxy_value, vix_value, gold_silver_ratio,
                    rsi_value, macd_signal, bb_position, stop_loss_price,
                    take_profit_price, position_size_suggestion, news_sentiment,
                    news_confidence, news_articles_count, backtest_total_return,
                    backtest_sharpe_ratio, backtest_max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp_utc'], result['signal'], result['signal_strength'],
                result['total_score'], result['confidence_level'],
                result['market_data']['gold_price'], result['execution_time_ms'],
                result['technical_scores']['trend'], result['technical_scores']['momentum'],
                result['technical_scores']['correlation'], result['technical_scores'].get('news', 0),
                result['technical_scores']['volatility'], result['technical_scores']['seasonal'],
                result['technical_scores']['support_resistance'],
                result['market_data']['dxy'], result['market_data']['vix'],
                result['market_data']['gold_silver_ratio'],
                result['technical_indicators']['rsi'], result['technical_indicators']['macd_signal'],
                result['technical_indicators']['bb_position'],
                result['risk_management']['stop_loss_price'],
                result['risk_management']['take_profit_price'],
                result['risk_management']['position_size_percent'],
                result['news_analysis'].get('news_score', 0),
                result['news_analysis'].get('confidence_level', 0),
                len(result['news_analysis'].get('headlines', [])),
                result['backtest_results']['total_return_percent'],
                result['backtest_results']['sharpe_ratio'],
                result['backtest_results']['max_drawdown_percent']
            ))
            
            analysis_id = cursor.lastrowid
            
            # حفظ الأخبار التفصيلية
            for headline in result['news_analysis'].get('headlines', []):
                cursor.execute('''
                    INSERT INTO detailed_news 
                    (analysis_id, headline, source, sentiment_score, confidence_score, 
                     relevance_score, matched_keywords, published_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id, headline.get('title', ''), headline.get('source', ''),
                    headline.get('sentiment_score', 0), headline.get('confidence', 0),
                    headline.get('relevance_score', 0), 
                    json.dumps(headline.get('matched_keywords', [])),
                    headline.get('published_at', '')
                ))
            
            # حفظ نتائج الباك تيست
            cursor.execute('''
                INSERT INTO backtest_results 
                (analysis_id, test_period_days, total_return, annualized_return,
                 sharpe_ratio, max_drawdown, win_rate, total_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id, result['performance_info']['backtest_period_days'],
                result['backtest_results']['total_return_percent'],
                result['backtest_results']['annualized_return_percent'],
                result['backtest_results']['sharpe_ratio'],
                result['backtest_results']['max_drawdown_percent'],
                result['backtest_results']['win_rate_percent'],
                result['backtest_results']['total_trades']
            ))
            
            conn.commit()
            conn.close()
            
            # حفظ في ملف JSON
            with open("gold_analysis_ultimate.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 تم حفظ التحليل الشامل - ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ النتائج: {e}")
            return -1

# =============================================================================
# 🎯 تشغيل التحليل الرئيسي
# =============================================================================

async def main():
    """الدالة الرئيسية لتشغيل التحليل"""
    try:
        # إنشاء المحلل المتقدم
        analyzer = AdvancedGoldAnalyzer()
        
        # تشغيل التحليل الشامل
        results = await analyzer.run_ultimate_analysis()
        
        if results.get("status") == "error":
            logger.error(f"❌ فشل التحليل: {results.get('error')}")
            return results
        
        # عرض النتائج
        print("\n" + "="*80)
        print("🏆 التحليل النهائي الشامل للذهب")
        print("="*80)
        print(f"⏱️  وقت التنفيذ: {results['execution_time_ms']}ms")
        print(f"🎯 الإشارة: {results['signal']} ({results['signal_strength']})")
        print(f"📊 النتيجة الإجمالية: {results['total_score']}")
        print(f"🔒 مستوى الثقة: {results['confidence_level']:.1%}")
        print(f"💰 سعر الذهب: ${results['market_data']['gold_price']}")
        print(f"🛑 وقف الخسارة: ${results['risk_management']['stop_loss_price']}")
        print(f"🎯 جني الأرباح: ${results['risk_management']['take_profit_price']}")
        print(f"📏 حجم المركز المقترح: {results['risk_management']['position_size_percent']:.1f}%")
        
        print(f"\n📈 مكونات التحليل:")
        for component, score in results['technical_scores'].items():
            print(f"  • {component.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n📊 المؤشرات الفنية الرئيسية:")
        tech_indicators = results['technical_indicators']
        print(f"  • RSI: {tech_indicators['rsi']}")
        print(f"  • MACD: {tech_indicators['macd_signal']}")
        print(f"  • Bollinger Bands: {tech_indicators['bb_position']:.1f}%")
        
        print(f"\n🔬 نتائج الباك تيست:")
        bt = results['backtest_results']
        print(f"  • العائد الإجمالي: {bt['total_return_percent']:.2f}%")
        print(f"  • نسبة شارب: {bt['sharpe_ratio']:.2f}")
        print(f"  • أقصى انخفاض: {bt['max_drawdown_percent']:.2f}%")
        print(f"  • معدل الفوز: {bt['win_rate_percent']:.1f}%")
        
        print(f"\n📰 تحليل الأخبار:")
        news = results['news_analysis']
        print(f"  • حالة التحليل: {news['status']}")
        print(f"  • نتيجة المشاعر: {news.get('news_score', 0):.3f}")
        print(f"  • مستوى الثقة: {news.get('confidence_level', 0):.3f}")
        print(f"  • عدد المقالات: {len(news.get('headlines', []))}")
        
        if news.get('headlines'):
            print(f"\n📋 أهم العناوين:")
            for i, headline in enumerate(news['headlines'][:5], 1):
                print(f"  {i}. {headline['title'][:70]}... [{headline['source']}]")
        
        print(f"\n📁 تم حفظ التحليل الكامل في:")
        print("  • gold_analysis_ultimate.json")
        print("  • gold_analysis_advanced.db")
        print("  • gold_analysis_advanced.log")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ خطأ في التشغيل الرئيسي: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # تشغيل التحليل
    results = asyncio.run(main())
    
    # حفظ النتائج للمعالجة اللاحقة إذا لزم الأمر
    if results and results.get("status") != "error":
        logger.info("🎉 تم إنجاز جميع العمليات بنجاح!")
    else:
        logger.error("💥 فشل في إنجاز التحليل")
        exit(1)