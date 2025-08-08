#!/usr/bin/env python3
"""
🏆 محلل الذهب الاحترافي النهائي
"""
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
import time
from typing import Dict, List, Optional
import pytz

warnings.filterwarnings('ignore')

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analysis_pro.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalGoldAnalyzerFinal:
    # --- كل الدوال السابقة تبقى كما هي ---
    # __init__, _setup_database, _load_sentiment_model, fetch_market_data_optimized, 
    # enhanced_news_analysis, calculate_gold_specific_indicators, run_simple_backtest,
    # calculate_comprehensive_technical_indicators, calculate_final_scores
    # --- يجب نسخها بالكامل من نسختك الأخيرة الناجحة ---
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'silver': 'SI=F',
            'dxy': 'DX-Y.NYB', 'vix': '^VIX', 'treasury': '^TNX',
            'oil': 'CL=F', 'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        self.db_path = "gold_analysis_history.db"
        self._setup_database()
        self._load_sentiment_model()
        logger.info("🚀 محلل الذهب الاحترافي جاهز")

    def _setup_database(self):
        # ... (الكود الكامل لقاعدة البيانات)
        pass # Placeholder

    def _load_sentiment_model(self):
        try:
            logger.info("🧠 تحميل نموذج تحليل المشاعر...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
            logger.info("✅ نموذج المشاعر جاهز")
        except Exception as e:
            logger.warning(f"⚠️ فشل تحميل المشاعر: {e}")

    def fetch_market_data_optimized(self) -> pd.DataFrame | None:
        # ... (الكود الكامل لجلب البيانات)
        logger.info("📊 جلب بيانات السوق...")
        try:
            data = yf.download(list(self.symbols.values()), period="15mo", interval="1d", threads=True, progress=False)
            if data.empty or ('Close', self.symbols['gold']) not in data.columns or data[('Close', self.symbols['gold'])].isnull().all():
                self.symbols['gold'] = 'GLD'
                data = yf.download(list(self.symbols.values()), period="15mo", interval="1d", threads=True, progress=False)
            
            gold_close_col = ('Close', self.symbols['gold'])
            if data.empty or gold_close_col not in data.columns or data[gold_close_col].isnull().all():
                raise ValueError("لا توجد بيانات صالحة للذهب")
            
            data = data.dropna(subset=[gold_close_col])
            if len(data) < 200: raise ValueError(f"بيانات غير كافية: {len(data)}")
            return data
        except Exception as e:
            logger.error(f"❌ خطأ جلب البيانات: {e}")
            return None
    
    def enhanced_news_analysis(self) -> Dict:
        # ... (الكود الكامل لتحليل الأخبار)
        return {"status": "skipped", "news_score": 0, "headlines": [], "confidence": 0} # Placeholder

    def calculate_gold_specific_indicators(self, gold_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        # ... (الكود الكامل للمؤشرات المتخصصة)
        return {'total_gold_specific_score': 0} # Placeholder

    def run_simple_backtest(self, gold_data: pd.DataFrame) -> Dict:
        # ... (الكود الكامل للاختبار الخلفي)
        return {'total_return_percent': 0, 'sharpe_ratio': 0, 'max_drawdown_percent': 0, 'win_rate_percent': 0} # Placeholder

    def calculate_comprehensive_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        # ... (الكود الكامل للمؤشرات الفنية)
        return pd.DataFrame() # Placeholder

    def calculate_final_scores(self, gold_data: pd.DataFrame, market_data: pd.DataFrame, gold_indicators: Dict, news_result: Dict) -> Dict:
        # ... (الكود الكامل لحساب النقاط)
        return {'trend': 0, 'momentum': 0} # Placeholder

    def _save_results_to_database(self, result: Dict):
        # ... (الكود الكامل لحفظ قاعدة البيانات)
        pass # Placeholder

    # --- ✅ هذه هي الدالة الكاملة والمصححة ---
    def run_complete_analysis(self) -> Dict:
        """التحليل الكامل النهائي"""
        start_time = time.time()
        logger.info("🚀 بدء التحليل الكامل النهائي...")
        
        try:
            market_data = self.fetch_market_data_optimized()
            if market_data is None: raise ValueError("فشل جلب بيانات السوق")

            gold_data_tech = self.calculate_comprehensive_technical_indicators(market_data)
            if gold_data_tech.empty: raise ValueError("فشل حساب المؤشرات الفنية")

            # استدعاء باقي الدوال التحليلية
            news_result = self.enhanced_news_analysis()
            gold_indicators = self.calculate_gold_specific_indicators(gold_data_tech, market_data)
            backtest_results = self.run_simple_backtest(gold_data_tech)
            scores = self.calculate_final_scores(gold_data_tech, market_data, gold_indicators, news_result)

            # ... (باقي منطق حساب النقاط النهائية وتحديد الإشارة)
            weights = {'trend': 0.30, 'momentum': 0.25, 'correlation': 0.20, 'gold_specific': 0.10, 'volatility': 0.10, 'seasonal': 0.05}
            technical_score = sum(scores.get(c, 0) * w for c, w in weights.items())
            news_contribution = news_result.get('news_score', 0) * 0.15
            final_score = technical_score + news_contribution
            
            signal, strength = "Hold", "Hold"
            if final_score >= 1.0: signal, strength = "Buy", "Strong Buy"
            elif final_score >= 0.5: signal, strength = "Buy", "Buy"
            elif final_score <= -1.0: signal, strength = "Sell", "Strong Sell"
            elif final_score <= -0.5: signal, strength = "Sell", "Sell"
            
            # ... (باقي منطق إدارة المخاطر والثقة)
            confidence_level = 0.5 # Placeholder
            
            # إعداد النتيجة النهائية
            final_result = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "status": "success",
                "signal": signal,
                "signal_strength": strength,
                "total_score": round(final_score, 3),
                "confidence_level": round(confidence_level, 3),
                "score_components": {k: round(v, 3) for k, v in scores.items()},
                "backtest_results": backtest_results,
                "news_analysis": news_result,
                # ... (باقي هيكل JSON)
            }

            # ✅ الخطوة المفقودة: حفظ النتيجة في ملف JSON
            with open("gold_analysis_pro.json", 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            logger.info("💾 تم حفظ النتائج في gold_analysis_pro.json")

            # حفظ في قاعدة البيانات
            self._save_results_to_database(final_result)
            
            logger.info(f"✅ اكتمل التحليل. الإشارة: {signal} ({strength})")
            return final_result

        except Exception as e:
            logger.error(f"❌ خطأ في التحليل الكامل: {e}")
            # في حالة الخطأ، نقوم بإنشاء ملف خطأ لتراه خطوة الحفظ
            error_result = {"status": "error", "error": str(e)}
            with open("gold_analysis_pro.json", 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
            return error_result

def main():
    try:
        analyzer = ProfessionalGoldAnalyzerFinal()
        analyzer.run_complete_analysis()
        print("\n🎉 تم إنجاز التحليل بنجاح!")
    except Exception as e:
        logger.critical(f"💥 خطأ فادح في التشغيل: {e}")

if __name__ == "__main__":
    main()
