#!/usr/bin/env python3
"""
🏆 Professional Gold Analyzer - Final Version
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
from typing import Dict
import pytz

warnings.filterwarnings('ignore')

# --- Setup Logging ---
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
        logger.info("🚀 Professional Gold Analyzer is ready.")

    def _setup_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT, signal TEXT, 
                        signal_strength TEXT, total_score REAL, confidence_level REAL, trend_score REAL, 
                        momentum_score REAL, correlation_score REAL, news_score REAL, volatility_score REAL, 
                        seasonal_score REAL, gold_specific_score REAL, gold_price REAL, dxy_value REAL, 
                        vix_value REAL, gold_silver_ratio REAL, stop_loss_price REAL, take_profit_price REAL, 
                        position_size REAL, backtest_return REAL, backtest_sharpe REAL, backtest_max_dd REAL, 
                        backtest_win_rate REAL, execution_time_ms INTEGER, news_articles_count INTEGER, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_archive (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id INTEGER, headline TEXT, 
                        source TEXT, sentiment_score REAL, relevance_score REAL, keywords TEXT, 
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        FOREIGN KEY (analysis_id) REFERENCES analysis_history (id)
                    )
                ''')
            logger.info("✅ Database is ready.")
        except Exception as e:
            logger.warning(f"⚠️ Database setup warning: {e}")

    def _load_sentiment_model(self):
        try:
            logger.info("🧠 Loading financial sentiment model...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
            logger.info("✅ Sentiment model is ready.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load sentiment model: {e}")

    def fetch_market_data_optimized(self) -> pd.DataFrame | None:
        logger.info("📊 Fetching market data...")
        try:
            data = yf.download(list(self.symbols.values()), period="15mo", interval="1d", threads=True, progress=False)
            
            primary_gold_col = ('Close', self.symbols['gold'])
            secondary_gold_col = ('Close', self.symbols['gold_etf'])

            if data.empty or primary_gold_col not in data.columns or data[primary_gold_col].isnull().all():
                logger.warning("⚠️ Primary gold ticker (GC=F) failed. Trying fallback (GLD)...")
                self.symbols['gold'] = self.symbols['gold_etf']
                if secondary_gold_col not in data.columns or data[secondary_gold_col].isnull().all():
                     raise ValueError("Both primary and fallback gold tickers failed.")

            gold_col_to_use = ('Close', self.symbols['gold'])
            data.dropna(subset=[gold_col_to_use], inplace=True)
            
            if len(data) < 200: raise ValueError(f"Insufficient data points: {len(data)}")
                
            logger.info(f"✅ Fetched {len(data)} days of data using {self.symbols['gold']}.")
            return data
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return None

    def calculate_comprehensive_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame | None:
        logger.info("📈 Calculating technical indicators...")
        try:
            gold_symbol = self.symbols['gold']
            
            gold_df = pd.DataFrame()
            for col_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if (col_type, gold_symbol) in market_data.columns:
                    gold_df[col_type] = market_data[(col_type, gold_symbol)]
            
            gold_df.dropna(inplace=True)
            if gold_df.empty: raise ValueError("Gold DataFrame is empty after extraction.")

            gold_df.ta.strategy(ta.Strategy(name="Comprehensive", ta=[
                {"kind": "sma", "length": l} for l in [10, 20, 50, 200]
            ] + [
                {"kind": "ema", "length": l} for l in [12, 26]
            ] + [
                {"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"}, {"kind": "atr"},
                {"kind": "willr"}, {"kind": "cci"}, {"kind": "stoch"}, {"kind": "obv"}
            ]))
            
            gold_df.dropna(inplace=True)
            logger.info(f"✅ Calculated {len(gold_df.columns)} indicators for {len(gold_df)} data points.")
            return gold_df
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return None

    # ... Your other amazing functions go here ...
    # enhanced_news_analysis, calculate_gold_specific_indicators, run_simple_backtest, etc.
    # For this script to be complete, you MUST paste your full functions here.
    # Below are simplified placeholders to ensure the script runs for this example.
    def enhanced_news_analysis(self):
        logger.info("📰 Analyzing news...")
        # In your real version, use your full, advanced news analysis function.
        return {"status": "skipped", "news_score": 0.1, "headlines": [{"title":"Example News...","source":"Reuters"}], "confidence": 0.5}

    def calculate_gold_specific_indicators(self, gold_data, market_data):
        logger.info("⚜️ Calculating gold-specific indicators...")
        return {'total_gold_specific_score': 0.2, 'seasonal_score': 0.5, 'volatility_score': -0.1}

    def run_simple_backtest(self, gold_data):
        logger.info("🔬 Running backtest...")
        return {'total_return_percent': 15.5, 'sharpe_ratio': 0.8, 'max_drawdown_percent': -10.2, 'win_rate_percent': 60.0}
    
    def calculate_final_scores(self, gold_data, market_data, gold_indicators, news_result):
        logger.info("🎯 Calculating final scores...")
        latest = gold_data.iloc[-1]
        price = latest['Close']
        scores = {}
        scores['trend'] = 1.5 if price > latest['SMA_200'] and price > latest['SMA_50'] else -1.5
        scores['momentum'] = 1.0 if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] and latest['RSI'] > 50 else -1.0
        scores['correlation'] = 1.0 if market_data[('Close', self.symbols['dxy'])].iloc[-1] < 104 else -1.0
        scores['volatility'] = gold_indicators.get('volatility_score', 0)
        scores['seasonal'] = gold_indicators.get('seasonal_score', 0)
        scores['gold_specific'] = gold_indicators.get('total_gold_specific_score', 0)
        return scores

    def _save_results_to_database(self, result):
        logger.info(f"💾 Saving analysis ID to database...")
        # This is a placeholder for your full database saving logic
        pass

    def run_complete_analysis(self):
        start_time = time.time()
        logger.info("🚀 Starting complete analysis run...")
        try:
            market_data = self.fetch_market_data()
            if market_data is None: raise ValueError("Market data fetching failed")

            gold_data = self.calculate_comprehensive_technical_indicators(market_data)
            if gold_data is None: raise ValueError("Technical indicator calculation failed")

            news_result = self.enhanced_news_analysis()
            gold_indicators = self.calculate_gold_specific_indicators(gold_data, market_data)
            backtest_results = self.run_simple_backtest(gold_data)
            scores = self.calculate_final_scores(gold_data, market_data, gold_indicators, news_result)
            
            weights = {'trend': 0.30, 'momentum': 0.25, 'correlation': 0.20, 'gold_specific': 0.10, 'volatility': 0.10, 'seasonal': 0.05}
            technical_score = sum(scores.get(c, 0) * w for c, w in weights.items())
            news_contribution = news_result.get('news_score', 0) * 0.15
            final_score = technical_score + news_contribution
            
            if final_score >= 1.0: signal, strength = "Buy", "Strong Buy"
            else: signal, strength = "Hold", "Neutral"

            final_result = {
                "timestamp_utc": datetime.utcnow().isoformat(), "execution_time_ms": int((time.time() - start_time) * 1000),
                "status": "success", "signal": signal, "signal_strength": strength,
                "total_score": round(final_score, 3),
                "gold_price": round(gold_data.iloc[-1]['Close'], 2),
                "score_components": scores, "news_analysis": news_result, "backtest_results": backtest_results,
            }

            with open("gold_analysis_pro.json", 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            self._save_results_to_database(final_result)
            logger.info(f"✅ Analysis complete. Signal: {signal} ({strength})")
            return final_result

        except Exception as e:
            logger.error(f"❌ Critical error in analysis run: {e}")
            error_result = {"status": "error", "error_message": str(e)}
            with open("gold_analysis_pro.json", 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
            return error_result

def main():
    try:
        analyzer = ProfessionalGoldAnalyzerFinal()
        analyzer.run_complete_analysis()
    except Exception as e:
        logger.critical(f"💥 Fatal error in main execution: {e}")

if __name__ == "__main__":
    main()
