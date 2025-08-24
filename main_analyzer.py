#!/usr/bin/env python3
"""
Ultra-Lightweight Gold Analyzer for GitHub Actions
Optimized for fast installation and execution without heavy dependencies
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sqlite3
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import sys
import time

# Lightweight imports with fallbacks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Lightweight logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LightweightGoldAnalyzer:
    """Ultra-lightweight Gold Analyzer optimized for GitHub Actions"""
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'spy': 'SPY'
        }
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.db_path = "analysis_history.db"
        self.init_database()
        
        logger.info("🚀 Lightweight Gold Analyzer initialized")

    def init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        analysis_date DATE UNIQUE,
                        gold_price REAL,
                        signal TEXT,
                        confidence TEXT,
                        total_score REAL,
                        rsi REAL,
                        macd REAL,
                        news_sentiment REAL
                    )
                ''')
                conn.commit()
                logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")

    def fetch_market_data(self, period="3mo") -> Optional[pd.DataFrame]:
        """Fetch market data with lightweight approach"""
        logger.info("📊 Fetching market data...")
        
        try:
            # Fetch fewer symbols for speed
            symbols = [self.symbols['gold'], self.symbols['spy']]
            
            data = yf.download(
                symbols,
                period=period,
                interval="1d",
                group_by='ticker',
                progress=False,
                threads=False
            )
            
            if data.empty:
                logger.error("❌ No data received")
                return None
            
            logger.info(f"✅ Fetched data for {len(data)} days")
            return data
            
        except Exception as e:
            logger.error(f"❌ Data fetch failed: {e}")
            return None

    def process_technical_data(self, raw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process technical indicators with lightweight calculations"""
        logger.info("⚙️ Processing technical indicators...")
        
        try:
            # Get gold data
            gold_symbol = self.symbols['gold']
            if isinstance(raw_data.columns, pd.MultiIndex):
                if gold_symbol in raw_data.columns.levels[0]:
                    gold_df = raw_data[gold_symbol].copy()
                else:
                    # Fallback to single symbol data
                    gold_df = raw_data.copy()
            else:
                gold_df = raw_data.copy()
            
            gold_df = gold_df.dropna(subset=['Close'])
            
            if len(gold_df) < 30:
                logger.warning("⚠️ Insufficient data for analysis")
                return None

            # Lightweight technical indicators
            # Simple Moving Averages
            gold_df['SMA_20'] = gold_df['Close'].rolling(20, min_periods=1).mean()
            gold_df['SMA_50'] = gold_df['Close'].rolling(50, min_periods=1).mean()

            # RSI (simplified)
            delta = gold_df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            gold_df['RSI'] = 100 - (100 / (1 + rs))

            # MACD (simplified)
            ema12 = gold_df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = gold_df['Close'].ewm(span=26, adjust=False).mean()
            gold_df['MACD'] = ema12 - ema26
            gold_df['MACD_Signal'] = gold_df['MACD'].ewm(span=9, adjust=False).mean()

            logger.info(f"✅ Processed {len(gold_df)} days of technical data")
            return gold_df.dropna()
            
        except Exception as e:
            logger.error(f"❌ Technical processing failed: {e}")
            return None

    async def fetch_simple_news_sentiment(self) -> float:
        """Simplified news sentiment without heavy NLP libraries"""
        if not self.news_api_key:
            logger.info("ℹ️ News analysis skipped (no API key)")
            return 0.0
        
        logger.info("📰 Fetching simple news sentiment...")
        
        try:
            async with aiohttp.ClientSession() as session:
                url = (f"https://newsapi.org/v2/everything?"
                       f"q=gold+price&language=en&sortBy=publishedAt&pageSize=10&"
                       f"apiKey={self.news_api_key}")
                
                async with session.get(url, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        # Simple sentiment analysis based on keywords
                        positive_keywords = ['rise', 'up', 'gain', 'bull', 'surge', 'rally', 'high']
                        negative_keywords = ['fall', 'down', 'drop', 'bear', 'crash', 'low', 'decline']
                        
                        sentiment_score = 0
                        article_count = 0
                        
                        for article in articles[:5]:  # Limit for speed
                            try:
                                title = article.get('title', '').lower()
                                description = article.get('description', '').lower()
                                text = f"{title} {description}"
                                
                                if text.strip():
                                    positive_count = sum(1 for word in positive_keywords if word in text)
                                    negative_count = sum(1 for word in negative_keywords if word in text)
                                    sentiment_score += (positive_count - negative_count)
                                    article_count += 1
                            except:
                                continue
                        
                        if article_count > 0:
                            avg_sentiment = sentiment_score / article_count
                            logger.info(f"✅ Simple news sentiment: {avg_sentiment:.2f} from {article_count} articles")
                            return avg_sentiment
                        
            return 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ News fetch failed: {e}")
            return 0.0

    def generate_lightweight_signal(self, tech_data: pd.DataFrame, news_sentiment: float = 0) -> Dict:
        """Generate trading signal with lightweight analysis"""
        logger.info("🎯 Generating lightweight trading signal...")
        
        try:
            latest = tech_data.iloc[-1]
            
            # Simplified scoring system
            scores = {}
            
            # Trend analysis (simplified)
            if latest['Close'] > latest['SMA_50']:
                scores['trend'] = 1
            else:
                scores['trend'] = -1
            
            # Momentum analysis (simplified)
            momentum_score = 0
            if latest['MACD'] > latest['MACD_Signal']:
                momentum_score += 1
            else:
                momentum_score -= 1
            
            # RSI analysis
            if latest['RSI'] > 70:
                momentum_score -= 0.5  # Overbought
            elif latest['RSI'] < 30:
                momentum_score += 0.5  # Oversold
            
            scores['momentum'] = momentum_score
            scores['news_sentiment'] = news_sentiment
            
            # Simple weighted calculation
            total_score = (scores['trend'] * 0.5 + 
                          scores['momentum'] * 0.4 + 
                          scores['news_sentiment'] * 0.1)
            
            # Generate signal
            if total_score >= 1.0:
                signal, confidence = "Buy", "High"
            elif total_score > 0.2:
                signal, confidence = "Buy", "Medium"
            elif total_score <= -1.0:
                signal, confidence = "Sell", "High"
            elif total_score < -0.2:
                signal, confidence = "Sell", "Medium"
            else:
                signal, confidence = "Hold", "Low"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'total_score': round(total_score, 2),
                'current_price': round(latest['Close'], 2),
                'rsi': round(latest['RSI'], 1),
                'macd': round(latest['MACD'], 2),
                'news_sentiment': round(news_sentiment, 2),
                'component_scores': scores,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return {"error": str(e)}

    def save_analysis(self, analysis: Dict):
        """Save analysis to database and files"""
        try:
            # Save to database
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                analysis_date = datetime.now().date().isoformat()
                
                conn.execute('''
                    INSERT OR REPLACE INTO analysis_history 
                    (analysis_date, gold_price, signal, confidence, total_score, 
                     rsi, macd, news_sentiment) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_date,
                    analysis.get('current_price'),
                    analysis.get('signal'),
                    analysis.get('confidence'),
                    analysis.get('total_score'),
                    analysis.get('rsi'),
                    analysis.get('macd'),
                    analysis.get('news_sentiment')
                ))
            
            # Save to JSON file
            with open("gold_analysis_v3_enhanced.json", 'w', encoding='utf-8') as f:
                json.dump({
                    'status': 'success',
                    'gold_analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'version': '3.0-Lightweight'
                }, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("✅ Analysis saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")

    def generate_report(self, analysis: Dict) -> str:
        """Generate lightweight analysis report"""
        if 'error' in analysis:
            return f"❌ Analysis failed: {analysis['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("📊 LIGHTWEIGHT GOLD ANALYSIS REPORT")
        report.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append("-" * 60)
        report.append(f"🎯 SIGNAL: {analysis['signal']} ({analysis['confidence']} confidence)")
        report.append(f"💰 Current Price: ${analysis['current_price']}")
        report.append(f"📊 Total Score: {analysis['total_score']}")
        report.append(f"📈 RSI: {analysis['rsi']}")
        report.append(f"📊 MACD: {analysis['macd']}")
        
        if analysis.get('news_sentiment', 0) != 0:
            report.append(f"📰 News Sentiment: {analysis['news_sentiment']:.2f}")
        
        report.append("-" * 60)
        report.append("⚡ Lightweight Mode: Fast execution, core features only")
        report.append("=" * 60)
        
        return "\n".join(report)

    async def run_lightweight_analysis(self):
        """Run lightweight analysis optimized for speed"""
        logger.info("🚀 Starting Lightweight Gold Analysis...")
        
        start_time = time.time()
        
        try:
            # Fetch and process data
            market_data = self.fetch_market_data()
            if market_data is None:
                raise ValueError("Failed to fetch market data")
            
            tech_data = self.process_technical_data(market_data)
            if tech_data is None:
                raise ValueError("Failed to process technical data")
            
            # Get simple news sentiment
            news_sentiment = await self.fetch_simple_news_sentiment()
            
            # Generate analysis
            analysis = self.generate_lightweight_signal(tech_data, news_sentiment)
            
            if 'error' not in analysis:
                # Save results
                self.save_analysis(analysis)
                
                # Generate and print report
                report = self.generate_report(analysis)
                print(report)
                
                # Log execution time
                execution_time = time.time() - start_time
                logger.info(f"✅ Lightweight analysis completed in {execution_time:.2f} seconds")
                
                # GitHub Actions specific output
                if os.getenv('GITHUB_ACTIONS'):
                    github_output = os.getenv('GITHUB_OUTPUT')
                    if github_output:
                        with open(github_output, 'a') as fh:
                            print(f"signal={analysis['signal']}", file=fh)
                            print(f"price={analysis['current_price']}", file=fh)
                            print(f"confidence={analysis['confidence']}", file=fh)
                
            else:
                logger.error(f"❌ Analysis failed: {analysis['error']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error: {e}")
            return False

def main():
    """Main function optimized for GitHub Actions"""
    try:
        if os.getenv('GITHUB_ACTIONS'):
            print("🐙 Running Lightweight Gold Analyzer on GitHub Actions")
        
        analyzer = LightweightGoldAnalyzer()
        success = asyncio.run(analyzer.run_lightweight_analysis())
        
        if success:
            print("🎉 Lightweight analysis completed successfully!")
            sys.exit(0)
        else:
            print("❌ Analysis failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Analysis interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
