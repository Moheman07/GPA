#!/usr/bin/env python3
"""
GitHub-Optimized Gold Analyzer
Specifically designed to run on GitHub Actions with enhanced CI/CD compatibility
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sqlite3
import joblib
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

# GitHub Actions optimized imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    print("⚠️ XGBoost not available, using RandomForest")
    XGB_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    print("✅ TextBlob available")
except ImportError:
    print("⚠️ TextBlob not available, news analysis disabled")
    TEXTBLOB_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("✅ spaCy model loaded")
except (ImportError, OSError):
    print("⚠️ spaCy not available")
    SPACY_AVAILABLE = False

warnings.filterwarnings('ignore')

# GitHub Actions compatible logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GitHubGoldAnalyzer:
    """GitHub Actions optimized Gold Analyzer"""
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'treasury': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY'
        }
        
        # Initialize components
        self.ml_predictor = None
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.db_path = "analysis_history.db"
        self.init_database()
        
        logger.info("🚀 GitHub Gold Analyzer initialized")

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
                        news_sentiment REAL,
                        success_probability REAL
                    )
                ''')
                conn.commit()
                logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")

    def fetch_market_data(self, period="6mo") -> Optional[pd.DataFrame]:
        """Fetch market data with GitHub Actions optimization"""
        logger.info("📊 Fetching market data...")
        
        try:
            # Use shorter period for faster execution in CI/CD
            data = yf.download(
                list(self.symbols.values()),
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
        """Process technical indicators"""
        logger.info("⚙️ Processing technical indicators...")
        
        try:
            # Get gold data
            gold_symbol = self.symbols['gold']
            if isinstance(raw_data.columns, pd.MultiIndex):
                if gold_symbol not in raw_data.columns.levels[0]:
                    gold_symbol = self.symbols['gold_etf']
                gold_df = raw_data[gold_symbol].copy()
            else:
                gold_df = raw_data.copy()
            
            gold_df = gold_df.dropna(subset=['Close'])
            
            if len(gold_df) < 50:
                logger.warning("⚠️ Insufficient data for analysis")
                return None

            # Calculate technical indicators
            # Moving averages
            gold_df['SMA_20'] = gold_df['Close'].rolling(20, min_periods=1).mean()
            gold_df['SMA_50'] = gold_df['Close'].rolling(50, min_periods=1).mean()

            # RSI
            delta = gold_df['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
            loss = -delta.where(delta < 0, 0).ewm(com=13, adjust=False).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            gold_df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = gold_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = gold_df['Close'].ewm(span=26, adjust=False).mean()
            gold_df['MACD'] = exp1 - exp2
            gold_df['MACD_Signal'] = gold_df['MACD'].ewm(span=9, adjust=False).mean()

            logger.info(f"✅ Processed {len(gold_df)} days of technical data")
            return gold_df.dropna()
            
        except Exception as e:
            logger.error(f"❌ Technical processing failed: {e}")
            return None

    async def fetch_news_sentiment(self) -> float:
        """Fetch and analyze news sentiment"""
        if not self.news_api_key or not TEXTBLOB_AVAILABLE:
            logger.info("ℹ️ News analysis skipped (no API key or TextBlob)")
            return 0.0
        
        logger.info("📰 Fetching news sentiment...")
        
        try:
            async with aiohttp.ClientSession() as session:
                url = (f"https://newsapi.org/v2/everything?"
                       f"q=gold+price&language=en&sortBy=publishedAt&pageSize=20&"
                       f"apiKey={self.news_api_key}")
                
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        sentiments = []
                        for article in articles[:10]:  # Limit for speed
                            try:
                                title = article.get('title', '')
                                description = article.get('description', '')
                                text = f"{title} {description}"
                                
                                if text.strip():
                                    sentiment = TextBlob(text).sentiment.polarity
                                    sentiments.append(sentiment)
                            except:
                                continue
                        
                        if sentiments:
                            avg_sentiment = np.mean(sentiments)
                            logger.info(f"✅ News sentiment: {avg_sentiment:.3f} from {len(sentiments)} articles")
                            return avg_sentiment * 5  # Scale to -5 to +5
                        
            return 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ News fetch failed: {e}")
            return 0.0

    def generate_signal(self, tech_data: pd.DataFrame, news_sentiment: float = 0) -> Dict:
        """Generate trading signal"""
        logger.info("🎯 Generating trading signal...")
        
        try:
            latest = tech_data.iloc[-1]
            
            # Component scores
            scores = {}
            
            # Trend analysis
            if latest['Close'] > latest['SMA_50']:
                scores['trend'] = 2
            elif latest['Close'] < latest['SMA_50']:
                scores['trend'] = -2
            else:
                scores['trend'] = 0
            
            # Momentum analysis
            momentum_score = 0
            if latest['MACD'] > latest['MACD_Signal']:
                momentum_score += 1
            else:
                momentum_score -= 1
            
            # RSI analysis
            if latest['RSI'] > 70:
                momentum_score -= 1  # Overbought
            elif latest['RSI'] < 30:
                momentum_score += 1  # Oversold
            elif latest['RSI'] > 60:
                momentum_score += 0.5
            elif latest['RSI'] < 40:
                momentum_score -= 0.5
            
            scores['momentum'] = momentum_score
            scores['news_sentiment'] = news_sentiment
            
            # Calculate total score
            weights = {'trend': 0.4, 'momentum': 0.4, 'news_sentiment': 0.2}
            total_score = sum(scores.get(k, 0) * v for k, v in weights.items())
            
            # Generate signal
            if total_score >= 1.5:
                signal, confidence = "Strong Buy", "High"
            elif total_score > 0.5:
                signal, confidence = "Buy", "Medium"
            elif total_score <= -1.5:
                signal, confidence = "Strong Sell", "High"
            elif total_score < -0.5:
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
                    'version': '3.0-GitHub'
                }, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("✅ Analysis saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")

    def generate_report(self, analysis: Dict) -> str:
        """Generate analysis report"""
        if 'error' in analysis:
            return f"❌ Analysis failed: {analysis['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("📊 GOLD MARKET ANALYSIS REPORT")
        report.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append("-" * 60)
        report.append(f"🎯 SIGNAL: {analysis['signal']} ({analysis['confidence']} confidence)")
        report.append(f"💰 Current Price: ${analysis['current_price']}")
        report.append(f"📊 Total Score: {analysis['total_score']}")
        report.append(f"📈 RSI: {analysis['rsi']}")
        report.append(f"📊 MACD: {analysis['macd']}")
        
        if analysis.get('news_sentiment', 0) != 0:
            report.append(f"📰 News Sentiment: {analysis['news_sentiment']:.2f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)

    async def run_analysis(self):
        """Run complete analysis"""
        logger.info("🚀 Starting GitHub Gold Analysis...")
        
        start_time = time.time()
        
        try:
            # Fetch and process data
            market_data = self.fetch_market_data()
            if market_data is None:
                raise ValueError("Failed to fetch market data")
            
            tech_data = self.process_technical_data(market_data)
            if tech_data is None:
                raise ValueError("Failed to process technical data")
            
            # Get news sentiment
            news_sentiment = await self.fetch_news_sentiment()
            
            # Generate analysis
            analysis = self.generate_signal(tech_data, news_sentiment)
            
            if 'error' not in analysis:
                # Save results
                self.save_analysis(analysis)
                
                # Generate and print report
                report = self.generate_report(analysis)
                print(report)
                
                # Log execution time
                execution_time = time.time() - start_time
                logger.info(f"✅ Analysis completed in {execution_time:.2f} seconds")
                
                # GitHub Actions specific output
                if os.getenv('GITHUB_ACTIONS'):
                    github_output = os.getenv('GITHUB_OUTPUT')
                    if github_output:
                        with open(github_output, 'a') as fh:
                            print(f"signal={analysis['signal']}", file=fh)
                            print(f"price={analysis['current_price']}", file=fh)
                            print(f"confidence={analysis['confidence']}", file=fh)
                    else:
                        # Fallback for older GitHub Actions
                        print(f"::set-output name=signal::{analysis['signal']}")
                        print(f"::set-output name=price::{analysis['current_price']}")
                        print(f"::set-output name=confidence::{analysis['confidence']}")
                
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
        # GitHub Actions environment check
        if os.getenv('GITHUB_ACTIONS'):
            print("🐙 Running on GitHub Actions")
        
        analyzer = GitHubGoldAnalyzer()
        success = asyncio.run(analyzer.run_analysis())
        
        if success:
            print("🎉 Analysis completed successfully!")
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
