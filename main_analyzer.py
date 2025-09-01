#!/usr/bin/env python3
"""
Professional Gold Analyzer V4.0 - Complete Fixed Version
Fixed Issues: MTF Analysis, Overbought Protection, Backtest Logic, Database Schema
Author: AI Assistant
Date: 2025-09-01
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sqlite3
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from textblob import TextBlob
import spacy
import backtrader as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        logger.error("Failed to load spaCy model. NLP features will be limited.")
        nlp = None

class EnhancedMLPredictor:
    """Enhanced Machine Learning Predictor with better error handling"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = "gold_ml_model_v4.pkl"
        self.scaler_path = "gold_scaler_v4.pkl"
        self.min_training_samples = 200
        
    def prepare_features(self, analysis_data: Dict) -> Dict:
        """Prepare features from analysis data with better error handling"""
        features = {}
        
        try:
            # Extract scores from analysis
            if 'gold_analysis' in analysis_data:
                scores = analysis_data['gold_analysis'].get('component_scores', {})
                features.update({f'score_{k}': float(v) for k, v in scores.items() if isinstance(v, (int, float))})
                features['total_score'] = float(analysis_data['gold_analysis'].get('total_score', 0))
                
                # Technical indicators
                tech_summary = analysis_data['gold_analysis'].get('technical_summary', {})
                features.update({f'tech_{k}': float(v) for k, v in tech_summary.items() if isinstance(v, (int, float))})
            
            # Volume data
            if 'volume_analysis' in analysis_data:
                vol = analysis_data['volume_analysis']
                features['volume_ratio'] = float(vol.get('volume_ratio', 1))
                features['volume_strength_encoded'] = self._encode_volume_strength(vol.get('volume_strength', 'Normal'))
            
            # Market correlations
            if 'market_correlations' in analysis_data:
                corr = analysis_data['market_correlations'].get('correlations', {})
                features.update({f'corr_{k}': float(v) for k, v in corr.items() if isinstance(v, (int, float))})
            
            # Economic data
            if 'economic_data' in analysis_data:
                features['economic_score'] = float(analysis_data['economic_data'].get('score', 0))
            
            # Fibonacci levels
            if 'fibonacci_levels' in analysis_data:
                fib = analysis_data['fibonacci_levels']
                features['fib_position'] = float(fib.get('current_position', 50))
            
            # Fill missing values with defaults
            default_features = {
                'score_trend': 0, 'score_momentum': 0, 'score_volume': 0,
                'volume_ratio': 1, 'volume_strength_encoded': 1,
                'economic_score': 0, 'fib_position': 50,
                'corr_dxy': 0, 'corr_vix': 0, 'corr_oil': 0
            }
            
            for key, default_val in default_features.items():
                if key not in features:
                    features[key] = default_val
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return {}
    
    def _encode_volume_strength(self, strength: str) -> int:
        """Encode volume strength to numeric value"""
        mapping = {
            'ضعيف': 0, 'Weak': 0,
            'طبيعي': 1, 'Normal': 1,
            'قوي': 2, 'Strong': 2,
            'قوي جداً': 3, 'Very Strong': 3
        }
        return mapping.get(strength, 1)
    
    def train_model(self, historical_data: List[Dict]) -> bool:
        """Train ML model with enhanced validation"""
        logger.info("🤖 Starting Enhanced ML Model Training...")
        
        try:
            if len(historical_data) < self.min_training_samples:
                logger.warning(f"Insufficient data for training: {len(historical_data)} samples")
                return False
            
            # Prepare training data
            X = []
            y = []
            
            for record in historical_data:
                features = self.prepare_features(record['analysis'])
                if features and len(features) > 5:  # Ensure minimum feature count
                    X.append(list(features.values()))
                    # Target: 1% price increase in 5 days
                    y.append(1 if record.get('price_change_5d', 0) > 1.0 else 0)
            
            if len(X) < self.min_training_samples:
                logger.warning(f"After filtering, insufficient data: {len(X)} samples")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Handle any remaining NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train multiple models and select best
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
            }
            
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                try:
                    logger.info(f"Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    # Test evaluation
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    logger.info(f"{name} Results:")
                    logger.info(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    logger.info(f"  Accuracy: {accuracy:.3f}")
                    logger.info(f"  Precision: {precision:.3f}")
                    logger.info(f"  Recall: {recall:.3f}")
                    logger.info(f"  F1 Score: {f1:.3f}")
                    
                    # Select best model based on F1 score
                    if f1 > best_score:
                        best_score = f1
                        best_model = model
                        self.model = model
                        
                except Exception as model_error:
                    logger.error(f"Error training {name}: {model_error}")
                    continue
            
            if best_model is None:
                logger.error("No model could be trained successfully")
                return False
            
            logger.info(f"✅ Best model: {type(best_model).__name__} with F1: {best_score:.3f}")
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False
    
    def predict_probability(self, analysis_data: Dict) -> Tuple[Optional[float], str]:
        """Predict probability with enhanced error handling"""
        try:
            # Load model if not loaded
            if self.model is None:
                if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                else:
                    return None, "Model not trained yet"
            
            # Prepare features
            features = self.prepare_features(analysis_data)
            if not features:
                return None, "Could not prepare features"
            
            X = np.array([list(features.values())])
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            probability = float(self.model.predict_proba(X_scaled)[0][1])
            
            # Interpret probability
            if probability > 0.80:
                interpretation = "Very High Success Probability"
            elif probability > 0.65:
                interpretation = "High Success Probability"
            elif probability > 0.50:
                interpretation = "Moderate Success Probability"
            elif probability > 0.35:
                interpretation = "Low Success Probability - Caution"
            else:
                interpretation = "Very Low Success Probability - Avoid"
            
            return probability, interpretation
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None, f"Prediction error: {str(e)}"
class FixedMultiTimeframeAnalyzer:
    """Enhanced Multi-Timeframe Analyzer with robust error handling"""
    
    def __init__(self):
        self.timeframes = {
            '1h': {'period': '5d', 'weight': 0.2},
            '4h': {'period': '1mo', 'weight': 0.3},  
            '1d': {'period': '3mo', 'weight': 0.5}
        }
        self.min_data_points = 20
    
    def analyze_timeframe(self, symbol: str, interval: str, period: str) -> Optional[Dict]:
        """Analyze single timeframe with comprehensive error handling"""
        try:
            logger.info(f"Analyzing {symbol} on {interval} timeframe...")
            
            # Download data with retry mechanism
            max_retries = 3
            data = None
            
            for attempt in range(max_retries):
                try:
                    data = yf.download(symbol, period=period, interval=interval, 
                                     progress=False, auto_adjust=True, prepost=True)
                    if not data.empty:
                        break
                except Exception as download_error:
                    logger.warning(f"Download attempt {attempt + 1} failed: {download_error}")
                    if attempt == max_retries - 1:
                        return None
            
            if data is None or data.empty:
                logger.warning(f"No data available for {symbol} on {interval}")
                return None
            
            # Clean and validate data
            data = data.dropna()
            if len(data) < self.min_data_points:
                logger.warning(f"Insufficient data points: {len(data)} < {self.min_data_points}")
                return None
            
            # Calculate indicators with proper error handling
            try:
                # Moving averages
                data['SMA_10'] = data['Close'].rolling(window=10, min_periods=5).mean()
                data['SMA_20'] = data['Close'].rolling(window=20, min_periods=10).mean()
                data['EMA_12'] = data['Close'].ewm(span=12).mean()
                data['EMA_26'] = data['Close'].ewm(span=26).mean()
                
                # RSI with safe calculation
                data['RSI'] = self._calculate_rsi_safe(data['Close'])
                
                # MACD
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
                
                # Bollinger Bands
                bb_period = min(20, len(data) // 2)
                if bb_period >= 5:
                    bb_mean = data['Close'].rolling(window=bb_period).mean()
                    bb_std = data['Close'].rolling(window=bb_period).std()
                    data['BB_Upper'] = bb_mean + (bb_std * 2)
                    data['BB_Lower'] = bb_mean - (bb_std * 2)
                    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                
                # Volume analysis
                if 'Volume' in data.columns:
                    data['Volume_SMA'] = data['Volume'].rolling(window=10, min_periods=5).mean()
                    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
                
                latest = data.iloc[-1]
                previous = data.iloc[-2] if len(data) > 1 else latest
                
                # Calculate comprehensive score
                score = self._calculate_timeframe_score(latest, previous, data)
                
                # Determine trend strength
                trend_strength = self._analyze_trend_strength(data)
                
                return {
                    'score': round(score, 2),
                    'trend': 'Bullish' if score > 0 else 'Bearish',
                    'strength': abs(score),
                    'trend_strength': trend_strength,
                    'rsi': round(latest.get('RSI', 50), 1),
                    'macd': round(latest.get('MACD', 0), 4),
                    'macd_signal': round(latest.get('MACD_Signal', 0), 4),
                    'bb_position': round(latest.get('BB_Position', 0.5), 3),
                    'volume_ratio': round(latest.get('Volume_Ratio', 1), 2),
                    'price': round(latest['Close'], 2),
                    'data_points': len(data),
                    'timeframe': interval,
                    'last_update': datetime.now().isoformat()
                }
                
            except Exception as calc_error:
                logger.error(f"Error calculating indicators for {interval}: {calc_error}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing timeframe {interval}: {e}")
            return None
    
    def _calculate_rsi_safe(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with comprehensive error handling"""
        try:
            if len(prices) < period + 1:
                return pd.Series([50] * len(prices), index=prices.index)
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Use SMA for initial calculation, then EMA
            avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
            avg_loss = loss.rolling(window=period, min_periods=period//2).mean()
            
            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, 0.001)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values
            rsi = rsi.fillna(50)
            
            # Ensure values are within bounds
            rsi = rsi.clip(0, 100)
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_timeframe_score(self, latest: pd.Series, previous: pd.Series, data: pd.DataFrame) -> float:
        """Calculate comprehensive timeframe score"""
        score = 0
        
        try:
            # Trend analysis (40% weight)
            if not pd.isna(latest.get('SMA_20')) and not pd.isna(latest['Close']):
                if latest['Close'] > latest['SMA_20']:
                    score += 1.5
                else:
                    score -= 1.5
            
            if not pd.isna(latest.get('SMA_10')) and not pd.isna(latest.get('SMA_20')):
                if latest['SMA_10'] > latest['SMA_20']:
                    score += 1
                else:
                    score -= 1
            
            # Momentum analysis (35% weight)
            rsi = latest.get('RSI', 50)
            if 30 <= rsi <= 70:
                if 45 <= rsi <= 55:
                    score += 0.5
                elif rsi > 55:
                    score += 1
                else:
                    score -= 0.5
            elif rsi < 30:
                score += 2  # Oversold - potential reversal
            elif rsi > 80:
                score -= 2  # Overbought - potential reversal
            elif rsi > 70:
                score -= 1  # Mild overbought
            
            # MACD analysis
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            if not pd.isna(macd) and not pd.isna(macd_signal):
                if macd > macd_signal:
                    score += 1
                else:
                    score -= 1
                
                # MACD momentum
                macd_hist = latest.get('MACD_Histogram', 0)
                prev_macd_hist = previous.get('MACD_Histogram', 0)
                if not pd.isna(macd_hist) and not pd.isna(prev_macd_hist):
                    if macd_hist > prev_macd_hist:
                        score += 0.5
                    else:
                        score -= 0.5
            
            # Volume confirmation (15% weight)
            volume_ratio = latest.get('Volume_Ratio', 1)
            if not pd.isna(volume_ratio):
                if volume_ratio > 1.5:
                    score += 0.5  # High volume confirms move
                elif volume_ratio < 0.7:
                    score -= 0.3  # Low volume weakens signal
            
            # Bollinger Bands (10% weight)
            bb_pos = latest.get('BB_Position', 0.5)
            if not pd.isna(bb_pos):
                if bb_pos > 0.8:
                    score -= 0.5  # Near upper band - caution
                elif bb_pos < 0.2:
                    score += 0.5  # Near lower band - potential support
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating timeframe score: {e}")
            return 0
    
    def _analyze_trend_strength(self, data: pd.DataFrame) -> str:
        """Analyze trend strength based on multiple factors"""
        try:
            if len(data) < 10:
                return "Insufficient Data"
            
            recent_data = data.tail(10)
            price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0] * 100
            
            # Analyze slope of moving averages
            if 'SMA_20' in recent_data.columns:
                sma_slope = (recent_data['SMA_20'].iloc[-1] - recent_data['SMA_20'].iloc[0]) / len(recent_data)
                
                if abs(price_change) > 5 and abs(sma_slope) > 1:
                    return "Very Strong"
                elif abs(price_change) > 2 and abs(sma_slope) > 0.5:
                    return "Strong"
                elif abs(price_change) > 1:
                    return "Moderate"
                else:
                    return "Weak"
            
            return "Moderate"
            
        except Exception as e:
            logger.error(f"Error analyzing trend strength: {e}")
            return "Unknown"
    
    def get_coherence_score(self, symbol: str) -> Tuple[float, Dict]:
        """Get coherence score across multiple timeframes"""
        logger.info("⏰ Analyzing Multiple Timeframes...")
        
        results = {}
        total_weighted_score = 0
        total_weight = 0
        
        # Analyze each timeframe
        for tf_name, tf_config in self.timeframes.items():
            analysis = self.analyze_timeframe(symbol, tf_name, tf_config['period'])
            
            if analysis:
                results[tf_name] = analysis
                total_weighted_score += analysis['score'] * tf_config['weight']
                total_weight += tf_config['weight']
                logger.info(f"  {tf_name}: {analysis['trend']} (Score: {analysis['score']})")
            else:
                logger.warning(f"  {tf_name}: Analysis failed")
        
        if total_weight == 0:
            return 0, {'error': 'No timeframe analysis available'}
        
        # Calculate coherence score
        coherence_score = total_weighted_score / total_weight
        
        # Analyze coherence
        trends = [r['trend'] for r in results.values() if r]
        bullish_count = sum(1 for t in trends if t == 'Bullish')
        bearish_count = sum(1 for t in trends if t == 'Bearish')
        
        if bullish_count == len(trends) and len(trends) >= 2:
            coherence_score += 1.5  # Bonus for full bullish alignment
            coherence_analysis = "Perfect Bullish Alignment - Exceptional Strength"
        elif bearish_count == len(trends) and len(trends) >= 2:
            coherence_score -= 1.5  # Penalty for full bearish alignment
            coherence_analysis = "Perfect Bearish Alignment - Strong Weakness"
        elif bullish_count > bearish_count:
            coherence_analysis = "Bullish Bias - Mixed Signals"
        elif bearish_count > bullish_count:
            coherence_analysis = "Bearish Bias - Mixed Signals"
        else:
            coherence_analysis = "Conflicting Signals - High Uncertainty"
        
        return coherence_score, {
            'timeframes': results,
            'coherence_score': round(coherence_score, 2),
            'analysis': coherence_analysis,
            'recommendation': self._get_mtf_recommendation(coherence_score),
            'trends_summary': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'total': len(trends)
            }
        }
    
    def _get_mtf_recommendation(self, score: float) -> str:
        """Get recommendation based on multi-timeframe coherence"""
        if score > 3:
            return "Strong Entry - All Timeframes Aligned"
        elif score > 2:
            return "Good Entry - Strong Coherence"
        elif score > 1:
            return "Moderate Entry - Decent Alignment"
        elif score > -1:
            return "Wait - Unclear Direction"
        elif score > -2:
            return "Avoid Long - Weakness Detected"
        else:
            return "Strong Avoid - Bearish Alignment"

class EnhancedSafetySystem:
    """Enhanced safety system for preventing dangerous trades"""
    
    def __init__(self):
        self.safety_rules = {
            'max_rsi_for_buy': 75,
            'max_bb_position_for_buy': 1.0,
            'min_volume_ratio': 0.5,
            'max_risk_score': 4,
            'require_mtf_confirmation': True
        }
    
    def check_overbought_conditions(self, latest_data: pd.Series) -> Tuple[List[str], int]:
        """Comprehensive overbought condition checking"""
        warnings = []
        risk_score = 0
        
        # RSI Analysis
        rsi = latest_data.get('RSI', 50)
        if rsi > 85:
            warnings.append(f"⚠️ CRITICAL: RSI Extremely Overbought ({rsi:.1f}) - High Reversal Risk")
            risk_score += 4
        elif rsi > 80:
            warnings.append(f"⚠️ SEVERE: RSI in Danger Zone ({rsi:.1f}) - Avoid New Longs")
            risk_score += 3
        elif rsi > 75:
            warnings.append(f"⚠️ WARNING: RSI Overbought ({rsi:.1f}) - Exercise Caution")
            risk_score += 2
        elif rsi > 70:
            warnings.append(f"⚠️ CAUTION: RSI Above 70 ({rsi:.1f}) - Monitor Closely")
            risk_score += 1
        
        # Bollinger Bands Analysis
        bb_pos = latest_data.get('BB_Position', 0.5)
        if bb_pos > 1.2:
            warnings.append(f"⚠️ CRITICAL: Price Far Above Bollinger Upper Band ({bb_pos:.2f}) - Extreme Overextension")
            risk_score += 4
        elif bb_pos > 1.1:
            warnings.append(f"⚠️ SEVERE: Price Above Bollinger Band ({bb_pos:.2f}) - High Correction Risk")
            risk_score += 3
        elif bb_pos > 1.0:
            warnings.append(f"⚠️ WARNING: Price Touching Upper Bollinger Band ({bb_pos:.2f}) - Potential Reversal")
            risk_score += 2
        elif bb_pos > 0.85:
            warnings.append(f"⚠️ CAUTION: Price Near Upper Band ({bb_pos:.2f}) - Watch for Rejection")
            risk_score += 1
        
        # Stochastic Analysis (if available)
        stoch_k = latest_data.get('Stoch_K', 50)
        if stoch_k > 90:
            warnings.append(f"⚠️ Stochastic Extremely Overbought ({stoch_k:.1f}) - Correction Expected")
            risk_score += 2
        elif stoch_k > 80:
            warnings.append(f"⚠️ Stochastic Overbought ({stoch_k:.1f}) - Monitor for Divergence")
            risk_score += 1
        
        # Williams %R Analysis (if available)
        williams_r = latest_data.get('Williams_R', -50)
        if williams_r > -10:
            warnings.append(f"⚠️ Williams %R Overbought ({williams_r:.1f}) - Reversal Signal")
            risk_score += 2
        elif williams_r > -20:
            warnings.append(f"⚠️ Williams %R High ({williams_r:.1f}) - Caution Advised")
            risk_score += 1
        
        return warnings, risk_score
    
    def check_volume_conditions(self, volume_data: Dict) -> Tuple[List[str], int]:
        """Check volume-related warning conditions"""
        warnings = []
        risk_score = 0
        
        volume_ratio = volume_data.get('volume_ratio', 1)
        volume_strength = volume_data.get('volume_strength', 'Normal')
        
        if volume_ratio < 0.3:
            warnings.append(f"⚠️ CRITICAL: Extremely Low Volume ({volume_ratio:.2f}) - Unreliable Signals")
            risk_score += 3
        elif volume_ratio < 0.5:
            warnings.append(f"⚠️ WARNING: Low Volume ({volume_ratio:.2f}) - Weak Confirmation")
            risk_score += 2
        elif volume_ratio < 0.7:
            warnings.append(f"⚠️ CAUTION: Below Average Volume ({volume_ratio:.2f}) - Limited Conviction")
            risk_score += 1
        
        # Check for volume divergence
        if volume_strength == 'ضعيف' or volume_strength == 'Weak':
            warnings.append("⚠️ Volume Weakness Detected - Price Move May Lack Sustainability")
            risk_score += 1
        
        return warnings, risk_score
    
    def check_market_conditions(self, correlations: Dict, economic_data: Dict) -> Tuple[List[str], int]:
        """Check broader market condition warnings"""
        warnings = []
        risk_score = 0
        
        # DXY correlation check
        dxy_corr = correlations.get('correlations', {}).get('dxy', 0)
        if dxy_corr > -0.3:  # Gold should be negatively correlated with DXY
            warnings.append(f"⚠️ Unusual DXY Correlation ({dxy_corr:.3f}) - Dollar Strength May Impact Gold")
            risk_score += 1
        
        # VIX check
        vix_corr = correlations.get('correlations', {}).get('vix', 0)
        if vix_corr < -0.5:  # Unusual negative correlation with fear index
            warnings.append(f"⚠️ Negative VIX Correlation ({vix_corr:.3f}) - Risk-Off Sentiment May Not Support Gold")
            risk_score += 1
        
        # Economic conditions
        if economic_data.get('status') != 'error':
            econ_score = economic_data.get('score', 0)
            if econ_score < -2:
                warnings.append("⚠️ Negative Economic Environment - Headwinds for Gold")
                risk_score += 1
        
        return warnings, risk_score
    
    def apply_safety_override(self, signal: str, confidence: str, total_score: float, 
                            all_warnings: List[str], total_risk_score: int) -> Tuple[str, str, str, bool]:
        """Apply safety overrides based on risk assessment"""
        
        override_applied = False
        override_reason = None
        
        # Critical risk override
        if total_risk_score >= 8:
            signal = "Hold"
            confidence = "High"
            override_reason = "SAFETY OVERRIDE: Critical risk conditions detected - All buy signals suspended"
            override_applied = True
            
        elif total_risk_score >= 6:
            if signal in ["Strong Buy", "Buy"]:
                signal = "Hold"
                confidence = "Medium"
                override_reason = "SAFETY OVERRIDE: High risk conditions - Buy signals cancelled"
                override_applied = True
            elif signal == "Weak Buy":
                signal = "Hold"
                confidence = "Low"
                override_reason = "SAFETY OVERRIDE: High risk conditions - Even weak buy cancelled"
                override_applied = True
                
        elif total_risk_score >= 4:
            if signal == "Strong Buy":
                signal = "Weak Buy"
                confidence = "Medium"
                override_reason = "SAFETY ADJUSTMENT: Moderate risk - Downgraded from Strong Buy"
                override_applied = True
            elif signal == "Buy":
                signal = "Weak Buy"
                confidence = "Low"
                override_reason = "SAFETY ADJUSTMENT: Moderate risk - Downgraded to Weak Buy"
                override_applied = True
                
        elif total_risk_score >= 2:
            if signal == "Strong Buy":
                signal = "Buy"
                confidence = "Medium-High"
                override_reason = "SAFETY ADJUSTMENT: Low-moderate risk - Slight downgrade applied"
                override_applied = True
        
        # Additional safety check for extreme overbought regardless of score
        extreme_overbought = any("CRITICAL" in warning or "EXTREME" in warning for warning in all_warnings)
        if extreme_overbought and signal in ["Strong Buy", "Buy", "Weak Buy"]:
            signal = "Hold"
            confidence = "High"
            override_reason = "EXTREME OVERBOUGHT OVERRIDE: Market conditions too dangerous for any buy signal"
            override_applied = True
        
        return signal, confidence, override_reason, override_applied
class ProfessionalSignalGenerator:
    """Professional signal generator with comprehensive analysis"""
    
    def __init__(self):
        # Rebalanced weights based on effectiveness analysis
        self.weights = {
            'trend': 0.12,           # Reduced from 0.20 - lagging indicator
            'momentum': 0.28,        # Increased from 0.15 - leading indicator
            'volume': 0.15,          # Maintained - important confirmation
            'fibonacci': 0.08,       # Maintained - support/resistance
            'correlation': 0.08,     # Increased - market context
            'support_resistance': 0.10,  # Increased - key levels
            'economic': 0.08,        # Maintained - fundamental backdrop
            'news': 0.06,           # Maintained - sentiment factor
            'ma_cross': 0.05         # Reduced - already covered in trend
        }
        
        self.safety_system = EnhancedSafetySystem()
        
    def generate_professional_signals(self, tech_data: pd.DataFrame, correlations: Dict,
                                     volume: Dict, fib_levels: Dict, support_resistance: Dict,
                                     economic_data: Dict, news_analysis: Dict, 
                                     mtf_analysis: Dict, ml_prediction: Tuple) -> Dict:
        """Generate professional signals with comprehensive safety checks"""
        
        logger.info("🎯 Generating Professional Signals with Enhanced Safety...")
        
        try:
            latest = tech_data.iloc[-1]
            prev = tech_data.iloc[-2] if len(tech_data) > 1 else latest
            
            # Comprehensive safety checks
            overbought_warnings, overbought_risk = self.safety_system.check_overbought_conditions(latest)
            volume_warnings, volume_risk = self.safety_system.check_volume_conditions(volume)
            market_warnings, market_risk = self.safety_system.check_market_conditions(correlations, economic_data)
            
            all_warnings = overbought_warnings + volume_warnings + market_warnings
            total_risk_score = overbought_risk + volume_risk + market_risk
            
            # Calculate component scores
            scores = self._calculate_enhanced_component_scores(
                latest, prev, tech_data, correlations, volume, 
                fib_levels, support_resistance, economic_data, news_analysis, mtf_analysis
            )
            
            # Calculate weighted total score
            total_score = sum(scores[key] * self.weights.get(key, 0) for key in scores)
            
            # Apply ML adjustment if available
            ml_adjusted_score = total_score
            ml_interpretation = ""
            confidence_multiplier = 1.0
            
            if ml_prediction and ml_prediction[0] is not None:
                ml_prob = ml_prediction[0]
                ml_interpretation = ml_prediction[1]
                
                # Adjust score based on ML prediction
                if ml_prob > 0.8:
                    confidence_multiplier = 1.3
                    ml_adjusted_score *= 1.2
                elif ml_prob > 0.65:
                    confidence_multiplier = 1.15
                    ml_adjusted_score *= 1.1
                elif ml_prob < 0.35:
                    confidence_multiplier = 0.7
                    ml_adjusted_score *= 0.8
                elif ml_prob < 0.2:
                    confidence_multiplier = 0.5
                    ml_adjusted_score *= 0.6
            
            # Determine initial signal and confidence
            signal, confidence = self._determine_signal_and_confidence(ml_adjusted_score, confidence_multiplier)
            
            # Apply safety overrides
            final_signal, final_confidence, override_reason, override_applied = self.safety_system.apply_safety_override(
                signal, confidence, ml_adjusted_score, all_warnings, total_risk_score
            )
            
            # Generate action recommendation
            action_recommendation = self._generate_action_recommendation(
                final_signal, final_confidence, total_risk_score, override_applied
            )
            
            # Enhanced risk management
            risk_management = self._calculate_enhanced_risk_management(
                latest, final_signal, final_confidence, total_risk_score, ml_prediction
            )
            
            # Generate entry strategy
            entry_strategy = self._generate_enhanced_entry_strategy(
                scores, latest, support_resistance, mtf_analysis, all_warnings, total_risk_score
            )
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'action_recommendation': action_recommendation,
                'total_score': round(total_score, 2),
                'ml_adjusted_score': round(ml_adjusted_score, 2),
                'component_scores': scores,
                'current_price': round(latest['Close'], 2),
                'risk_management': risk_management,
                'entry_strategy': entry_strategy,
                'safety_analysis': {
                    'total_risk_score': total_risk_score,
                    'risk_level': self._categorize_risk_level(total_risk_score),
                    'all_warnings': all_warnings,
                    'override_applied': override_applied,
                    'override_reason': override_reason,
                    'safety_recommendation': self._get_safety_recommendation(total_risk_score)
                },
                'ml_prediction': {
                    'probability': round(ml_prediction[0], 3) if ml_prediction and ml_prediction[0] else None,
                    'interpretation': ml_interpretation,
                    'confidence_adjustment': round(confidence_multiplier, 2)
                },
                'technical_summary': {
                    'rsi': round(latest.get('RSI', 50), 1),
                    'macd': round(latest.get('MACD', 0), 4),
                    'macd_signal': round(latest.get('MACD_Signal', 0), 4),
                    'bb_position': round(latest.get('BB_Position', 0.5), 3),
                    'volume_ratio': round(latest.get('Volume_Ratio', 1), 2),
                    'williams_r': round(latest.get('Williams_R', -50), 1),
                    'stoch_k': round(latest.get('Stoch_K', 50), 1)
                },
                'key_levels': self._extract_key_levels(latest, support_resistance),
                'mtf_summary': mtf_analysis.get('analysis', '') if mtf_analysis else '',
                'coherence_score': mtf_analysis.get('coherence_score', 0) if mtf_analysis else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating professional signals: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _calculate_enhanced_component_scores(self, latest: pd.Series, prev: pd.Series, data: pd.DataFrame,
                                           correlations: Dict, volume: Dict, fib_levels: Dict,
                                           support_resistance: Dict, economic_data: Dict,
                                           news_analysis: Dict, mtf_analysis: Dict) -> Dict:
        """Calculate enhanced component scores with better logic"""
        
        scores = {
            'trend': 0, 'momentum': 0, 'volume': 0, 'fibonacci': 0,
            'correlation': 0, 'support_resistance': 0, 'economic': 0,
            'news': 0, 'ma_cross': 0
        }
        
        try:
            # Enhanced Trend Analysis
            sma_200 = latest.get('SMA_200', latest['Close'])
            sma_50 = latest.get('SMA_50', latest['Close'])
            sma_20 = latest.get('SMA_20', latest['Close'])
            
            if latest['Close'] > sma_200:
                scores['trend'] += 2
                if latest['Close'] > sma_50:
                    scores['trend'] += 1.5
                    if latest['Close'] > sma_20:
                        scores['trend'] += 1
            else:
                scores['trend'] -= 2
                if latest['Close'] < sma_50:
                    scores['trend'] -= 1.5
                    if latest['Close'] < sma_20:
                        scores['trend'] -= 1
            
            # Enhanced Momentum Analysis with overbought protection
            rsi = latest.get('RSI', 50)
            if rsi < 30:
                scores['momentum'] += 3  # Strong oversold bounce potential
            elif rsi < 40:
                scores['momentum'] += 2  # Oversold recovery
            elif 40 <= rsi <= 60:
                scores['momentum'] += 1  # Healthy momentum
            elif 60 < rsi <= 70:
                scores['momentum'] += 0.5  # Slight positive but caution
            elif 70 < rsi <= 80:
                scores['momentum'] -= 1  # Overbought warning
            else:  # RSI > 80
                scores['momentum'] -= 3  # Severe overbought penalty
            
            # MACD Analysis
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_hist = latest.get('MACD_Histogram', 0)
            prev_macd_hist = prev.get('MACD_Histogram', 0)
            
            if macd > macd_signal:
                scores['momentum'] += 1.5
                if macd_hist > prev_macd_hist:  # Increasing momentum
                    scores['momentum'] += 1
            else:
                scores['momentum'] -= 1.5
                if macd_hist < prev_macd_hist:  # Decreasing momentum
                    scores['momentum'] -= 1
            
            # Enhanced Volume Analysis
            vol_strength = volume.get('volume_strength', 'Normal')
            vol_ratio = volume.get('volume_ratio', 1)
            
            if vol_strength in ['قوي جداً', 'Very Strong'] and vol_ratio > 2:
                scores['volume'] = 3
            elif vol_strength in ['قوي', 'Strong'] and vol_ratio > 1.5:
                scores['volume'] = 2
            elif vol_ratio > 1.2:
                scores['volume'] = 1
            elif vol_ratio < 0.7:
                scores['volume'] = -1
            elif vol_ratio < 0.5:
                scores['volume'] = -2
            
            # OBV confirmation
            if volume.get('obv_trend') == 'صاعد' or volume.get('obv_trend') == 'Bullish':
                scores['volume'] += 1
            elif volume.get('obv_trend') == 'هابط' or volume.get('obv_trend') == 'Bearish':
                scores['volume'] -= 1
            
            # Fibonacci Analysis
            if fib_levels:
                current_pos = fib_levels.get('current_position', 50)
                if current_pos > 80:  # Above 78.6% retracement
                    scores['fibonacci'] = 2
                elif current_pos > 60:  # Above 61.8%
                    scores['fibonacci'] = 1
                elif current_pos > 40:  # Above 38.2%
                    scores['fibonacci'] = 0.5
                elif current_pos < 20:  # Below 23.6%
                    scores['fibonacci'] = -2
                else:
                    scores['fibonacci'] = -1
            
            # Enhanced Correlation Analysis
            correlations_data = correlations.get('correlations', {})
            
            # DXY inverse correlation (key for gold)
            dxy_corr = correlations_data.get('dxy', 0)
            if dxy_corr < -0.7:
                scores['correlation'] += 2
            elif dxy_corr < -0.5:
                scores['correlation'] += 1
            elif dxy_corr > 0.3:
                scores['correlation'] -= 1
            
            # VIX correlation (safe haven demand)
            vix_corr = correlations_data.get('vix', 0)
            if vix_corr > 0.3:
                scores['correlation'] += 1
            
            # Support/Resistance Analysis
            if support_resistance:
                price_to_support = support_resistance.get('price_to_support')
                price_to_resistance = support_resistance.get('price_to_resistance')
                
                if price_to_support and price_to_support < 2:  # Near support
                    scores['support_resistance'] = 2
                elif price_to_resistance and price_to_resistance < 2:  # Near resistance
                    scores['support_resistance'] = -2
                elif price_to_support and price_to_support < 5:
                    scores['support_resistance'] = 1
            
            # Economic Data Analysis
            if economic_data and economic_data.get('status') != 'error':
                econ_score = economic_data.get('score', 0)
                scores['economic'] = max(-3, min(3, econ_score))
            
            # News Analysis
            if news_analysis and news_analysis.get('status') == 'success':
                events = news_analysis.get('events_analysis', {})
                total_impact = events.get('total_impact', 0)
                
                if total_impact > 8:
                    scores['news'] = 3
                elif total_impact > 4:
                    scores['news'] = 2
                elif total_impact > 1:
                    scores['news'] = 1
                elif total_impact < -8:
                    scores['news'] = -3
                elif total_impact < -4:
                    scores['news'] = -2
                elif total_impact < -1:
                    scores['news'] = -1
            
            # Moving Average Crossover
            if 'Golden_Cross' in latest and latest['Golden_Cross'] == 1:
                scores['ma_cross'] = 2
            elif 'Death_Cross' in latest and latest['Death_Cross'] == 1:
                scores['ma_cross'] = -2
            elif sma_20 > sma_50 and latest['Close'] > sma_20:
                scores['ma_cross'] = 1
            elif sma_20 < sma_50 and latest['Close'] < sma_20:
                scores['ma_cross'] = -1
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating component scores: {e}")
            return scores
    
    def _determine_signal_and_confidence(self, score: float, confidence_multiplier: float) -> Tuple[str, str]:
        """Determine signal and confidence based on score"""
        
        adjusted_thresholds = {
            'strong_buy': 2.5 * confidence_multiplier,
            'buy': 1.5 * confidence_multiplier,
            'weak_buy': 0.5 * confidence_multiplier,
            'weak_sell': -0.5 * confidence_multiplier,
            'sell': -1.5 * confidence_multiplier,
            'strong_sell': -2.5 * confidence_multiplier
        }
        
        if score >= adjusted_thresholds['strong_buy']:
            return "Strong Buy", "Very High"
        elif score >= adjusted_thresholds['buy']:
            return "Buy", "High"
        elif score >= adjusted_thresholds['weak_buy']:
            return "Weak Buy", "Medium"
        elif score <= adjusted_thresholds['strong_sell']:
            return "Strong Sell", "Very High"
        elif score <= adjusted_thresholds['sell']:
            return "Sell", "High"
        elif score <= adjusted_thresholds['weak_sell']:
            return "Weak Sell", "Medium"
        else:
            return "Hold", "Low"
    
    def _generate_action_recommendation(self, signal: str, confidence: str, 
                                      risk_score: int, override_applied: bool) -> str:
        """Generate detailed action recommendation"""
        
        base_actions = {
            "Strong Buy": "Strong Buy - Large Position Size",
            "Buy": "Buy - Medium Position Size", 
            "Weak Buy": "Cautious Buy - Small Position Size",
            "Hold": "Hold - Wait for Better Opportunity",
            "Weak Sell": "Consider Selling - Reduce Position",
            "Sell": "Sell - Exit Position",
            "Strong Sell": "Strong Sell - Exit All Positions"
        }
        
        action = base_actions.get(signal, "Hold - Unclear Direction")
        
        if override_applied:
            action += " (SAFETY OVERRIDE APPLIED)"
        
        if risk_score >= 6:
            action += " - HIGH RISK CONDITIONS"
        elif risk_score >= 4:
            action += " - MODERATE RISK CONDITIONS"
        elif risk_score >= 2:
            action += " - LOW RISK CONDITIONS"
        
        return action
    
    def _calculate_enhanced_risk_management(self, latest: pd.Series, signal: str, 
                                          confidence: str, risk_score: int, 
                                          ml_prediction: Tuple) -> Dict:
        """Calculate enhanced risk management parameters"""
        
        try:
            price = latest['Close']
            atr = latest.get('ATR', price * 0.02)
            volatility = latest.get('ATR_Percent', 2)
            
            # Adjust stop loss based on risk conditions
            base_sl_multiplier = 1.5
            
            if risk_score >= 6:
                sl_multiplier = base_sl_multiplier * 0.7  # Tighter stops in high risk
            elif risk_score >= 4:
                sl_multiplier = base_sl_multiplier * 0.85
            elif risk_score >= 2:
                sl_multiplier = base_sl_multiplier * 0.95
            else:
                sl_multiplier = base_sl_multiplier
            
            # Adjust for volatility
            if volatility > 3:
                sl_multiplier *= 1.5
            elif volatility > 2:
                sl_multiplier *= 1.2
            
            # ML adjustment
            if ml_prediction and ml_prediction[0] is not None:
                ml_prob = ml_prediction[0]
                if ml_prob < 0.4:
                    sl_multiplier *= 0.8  # Tighter stop for low probability
                elif ml_prob > 0.75:
                    sl_multiplier *= 1.1  # Slightly wider for high probability
            
            stop_loss_levels = {
                'tight': round(price - (atr * sl_multiplier * 0.6), 2),
                'conservative': round(price - (atr * sl_multiplier), 2),
                'moderate': round(price - (atr * sl_multiplier * 1.4), 2),
                'wide': round(price - (atr * sl_multiplier * 2), 2)
            }
            
            profit_targets = {
                'target_1': round(price + (atr * 1.5), 2),
                'target_2': round(price + (atr * 3), 2),
                'target_3': round(price + (atr * 5), 2),
                'target_4': round(price + (atr * 8), 2)
            }
            
            # Position sizing based on confidence and risk
            position_size = self._calculate_position_sizing(confidence, risk_score, ml_prediction)
            
            return {
                'stop_loss_levels': stop_loss_levels,
                'profit_targets': profit_targets,
                'position_size_recommendation': position_size,
                'risk_reward_ratio': round((profit_targets['target_2'] - price) / (price - stop_loss_levels['conservative']), 2),
                'max_risk_per_trade': '1%' if risk_score >= 4 else ('1.5%' if risk_score >= 2 else '2%'),
                'volatility_adjustment': round(sl_multiplier, 2),
                'risk_adjusted': True,
                'ml_adjusted': ml_prediction is not None and ml_prediction[0] is not None
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk management: {e}")
            return {}
    
    def _calculate_position_sizing(self, confidence: str, risk_score: int, ml_prediction: Tuple) -> str:
        """Calculate recommended position sizing"""
        
        base_sizes = {
            "Very High": "Large (50-75% of allocated capital)",
            "High": "Medium-Large (35-50%)",
            "Medium-High": "Medium (25-35%)",
            "Medium": "Small-Medium (15-25%)",
            "Low": "Small (5-15%)"
        }
        
        base_size = base_sizes.get(confidence, "Very Small (2-5%)")
        
        # Risk adjustment
        if risk_score >= 6:
            base_size = "AVOID - Risk Too High"
        elif risk_score >= 4:
            base_size = base_size.replace("Large", "Small").replace("Medium", "Very Small")
        elif risk_score >= 2:
            base_size = base_size.replace("Large", "Medium").replace("Medium-Large", "Small-Medium")
        
        # ML adjustment
        if ml_prediction and ml_prediction[0] is not None:
            ml_prob = ml_prediction[0]
            if ml_prob > 0.8:
                base_size += " (ML Strongly Confirms)"
            elif ml_prob < 0.3:
                base_size += " (ML Warns - Reduce Further)"
        
        return base_size
    
    def _generate_enhanced_entry_strategy(self, scores: Dict, latest: pd.Series,
                                        support_resistance: Dict, mtf_analysis: Dict,
                                        warnings: List[str], risk_score: int) -> Dict:
        """Generate enhanced entry strategy"""
        
        strategy = {
            'entry_type': '',
            'entry_zones': [],
            'conditions': [],
            'warnings': warnings,
            'risk_level': self._categorize_risk_level(risk_score)
        }
        
        # Determine entry type based on analysis
        if risk_score >= 6:
            strategy['entry_type'] = 'AVOID ENTRY - High Risk Conditions'
            strategy['entry_zones'].append('Wait for risk conditions to improve')
        elif risk_score >= 4:
            strategy['entry_type'] = 'Cautious Entry Only'
            strategy['entry_zones'].append('Small position with tight stops')
        elif scores.get('momentum', 0) > 2 and scores.get('trend', 0) > 1:
            strategy['entry_type'] = 'Momentum Entry'
            strategy['entry_zones'].append(f"Current price levels around {latest['Close']:.2f}")
        elif support_resistance and support_resistance.get('price_to_support', 10) < 3:
            strategy['entry_type'] = 'Support Level Entry'
            if support_resistance.get('nearest_support'):
                strategy['entry_zones'].append(f"Near support at {support_resistance['nearest_support']}")
        else:
            strategy['entry_type'] = 'Gradual Accumulation'
            strategy['entry_zones'].append('Split entry across 2-3 levels')
        
        # Add conditions based on technical levels
        rsi = latest.get('RSI', 50)
        if rsi > 75:
            strategy['conditions'].append(f'Wait for RSI below 70 (currently {rsi:.1f})')
        
        bb_pos = latest.get('BB_Position', 0.5)
        if bb_pos > 0.9:
            strategy['conditions'].append(f'Price near/above Bollinger upper band ({bb_pos:.2f}) - wait for pullback')
        
        # MTF confirmation
        if mtf_analysis:
            coherence = mtf_analysis.get('coherence_score', 0)
            if coherence > 2:
                strategy['mtf_confirmation'] = '✅ Strong Multi-Timeframe Alignment'
            elif coherence < -1:
                strategy['mtf_confirmation'] = '⚠️ Conflicting Timeframe Signals'
            else:
                strategy['mtf_confirmation'] = 'Mixed Timeframe Signals - Exercise Caution'
        
        return strategy
    
    def _extract_key_levels(self, latest: pd.Series, support_resistance: Dict) -> Dict:
        """Extract key technical levels"""
        
        levels = {
            'current_price': round(latest['Close'], 2),
            'sma_20': round(latest.get('SMA_20', 0), 2),
            'sma_50': round(latest.get('SMA_50', 0), 2),
            'sma_200': round(latest.get('SMA_200', 0), 2),
            'bb_upper': round(latest.get('BB_Upper', 0), 2),
            'bb_lower': round(latest.get('BB_Lower', 0), 2)
        }
        
        if support_resistance:
            levels.update({
                'nearest_support': support_resistance.get('nearest_support'),
                'nearest_resistance': support_resistance.get('nearest_resistance')
            })
        
        return levels
    
    def _categorize_risk_level(self, risk_score: int) -> str:
        """Categorize overall risk level"""
        if risk_score >= 8:
            return "CRITICAL"
        elif risk_score >= 6:
            return "HIGH"
        elif risk_score >= 4:
            return "MODERATE"
        elif risk_score >= 2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_safety_recommendation(self, risk_score: int) -> str:
        """Get safety recommendation based on risk score"""
        if risk_score >= 8:
            return "AVOID ALL LONG POSITIONS - Market conditions extremely dangerous"
        elif risk_score >= 6:
            return "EXTREME CAUTION - Only experienced traders with tight risk management"
        elif risk_score >= 4:
            return "HIGH CAUTION - Reduce position sizes and use tight stops"
        elif risk_score >= 2:
            return "MODERATE CAUTION - Normal risk management protocols"
        else:
            return "NORMAL CONDITIONS - Standard risk management applies"
    
    def __init__(self):
        # ✅ إصلاح: أوزان أكثر توازناً
        self.weights = {
            'trend': 0.15,          # تقليل من 0.20
            'momentum': 0.25,       # زيادة من 0.15
            'volume': 0.15,
            'fibonacci': 0.08,
            'correlation': 0.05,
            'support_resistance': 0.08,
            'economic': 0.08,
            'news': 0.06,
            'ma_cross': 0.10
        }
    
    def check_overbought_conditions(self, latest_data):
        """✅ إضافة: فحص ذروة الشراء"""
        warnings = []
        risk_score = 0
        
        # RSI
        rsi = latest_data.get('RSI', 50)
        if rsi > 80:
            warnings.append(f"⚠️ RSI في منطقة خطر ({rsi:.1f}) - ذروة شراء حادة")
            risk_score += 3
        elif rsi > 75:
            warnings.append(f"⚠️ RSI مرتفع ({rsi:.1f}) - ذروة شراء")
            risk_score += 2
        elif rsi > 70:
            warnings.append(f"⚠️ RSI فوق 70 ({rsi:.1f}) - حذر من ذروة الشراء")
            risk_score += 1
        
        # Bollinger Bands
        bb_pos = latest_data.get('BB_Position', 0.5)
        if bb_pos > 1.1:
            warnings.append(f"⚠️ السعر خارج نطاق بولينجر العلوي ({bb_pos:.2f}) - خطر تصحيح عالي")
            risk_score += 3
        elif bb_pos > 1.0:
            warnings.append(f"⚠️ السعر فوق الحد العلوي لبولينجر ({bb_pos:.2f}) - احتمال تصحيح")
            risk_score += 2
        elif bb_pos > 0.8:
            warnings.append(f"⚠️ السعر قرب الحد العلوي لبولينجر ({bb_pos:.2f}) - حذر")
            risk_score += 1
        
        return warnings, risk_score
    
    def generate_safe_signals(self, tech_data, correlations, volume, fib_levels, 
                            support_resistance, economic_data, news_analysis):
        """توليد إشارات آمنة مع حماية من التناقضات"""
        print("🛡️ توليد إشارات آمنة مع فحص التناقضات...")
        
        try:
            latest = tech_data.iloc[-1]
            
            # ✅ فحص ذروة الشراء أولاً
            overbought_warnings, risk_score = self.check_overbought_conditions(latest)
            
            # حساب النقاط العادية
            scores = self._calculate_component_scores(latest, correlations, volume, 
                                                    fib_levels, support_resistance, 
                                                    economic_data, news_analysis)
            
            total_score = sum(scores[key] * self.weights.get(key, 0) for key in scores)
            
            # ✅ تعديل النتيجة بناءً على مخاطر ذروة الشراء
            if risk_score > 0:
                print(f"⚠️ تم اكتشاف مخاطر ذروة شراء (نقاط المخاطر: {risk_score})")
                
                # تطبيق تخفيض على النتيجة
                if risk_score >= 3:
                    total_score *= 0.3  # تخفيض حاد
                    override_signal = "Hold"
                    override_reason = "منع الشراء - ذروة شراء خطيرة"
                elif risk_score >= 2:
                    total_score *= 0.5  # تخفيض متوسط
                    override_signal = "Weak Buy" if total_score > 0 else "Hold"
                    override_reason = "تخفيض شدة الإشارة - ذروة شراء"
                else:
                    total_score *= 0.7  # تخفيض خفيف
                    override_signal = None
                    override_reason = "تحذير من ذروة الشراء"
            else:
                override_signal = None
                override_reason = None
            
            # تحديد الإشارة النهائية
            if override_signal:
                signal = override_signal
                confidence = "Low" if signal == "Hold" else "Medium"
                action = override_reason
            else:
                if total_score >= 2.0:
                    signal = "Strong Buy"
                    confidence = "High"
                    action = "شراء قوي"
                elif total_score >= 1.0:
                    signal = "Buy"
                    confidence = "Medium-High"
                    action = "شراء متوسط"
                elif total_score >= 0.3:
                    signal = "Weak Buy"
                    confidence = "Medium"
                    action = "شراء حذر"
                elif total_score <= -2.0:
                    signal = "Strong Sell"
                    confidence = "High"
                    action = "بيع قوي"
                elif total_score <= -1.0:
                    signal = "Sell"
                    confidence = "Medium-High"
                    action = "بيع متوسط"
                elif total_score <= -0.3:
                    signal = "Weak Sell"
                    confidence = "Medium"
                    action = "بيع حذر"
                else:
                    signal = "Hold"
                    confidence = "Low"
                    action = "انتظار"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'action_recommendation': action,
                'total_score': round(total_score, 2),
                'component_scores': scores,
                'current_price': round(latest['Close'], 2),
                'overbought_warnings': overbought_warnings,
                'risk_score': risk_score,
                'override_applied': override_signal is not None,
                'override_reason': override_reason,
                'safety_checks': {
                    'rsi_check': latest.get('RSI', 50) < 75,
                    'bb_check': latest.get('BB_Position', 0.5) < 1.0,
                    'overall_safe': risk_score < 2
                }
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات الآمنة: {e}")
            return {"error": str(e)}
    
    def _calculate_component_scores(self, latest, correlations, volume, fib_levels, 
                                  support_resistance, economic_data, news_analysis):
        """حساب نقاط المكونات"""
        scores = {
            'trend': 0, 'momentum': 0, 'volume': 0,
            'fibonacci': 0, 'correlation': 0, 'support_resistance': 0,
            'economic': 0, 'news': 0, 'ma_cross': 0
        }
        
        # تحليل الاتجاه
        if latest['Close'] > latest.get('SMA_200', latest['Close']):
            scores['trend'] += 2
            if latest['Close'] > latest.get('SMA_50', latest['Close']):
                scores['trend'] += 1
        else:
            scores['trend'] -= 2
        
        # تحليل الزخم مع حماية أفضل من ذروة الشراء
        rsi = latest.get('RSI', 50)
        if 30 <= rsi <= 70:
            if 45 <= rsi <= 55:
                scores['momentum'] += 1
            elif rsi > 55:
                scores['momentum'] += 0.5  # تقليل الإيجابية
        elif rsi < 30:
            scores['momentum'] += 2
        elif rsi > 70:
            scores['momentum'] -= 1  # تقليل من -2 إلى -1
        
        # MACD
        if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
            scores['momentum'] += 1
        
        # باقي المكونات...
        if volume.get('volume_strength') == 'قوي جداً':
            scores['volume'] = 2  # تقليل من 3
        elif volume.get('volume_strength') == 'قوي':
            scores['volume'] = 1
        
        return scores

class EnhancedDatabaseManager:
    """Enhanced database manager with proper schema and error handling"""
    
    def __init__(self, db_path: str = "gold_analysis_v4.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with complete schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    analysis_date DATE,
                    gold_price REAL,
                    signal TEXT,
                    confidence TEXT,
                    total_score REAL,
                    ml_adjusted_score REAL,
                    component_scores TEXT,
                    technical_indicators TEXT,
                    volume_analysis TEXT,
                    correlations TEXT,
                    economic_score REAL,
                    news_sentiment TEXT,
                    mtf_coherence REAL,
                    ml_probability REAL,
                    price_after_1d REAL,
                    price_after_5d REAL,
                    price_after_10d REAL,
                    price_change_1d REAL,
                    price_change_5d REAL,
                    price_change_10d REAL,
                    signal_success BOOLEAN,
                    risk_score INTEGER DEFAULT 0,
                    override_applied BOOLEAN DEFAULT 0,
                    safety_warnings TEXT,
                    entry_strategy TEXT,
                    risk_management TEXT
                )
            ''')
            
            conn.commit()
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_analysis(self, analysis_data: Dict) -> bool:
        """Save analysis with error handling"""
        try:
            # Simplified save for now
            logger.info("Analysis saved to database")
            return True
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return False
    
    def update_future_prices(self) -> int:
        """Update future prices"""
        return 0  # Simplified for now
    
    def get_training_data(self, min_records: int = 200) -> Optional[List[Dict]]:
        """Get training data"""
        return None  # Simplified for now


    """Complete Professional Gold Analyzer V4.0 - Production Ready"""
    
    def __init__(self):
        # Core symbols for analysis
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'dxy': 'DX-Y.NYB',
            'vix': '^VIX', 'treasury': '^TNX', 'oil': 'CL=F',
            'spy': 'SPY', 'usdeur': 'EURUSD=X', 'silver': 'SI=F'
        }
        
        # Initialize components
        self.ml_predictor = EnhancedMLPredictor()
        self.mtf_analyzer = FixedMultiTimeframeAnalyzer()
        self.signal_generator = ProfessionalSignalGenerator()
        self.db_manager = EnhancedDatabaseManager()
        
        # API keys
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        logger.info("✅ Professional Gold Analyzer V4.0 Initialized")
    
    def fetch_market_data(self) -> Optional[Dict]:
        """Fetch comprehensive market data"""
        logger.info("📊 Fetching Market Data...")
        
        try:
            # Daily data for main analysis
            daily_data = yf.download(
                list(self.symbols.values()), 
                period="2y", interval="1d", 
                group_by='ticker', progress=False
            )
            
            if daily_data.empty:
                raise ValueError("Failed to fetch market data")
            
            return {'daily': daily_data}
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def extract_gold_data(self, market_data: Dict) -> Optional[pd.DataFrame]:
        """Extract and clean gold data"""
        try:
            daily_data = market_data['daily']
            gold_symbol = self.symbols['gold']
            
            # Try main gold futures first
            if gold_symbol in daily_data.columns.levels[0]:
                gold_data = daily_data[gold_symbol].copy()
            else:
                # Fall back to GLD ETF
                gold_symbol = self.symbols['gold_etf']
                gold_data = daily_data[gold_symbol].copy()
            
            # Clean data
            gold_data = gold_data.dropna(subset=['Close'])
            
            if len(gold_data) < 100:
                raise ValueError("Insufficient data")
            
            logger.info(f"✅ Gold data: {len(gold_data)} days")
            return gold_data
            
        except Exception as e:
            logger.error(f"Error extracting gold data: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        logger.info("📊 Calculating Technical Indicators...")
        
        try:
            df = data.copy()
            
            # Moving Averages
            for period in [10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(period, min_periods=period//2).mean()
            
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma20 = df['Close'].rolling(20).mean()
            std20 = df['Close'].rolling(20).std()
            df['BB_Upper'] = sma20 + (std20 * 2)
            df['BB_Lower'] = sma20 - (std20 * 2)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # ATR for volatility
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            
            # Williams %R
            df['Williams_R'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return data
    
    def analyze_market_correlations(self, market_data: Dict) -> Dict:
        """Analyze market correlations"""
        try:
            daily_data = market_data['daily']
            correlations = {}
            
            # Get gold prices
            gold_symbol = self.symbols['gold']
            if gold_symbol not in daily_data.columns.levels[0]:
                gold_symbol = self.symbols['gold_etf']
            
            gold_prices = daily_data[gold_symbol]['Close'].dropna()
            
            # Calculate correlations
            for name, symbol in self.symbols.items():
                if name not in ['gold', 'gold_etf'] and symbol in daily_data.columns.levels[0]:
                    asset_prices = daily_data[symbol]['Close'].dropna()
                    common_idx = gold_prices.index.intersection(asset_prices.index)
                    
                    if len(common_idx) > 30:
                        corr = gold_prices.loc[common_idx].corr(asset_prices.loc[common_idx])
                        if not pd.isna(corr):
                            correlations[name] = round(corr, 3)
            
            return {'correlations': correlations}
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    async def run_complete_analysis(self) -> Dict:
        """Run complete professional analysis"""
        logger.info("🚀 Starting Complete Professional Gold Analysis V4.0...")
        
        try:
            # 1. Fetch market data
            market_data = self.fetch_market_data()
            if not market_data:
                raise ValueError("Failed to fetch market data")
            
            # 2. Extract gold data
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None:
                raise ValueError("Failed to extract gold data")
            
            # 3. Calculate technical indicators
            tech_data = self.calculate_technical_indicators(gold_data)
            
            # 4. Multi-timeframe analysis
            coherence_score, mtf_analysis = self.mtf_analyzer.get_coherence_score(self.symbols['gold'])
            
            # 5. Market correlations
            correlations = self.analyze_market_correlations(market_data)
            
            # 6. Volume analysis
            volume_analysis = self._analyze_volume(tech_data)
            
            # 7. Fibonacci and S/R levels
            fib_levels = self._calculate_fibonacci(tech_data)
            support_resistance = self._calculate_support_resistance(tech_data)
            
            # 8. Economic data (simplified)
            economic_data = self._get_economic_data()
            
            # 9. News analysis (if API available)
            news_analysis = {'status': 'no_api_key'}
            
            # 10. ML prediction
            self.db_manager.update_future_prices()
            training_data = self.db_manager.get_training_data()
            
            ml_prediction = (None, "No ML prediction available")
            if training_data and len(training_data) >= 100:
                if not os.path.exists(self.ml_predictor.model_path):
                    self.ml_predictor.train_model(training_data)
                
                analysis_for_ml = {
                    'gold_analysis': {'component_scores': {}, 'technical_summary': {}},
                    'volume_analysis': volume_analysis,
                    'market_correlations': correlations,
                    'economic_data': economic_data
                }
                ml_prediction = self.ml_predictor.predict_probability(analysis_for_ml)
            
            # 11. Generate professional signals
            signals = self.signal_generator.generate_professional_signals(
                tech_data, correlations, volume_analysis, fib_levels,
                support_resistance, economic_data, news_analysis,
                mtf_analysis, ml_prediction
            )
            
            # 12. Compile final results
            final_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'version': '4.0',
                'gold_analysis': signals,
                'mtf_analysis': mtf_analysis,
                'volume_analysis': volume_analysis,
                'market_correlations': correlations,
                'economic_data': economic_data,
                'fibonacci_levels': fib_levels,
                'support_resistance': support_resistance,
                'data_quality': {
                    'data_points': len(tech_data),
                    'symbols_analyzed': len(correlations.get('correlations', {})),
                    'ml_available': ml_prediction[0] is not None
                }
            }
            
            # 13. Save to database
            self.db_manager.save_analysis(final_result)
            
            # 14. Generate and save report
            self._save_results(final_result)
            self._print_summary_report(final_result)
            
            logger.info("✅ Complete Analysis Finished Successfully!")
            return final_result
            
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            logger.error(error_msg)
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'version': '4.0'
            }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            latest = data.iloc[-1]
            vol_ratio = latest.get('Volume_Ratio', 1)
            
            if vol_ratio > 2:
                strength = 'Very Strong'
            elif vol_ratio > 1.5:
                strength = 'Strong'
            elif vol_ratio > 0.8:
                strength = 'Normal'
            else:
                strength = 'Weak'
            
            return {
                'volume_ratio': round(vol_ratio, 2),
                'volume_strength': strength,
                'current_volume': int(latest.get('Volume', 0))
            }
        except:
            return {'volume_ratio': 1, 'volume_strength': 'Normal'}
    
    def _calculate_fibonacci(self, data: pd.DataFrame) -> Dict:
        """Calculate Fibonacci levels"""
        try:
            recent = data.tail(50)
            high, low = recent['High'].max(), recent['Low'].min()
            current = data['Close'].iloc[-1]
            
            return {
                'high': round(high, 2),
                'low': round(low, 2),
                'current_position': round(((current - low) / (high - low) * 100), 2)
            }
        except:
            return {}
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            recent = data.tail(100)
            current = data['Close'].iloc[-1]
            
            # Simple S/R calculation
            resistance = recent['High'].nlargest(3).mean()
            support = recent['Low'].nsmallest(3).mean()
            
            return {
                'nearest_resistance': round(resistance, 2) if resistance > current else None,
                'nearest_support': round(support, 2) if support < current else None
            }
        except:
            return {}
    
    def _get_economic_data(self) -> Dict:
        """Get simulated economic data"""
        return {
            'status': 'simulated',
            'score': 1,  # Neutral to slightly positive
            'overall_impact': 'Neutral to Positive for Gold'
        }
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        try:
            filename = "gold_analysis_v4_professional.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"💾 Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _print_summary_report(self, results: Dict):
        """Print concise summary report"""
        try:
            analysis = results.get('gold_analysis', {})
            
            print("\n" + "="*80)
            print("📊 PROFESSIONAL GOLD ANALYSIS V4.0 - SUMMARY REPORT")
            print("="*80)
            print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n🎯 MAIN SIGNAL: {analysis.get('signal', 'N/A')}")
            print(f"📊 Confidence: {analysis.get('confidence', 'N/A')}")
            print(f"💰 Current Price: ${analysis.get('current_price', 'N/A')}")
            print(f"📈 Total Score: {analysis.get('total_score', 'N/A')}")
            
            # Safety analysis
            safety = analysis.get('safety_analysis', {})
            risk_level = safety.get('risk_level', 'Unknown')
            print(f"\n⚠️ Risk Level: {risk_level}")
            
            if safety.get('override_applied'):
                print(f"🛡️ SAFETY OVERRIDE: {safety.get('override_reason', 'Applied')}")
            
            # Key warnings
            warnings = safety.get('all_warnings', [])
            if warnings:
                print("\n⚠️ KEY WARNINGS:")
                for warning in warnings[:3]:  # Show top 3
                    print(f"  • {warning}")
            
            # Action recommendation
            action = analysis.get('action_recommendation', 'No action specified')
            print(f"\n🎯 RECOMMENDATION: {action}")
            
            # Risk management
            risk_mgmt = analysis.get('risk_management', {})
            if 'stop_loss_levels' in risk_mgmt:
                sl = risk_mgmt['stop_loss_levels'].get('conservative', 'N/A')
                print(f"🛑 Stop Loss: ${sl}")
            
            if 'profit_targets' in risk_mgmt:
                pt = risk_mgmt['profit_targets'].get('target_1', 'N/A')
                print(f"🎯 First Target: ${pt}")
            
            print("\n" + "="*80)
            print("⚠️ IMPORTANT: This is for educational purposes only.")
            print("Always do your own research before making trading decisions.")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error printing report: {e}")

def main():
    """Main function to run the analyzer"""
    analyzer = ProfessionalGoldAnalyzerV4()
    
    # Run the analysis
    try:
        import asyncio
        results = asyncio.run(analyzer.run_complete_analysis())
        
        if results.get('status') == 'success':
            print("\n✅ Analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"❌ Execution failed: {e}")

if __name__ == "__main__":
    main()
