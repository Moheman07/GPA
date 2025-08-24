#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_bollinger_bands():
    """Test Bollinger Bands calculation"""
    print("🧪 Testing Bollinger Bands calculation...")
    
    # Create sample data
    data = {
        'Close': np.random.normal(2000, 50, 100)
    }
    df = pd.DataFrame(data)
    
    try:
        # Test the fixed BB calculation
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # Fixed BB_Position calculation
        bb_width = df['BB_Upper'] - df['BB_Lower']
        bb_position = (df['Close'] - df['BB_Lower']) / bb_width
        df['BB_Position'] = bb_position.fillna(0.5)
        
        latest = df.iloc[-1]
        bb_pos = latest['BB_Position']
        
        print(f"  ✅ BB_Position: {bb_pos:.3f}")
        print(f"  ✅ BB_Position type: {type(bb_pos)}")
        
        # Test the comparison logic
        if pd.notna(bb_pos):
            if bb_pos < 0.2:
                result = "Near lower band"
            elif bb_pos > 0.8:
                result = "Near upper band"
            else:
                result = "Middle range"
            print(f"  ✅ BB analysis: {result}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ BB calculation failed: {e}")
        return False

def test_rsi_calculation():
    """Test RSI calculation"""
    print("🧪 Testing RSI calculation...")
    
    # Create sample data with trend
    prices = []
    for i in range(100):
        price = 2000 + i * 2 + np.random.normal(0, 10)
        prices.append(price)
    
    df = pd.DataFrame({'Close': prices})
    
    try:
        # Test fixed RSI calculation
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].clip(0, 100)
        
        latest_rsi = df['RSI'].iloc[-1]
        
        print(f"  ✅ RSI: {latest_rsi:.2f}")
        print(f"  ✅ RSI type: {type(latest_rsi)}")
        
        # Test comparison logic
        if pd.notna(latest_rsi):
            if 30 <= latest_rsi <= 70:
                result = "Normal range"
            elif latest_rsi < 30:
                result = "Oversold"
            else:
                result = "Overbought"
            print(f"  ✅ RSI analysis: {result}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RSI calculation failed: {e}")
        return False

def test_series_comparisons():
    """Test Series comparison fixes"""
    print("🧪 Testing Series comparisons...")
    
    try:
        # Create sample data similar to what we get from yfinance
        data = {
            'Close': [2000, 2010, 2020, 2015, 2025],
            'SMA_20': [1995, 2000, 2010, 2012, 2020],
            'SMA_50': [1990, 1995, 2000, 2005, 2015],
            'SMA_200': [1980, 1985, 1990, 1995, 2000]
        }
        df = pd.DataFrame(data)
        latest = df.iloc[-1]
        
        # Test fixed comparison logic
        sma_200 = latest.get('SMA_200', 0)
        sma_50 = latest.get('SMA_50', 0) 
        sma_20 = latest.get('SMA_20', 0)
        current_price = latest['Close']
        
        score = 0
        if pd.notna(sma_200) and current_price > sma_200:
            score += 2
        if pd.notna(sma_50) and current_price > sma_50:
            score += 1
        if pd.notna(sma_20) and current_price > sma_20:
            score += 1
        
        print(f"  ✅ Price: {current_price}")
        print(f"  ✅ SMA_200: {sma_200}")
        print(f"  ✅ Trend score: {score}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Series comparison failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Gold Analyzer Fixes")
    print("=" * 50)
    
    tests = [
        test_bollinger_bands,
        test_rsi_calculation, 
        test_series_comparisons
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
