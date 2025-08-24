#!/usr/bin/env python3
"""
Simple Test Gold Analyzer
A minimal script to test GitHub Actions workflow
"""

import json
import os
from datetime import datetime

def simple_test_analysis():
    """
    Simple test function that creates basic output files
    """
    print("🧪 Starting Simple Test Gold Analysis...")
    print("=" * 50)
    
    try:
        # Simulate basic analysis results
        current_price = 2045.50
        daily_change = 12.30
        daily_change_pct = 0.60
        signal = "Buy"
        confidence = "Medium"
        
        # Create main analysis result
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "simple_test",
            "current_price": current_price,
            "daily_change": daily_change,
            "daily_change_pct": daily_change_pct,
            "signal": signal,
            "signal_strength": confidence,
            "rsi": 45.2,
            "status": "success",
            "test_mode": True,
            "message": "Simple test analysis completed successfully"
        }
        
        # Save main results file
        with open("gold_analysis_v3.json", "w") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        # Create summary file
        summary = {
            "signal": signal,
            "confidence": confidence,
            "price": current_price,
            "recommendation": f"{signal} - {confidence} confidence (TEST MODE)"
        }
        
        with open("gold_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create a simple text report
        with open("test_report.txt", "w") as f:
            f.write("SIMPLE TEST GOLD ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Timestamp: {analysis_result['timestamp']}\n")
            f.write(f"Price: ${current_price}\n")
            f.write(f"Change: +{daily_change} (+{daily_change_pct}%)\n")
            f.write(f"Signal: {signal}\n")
            f.write(f"Confidence: {confidence}\n")
            f.write("Status: TEST SUCCESSFUL\n")
        
        # Print results to console
        print("📊 TEST ANALYSIS RESULTS")
        print("=" * 50)
        print(f"💰 Current Price: ${current_price}")
        print(f"📈 Daily Change: +{daily_change} (+{daily_change_pct}%)")
        print(f"🎯 Signal: {signal}")
        print(f"💪 Confidence: {confidence}")
        print(f"📊 RSI: {analysis_result['rsi']}")
        print("=" * 50)
        print("✅ Test analysis completed successfully!")
        
        # Check environment variables (for testing)
        print("\n🔧 Environment Check:")
        analysis_type = os.getenv('ANALYSIS_TYPE', 'not_set')
        debug_mode = os.getenv('DEBUG_MODE', 'not_set')
        print(f"ANALYSIS_TYPE: {analysis_type}")
        print(f"DEBUG_MODE: {debug_mode}")
        
        # Check if we have API keys (just check existence, not values)
        news_api = "✅ Set" if os.getenv('NEWS_API_KEY') else "❌ Not Set"
        fred_api = "✅ Set" if os.getenv('FRED_API_KEY') else "❌ Not Set"
        print(f"NEWS_API_KEY: {news_api}")
        print(f"FRED_API_KEY: {fred_api}")
        
        return analysis_result
        
    except Exception as e:
        error_result = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error",
            "analysis_type": "simple_test_failed"
        }
        
        with open("gold_analysis_v3.json", "w") as f:
            json.dump(error_result, f, indent=2)
        
        print(f"❌ Test Error: {e}")
        return error_result

if __name__ == "__main__":
    print("🚀 Simple Test Gold Analyzer")
    print("Purpose: Test GitHub Actions workflow")
    print("=" * 50)
    simple_test_analysis()
