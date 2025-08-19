#!/usr/bin/env python3
import yfinance as yf
import os
from datetime import datetime

def test_basic_functionality():
    print("🔍 اختبار الوظائف الأساسية...")
    print("="*50)
    
    # 1. اختبار yfinance
    print("\n1. اختبار جلب بيانات الذهب:")
    try:
        gold_data = yf.download('GC=F', period='5d', progress=False)
        if not gold_data.empty:
            print(f"✅ نجح جلب البيانات - آخر سعر: ${gold_data['Close'].iloc[-1]:.2f}")
        else:
            print("❌ فشل - البيانات فارغة")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    # 2. اختبار مفاتيح API
    print("\n2. اختبار مفاتيح API:")
    news_key = os.getenv("NEWS_API_KEY")
    if news_key:
        print(f"✅ NEWS_API_KEY موجود - الطول: {len(news_key)} حرف")
    else:
        print("❌ NEWS_API_KEY مفقود")
    
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        print(f"✅ FRED_API_KEY موجود")
    else:
        print("⚠️ FRED_API_KEY مفقود (اختياري)")
    
    # 3. اختبار الرموز البديلة
    print("\n3. اختبار الرموز البديلة:")
    symbols = ['GC=F', 'GLD', 'XAUUSD=X']
    for symbol in symbols:
        try:
            data = yf.Ticker(symbol).history(period='1d')
            if not data.empty:
                print(f"✅ {symbol} يعمل - السعر: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {symbol} لا يعمل")
        except:
            print(f"❌ {symbol} فشل")
    
    print("\n" + "="*50)
    print("انتهى الاختبار")

if __name__ == "__main__":
    test_basic_functionality()
