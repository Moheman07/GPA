#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleGoldAnalyzer:
    """محلل مبسط للذهب لتجنب الأخطاء"""
    
    def __init__(self):
        self.gold_symbol = 'GC=F'
    
    def fetch_gold_data(self):
        """جلب بيانات الذهب بشكل آمن"""
        try:
            # جرب رموز مختلفة
            symbols = ['GC=F', 'GLD']
            
            for symbol in symbols:
                print(f"محاولة جلب بيانات {symbol}...")
                data = yf.download(symbol, period='3mo', interval='1d', progress=False, auto_adjust=True)
                
                if not data.empty and len(data) > 20:
                    print(f"✅ نجح جلب البيانات من {symbol}")
                    self.gold_symbol = symbol
                    return data
            
            raise ValueError("فشل جلب بيانات الذهب")
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
    def calculate_indicators(self, data):
        """حساب المؤشرات الأساسية"""
        try:
            df = data.copy()
            
            # المتوسطات المتحركة
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            return df.dropna()
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return data
    
    def generate_signal(self, data):
        """توليد إشارة بسيطة"""
        try:
            if data.empty or len(data) < 2:
                return None
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            score = 0
            
            # تحليل السعر والمتوسطات
            if latest['Close'] > latest.get('SMA_20', latest['Close']):
                score += 1
            if latest['Close'] > latest.get('SMA_50', latest['Close']):
                score += 1
            
            # تحليل RSI
            rsi = latest.get('RSI', 50)
            if 30 <= rsi <= 70:
                if rsi > 50:
                    score += 0.5
                else:
                    score -= 0.5
            elif rsi < 30:
                score += 1.5  # ذروة بيع
            else:
                score -= 1.5  # ذروة شراء
            
            # تحليل MACD
            if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
                score += 1
            else:
                score -= 1
            
            # تحديد الإشارة
            if score >= 2:
                signal = "Strong Buy"
                confidence = "High"
            elif score >= 1:
                signal = "Buy"
                confidence = "Medium"
            elif score <= -2:
                signal = "Strong Sell"
                confidence = "High"
            elif score <= -1:
                signal = "Sell"
                confidence = "Medium"
            else:
                signal = "Hold"
                confidence = "Low"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'score': round(score, 2),
                'price': round(latest['Close'], 2),
                'rsi': round(rsi, 1),
                'trend': 'صاعد' if score > 0 else 'هابط' if score < 0 else 'عرضي'
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارة: {e}")
            return None
    
    def run_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء التحليل المبسط للذهب...")
        
        # جلب البيانات
        data = self.fetch_gold_data()
        if data is None:
            return {'status': 'error', 'message': 'فشل جلب البيانات'}
        
        # حساب المؤشرات
        data_with_indicators = self.calculate_indicators(data)
        
        # توليد الإشارة
        signal_result = self.generate_signal(data_with_indicators)
        
        if signal_result is None:
            return {'status': 'error', 'message': 'فشل توليد الإشارة'}
        
        # النتيجة النهائية
        result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'symbol': self.gold_symbol,
            'analysis': signal_result,
            'data_points': len(data_with_indicators),
            'date_range': {
                'start': str(data.index[0].date()),
                'end': str(data.index[-1].date())
            }
        }
        
        # طباعة النتائج
        print("\n" + "="*50)
        print("📊 نتائج التحليل:")
        print(f"  • الرمز: {self.gold_symbol}")
        print(f"  • السعر: ${signal_result['price']}")
        print(f"  • الإشارة: {signal_result['signal']}")
        print(f"  • الثقة: {signal_result['confidence']}")
        print(f"  • RSI: {signal_result['rsi']}")
        print(f"  • الاتجاه: {signal_result['trend']}")
        print("="*50)
        
        return result

def main():
    analyzer = SimpleGoldAnalyzer()
    result = analyzer.run_analysis()
    
    # حفظ النتائج
    import json
    
    # حفظ الملخص
    if result.get('status') == 'success':
        summary = {
            'last_update': result['timestamp'],
            'version': '3.0-simplified',
            'signal': result['analysis']['signal'],
            'confidence': result['analysis']['confidence'],
            'price': result['analysis']['price'],
            'ml_probability': None,  # لا يوجد ML في النسخة المبسطة
            'market_condition': result['analysis']['trend']
        }
    else:
        summary = {
            'last_update': datetime.now().isoformat(),
            'version': '3.0-simplified',
            'signal': None,
            'confidence': None,
            'price': None,
            'ml_probability': None,
            'market_condition': 'خطأ في التحليل'
        }
    
    with open('gold_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # حفظ التحليل الكامل
    with open('gold_analysis_v3.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n✅ تم حفظ النتائج")

if __name__ == "__main__":
    main()
