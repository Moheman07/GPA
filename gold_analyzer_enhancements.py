#!/usr/bin/env python3
"""
تحسينات إضافية لمحلل الذهب المتقدم
ميزات إضافية قوية للتداول الاحترافي
"""

import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class GoldAnalyzerEnhancements:
    """تحسينات إضافية لمحلل الذهب"""
    
    def __init__(self):
        self.enhancements = {
            'fibonacci_levels': True,
            'support_resistance': True,
            'volume_profile': True,
            'market_structure': True,
            'divergence_detection': True,
            'correlation_analysis': True
        }
    
    def calculate_fibonacci_levels(self, data):
        """حساب مستويات فيبوناتشي"""
        try:
            high = data['High'].max()
            low = data['Low'].min()
            diff = high - low
            
            levels = {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.500': low + 0.500 * diff,
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high
            }
            
            return levels
        except Exception as e:
            print(f"❌ خطأ في حساب فيبوناتشي: {e}")
            return {}
    
    def detect_support_resistance(self, data, window=20):
        """كشف مستويات الدعم والمقاومة"""
        try:
            support_levels = []
            resistance_levels = []
            
            for i in range(window, len(data) - window):
                # كشف الدعم
                if data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window].min():
                    support_levels.append(data['Low'].iloc[i])
                
                # كشف المقاومة
                if data['High'].iloc[i] == data['High'].iloc[i-window:i+window].max():
                    resistance_levels.append(data['High'].iloc[i])
            
            return {
                'support': list(set(support_levels)),
                'resistance': list(set(resistance_levels))
            }
        except Exception as e:
            print(f"❌ خطأ في كشف الدعم/المقاومة: {e}")
            return {'support': [], 'resistance': []}
    
    def analyze_volume_profile(self, data):
        """تحليل ملف الحجم"""
        try:
            # تجميع الحجم حسب مستويات السعر
            price_volume = pd.DataFrame({
                'price': data['Close'],
                'volume': data['Volume']
            })
            
            # تجميع الحجم في مستويات سعرية
            price_bins = pd.cut(price_volume['price'], bins=50)
            volume_profile = price_volume.groupby(price_bins)['volume'].sum()
            
            # تحديد مناطق الحجم العالي
            high_volume_levels = volume_profile[volume_profile > volume_profile.quantile(0.8)]
            
            return {
                'volume_profile': volume_profile,
                'high_volume_levels': high_volume_levels,
                'poc': volume_profile.idxmax()  # Point of Control
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل ملف الحجم: {e}")
            return {}
    
    def analyze_market_structure(self, data):
        """تحليل هيكل السوق"""
        try:
            # تحديد القمم والقيعان
            highs = data['High'].rolling(5, center=True).max() == data['High']
            lows = data['Low'].rolling(5, center=True).min() == data['Low']
            
            # تحديد الاتجاه
            higher_highs = []
            higher_lows = []
            lower_highs = []
            lower_lows = []
            
            for i in range(10, len(data)):
                if highs.iloc[i]:
                    if data['High'].iloc[i] > data['High'].iloc[i-10:i].max():
                        higher_highs.append(i)
                    else:
                        lower_highs.append(i)
                
                if lows.iloc[i]:
                    if data['Low'].iloc[i] > data['Low'].iloc[i-10:i].max():
                        higher_lows.append(i)
                    else:
                        lower_lows.append(i)
            
            # تحديد مرحلة السوق
            if len(higher_highs) > len(lower_highs) and len(higher_lows) > len(lower_lows):
                market_phase = "اتجاه صاعد"
            elif len(lower_highs) > len(higher_highs) and len(lower_lows) > len(higher_lows):
                market_phase = "اتجاه هابط"
            else:
                market_phase = "سوق عرضي"
            
            return {
                'market_phase': market_phase,
                'higher_highs': len(higher_highs),
                'higher_lows': len(higher_lows),
                'lower_highs': len(lower_highs),
                'lower_lows': len(lower_lows)
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل هيكل السوق: {e}")
            return {}
    
    def detect_divergences(self, data):
        """كشف التناقضات (Divergences)"""
        try:
            divergences = []
            
            # RSI Divergence
            rsi = talib.RSI(data['Close'], timeperiod=14)
            
            # Bullish Divergence (السعر ينخفض، RSI يرتفع)
            for i in range(20, len(data)):
                if (data['Close'].iloc[i] < data['Close'].iloc[i-10] and 
                    rsi.iloc[i] > rsi.iloc[i-10]):
                    divergences.append({
                        'type': 'bullish',
                        'indicator': 'RSI',
                        'date': data.index[i],
                        'strength': 'medium'
                    })
            
            # Bearish Divergence (السعر يرتفع، RSI ينخفض)
            for i in range(20, len(data)):
                if (data['Close'].iloc[i] > data['Close'].iloc[i-10] and 
                    rsi.iloc[i] < rsi.iloc[i-10]):
                    divergences.append({
                        'type': 'bearish',
                        'indicator': 'RSI',
                        'date': data.index[i],
                        'strength': 'medium'
                    })
            
            return divergences
        except Exception as e:
            print(f"❌ خطأ في كشف التناقضات: {e}")
            return []
    
    def analyze_correlations(self, data, symbols=['SPY', 'DX-Y.NYB', '^VIX']):
        """تحليل الارتباطات مع الأصول الأخرى"""
        try:
            import yfinance as yf
            
            correlations = {}
            
            for symbol in symbols:
                try:
                    # جلب بيانات الأصول الأخرى
                    other_data = yf.download(symbol, period="1y", progress=False)
                    if not other_data.empty:
                        # حساب الارتباط
                        common_index = data.index.intersection(other_data.index)
                        if len(common_index) > 30:
                            correlation = data.loc[common_index, 'Close'].corr(
                                other_data.loc[common_index, 'Close']
                            )
                            correlations[symbol] = correlation
                except:
                    continue
            
            return correlations
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {}
    
    def calculate_advanced_metrics(self, data):
        """حساب مقاييس متقدمة"""
        try:
            # حساب مؤشر القوة النسبية المركب
            rsi = talib.RSI(data['Close'], timeperiod=14)
            stoch_k, stoch_d = talib.STOCH(data['High'], data['Low'], data['Close'])
            williams_r = talib.WILLR(data['High'], data['Low'], data['Close'])
            
            # مؤشر القوة المركب
            composite_strength = (rsi + stoch_k + (100 + williams_r)) / 3
            
            # مؤشر التذبذب المركب
            atr = talib.ATR(data['High'], data['Low'], data['Close'])
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'])
            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100
            
            composite_volatility = (atr / data['Close'] + bb_width / 100) / 2
            
            # مؤشر الزخم المركب
            macd, macd_signal, macd_hist = talib.MACD(data['Close'])
            roc = talib.ROC(data['Close'], timeperiod=10)
            mom = talib.MOM(data['Close'], timeperiod=10)
            
            composite_momentum = (macd_hist + roc + mom) / 3
            
            return {
                'composite_strength': composite_strength.iloc[-1],
                'composite_volatility': composite_volatility.iloc[-1],
                'composite_momentum': composite_momentum.iloc[-1],
                'trend_strength': abs(composite_momentum.iloc[-1]),
                'volatility_regime': 'high' if composite_volatility.iloc[-1] > 0.02 else 'low'
            }
        except Exception as e:
            print(f"❌ خطأ في حساب المقاييس المتقدمة: {e}")
            return {}
    
    def generate_enhanced_signals(self, data, enhancements):
        """توليد إشارات محسنة"""
        try:
            signals = []
            
            # تحليل فيبوناتشي
            if enhancements.get('fibonacci_levels'):
                fib_levels = self.calculate_fibonacci_levels(data)
                current_price = data['Close'].iloc[-1]
                
                for level, price in fib_levels.items():
                    if abs(current_price - price) / price < 0.01:  # ضمن 1%
                        if level in ['0.618', '0.786']:
                            signals.append({
                                'type': 'fibonacci_support',
                                'level': level,
                                'price': price,
                                'strength': 'strong'
                            })
                        elif level in ['0.236', '0.382']:
                            signals.append({
                                'type': 'fibonacci_resistance',
                                'level': level,
                                'price': price,
                                'strength': 'strong'
                            })
            
            # تحليل الدعم والمقاومة
            if enhancements.get('support_resistance'):
                sr_levels = self.detect_support_resistance(data)
                current_price = data['Close'].iloc[-1]
                
                for support in sr_levels['support']:
                    if abs(current_price - support) / support < 0.02:  # ضمن 2%
                        signals.append({
                            'type': 'support_level',
                            'price': support,
                            'strength': 'medium'
                        })
                
                for resistance in sr_levels['resistance']:
                    if abs(current_price - resistance) / resistance < 0.02:  # ضمن 2%
                        signals.append({
                            'type': 'resistance_level',
                            'price': resistance,
                            'strength': 'medium'
                        })
            
            # تحليل التناقضات
            if enhancements.get('divergence_detection'):
                divergences = self.detect_divergences(data)
                if divergences:
                    signals.extend(divergences)
            
            return signals
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات المحسنة: {e}")
            return []
    
    def create_enhanced_report(self, data, enhancements_results):
        """إنشاء تقرير محسن"""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 تقرير التحليل المحسن - الإصدار 6.0+")
            report.append("=" * 80)
            report.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # مستويات فيبوناتشي
            if 'fibonacci_levels' in enhancements_results:
                fib_levels = enhancements_results['fibonacci_levels']
                current_price = data['Close'].iloc[-1]
                report.append("📐 مستويات فيبوناتشي:")
                for level, price in fib_levels.items():
                    distance = abs(current_price - price) / price * 100
                    if distance < 5:  # ضمن 5%
                        report.append(f"  • مستوى {level}: ${price:.2f} (قريب: {distance:.1f}%)")
                report.append("")
            
            # الدعم والمقاومة
            if 'support_resistance' in enhancements_results:
                sr = enhancements_results['support_resistance']
                current_price = data['Close'].iloc[-1]
                report.append("🛡️ مستويات الدعم والمقاومة:")
                report.append(f"  • مستويات الدعم: {len(sr['support'])}")
                report.append(f"  • مستويات المقاومة: {len(sr['resistance'])}")
                report.append("")
            
            # هيكل السوق
            if 'market_structure' in enhancements_results:
                ms = enhancements_results['market_structure']
                report.append("🏗️ هيكل السوق:")
                report.append(f"  • مرحلة السوق: {ms['market_phase']}")
                report.append(f"  • قمم أعلى: {ms['higher_highs']}")
                report.append(f"  • قيعان أعلى: {ms['higher_lows']}")
                report.append("")
            
            # التناقضات
            if 'divergences' in enhancements_results:
                divergences = enhancements_results['divergences']
                if divergences:
                    report.append("🔄 التناقضات المكتشفة:")
                    for div in divergences:
                        report.append(f"  • {div['type']} {div['indicator']} - {div['strength']}")
                    report.append("")
            
            # المقاييس المتقدمة
            if 'advanced_metrics' in enhancements_results:
                am = enhancements_results['advanced_metrics']
                report.append("📊 المقاييس المتقدمة:")
                report.append(f"  • قوة مركبة: {am.get('composite_strength', 0):.2f}")
                report.append(f"  • تذبذب مركب: {am.get('composite_volatility', 0):.4f}")
                report.append(f"  • زخم مركب: {am.get('composite_momentum', 0):.2f}")
                report.append(f"  • قوة الاتجاه: {am.get('trend_strength', 0):.2f}")
                report.append(f"  • نظام التذبذب: {am.get('volatility_regime', 'unknown')}")
                report.append("")
            
            report.append("=" * 80)
            report.append("انتهى التقرير المحسن - الإصدار 6.0+")
            report.append("تم تطوير: فيبوناتشي | دعم/مقاومة | هيكل السوق | تناقضات | مقاييس متقدمة")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"خطأ في إنشاء التقرير المحسن: {e}"

def main():
    """الدالة الرئيسية للتحسينات"""
    print("🚀 تشغيل التحسينات الإضافية...")
    
    # إنشاء كائن التحسينات
    enhancer = GoldAnalyzerEnhancements()
    
    print("✅ تم تحميل التحسينات الإضافية بنجاح!")
    print("📋 التحسينات المتاحة:")
    for enhancement, enabled in enhancer.enhancements.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {enhancement}")
    
    print("\n🎯 هذه التحسينات تضيف:")
    print("  • مستويات فيبوناتشي")
    print("  • كشف الدعم والمقاومة")
    print("  • تحليل ملف الحجم")
    print("  • تحليل هيكل السوق")
    print("  • كشف التناقضات")
    print("  • تحليل الارتباطات")
    print("  • مقاييس متقدمة")

if __name__ == "__main__":
    main()
