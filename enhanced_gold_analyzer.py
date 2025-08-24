                    report.append(f"  • التأثير الإجمالي: {ea.get('total_impact', 0):.1f}")
                    report.append(f"  • التوصية: {ea.get('recommendation', 'N/A')}")
                    report.append("")
                elif na.get('status') == 'simulated':
                    report.append("📰 تحليل الأخبار (محاكاة):")
                    ea = na.get('events_analysis', {})
                    report.append(f"  • التأثير: {ea.get('total_impact', 0):.1f}")
                    report.append(f"  • التوصية: {ea.get('recommendation', 'N/A')}")
                    report.append("")
            
            # البيانات الاقتصادية
            if 'economic_data' in analysis_result:
                ed = analysis_result['economic_data']
                report.append("💰 البيانات الاقتصادية (محاكاة):")
                report.append(f"  • التأثير الإجمالي: {ed.get('overall_impact', 'N/A')}")
                report.append(f"  • النقاط: {ed.get('score', 0):.1f}")
                report.append("")
            
            # ملخص السوق
            if 'market_summary' in analysis_result:
                ms = analysis_result['market_summary']
                report.append("📊 ملخص حالة السوق:")
                report.append(f"  • الحالة العامة: {ms.get('market_condition', 'N/A')}")
                report.append(f"  • آخر تحديث: {ms.get('last_update', 'N/A')}")
                report.append("")
            
            report.append("=" * 80)
            report.append("انتهى التقرير - الإصدار 3.0 GitHub Enhanced")
            report.append("✅ متوافق مع GitHub Actions | 🤖 تعلم آلي | 📊 تحليل متقدم")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"
    
    def _save_results_v3(self, results):
        """حفظ النتائج في ملفات متعددة V3"""
        try:
            # حفظ JSON الرئيسي
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"gold_analysis_v3_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل في: {filename}")
            
            # حفظ ملف مختصر للعرض السريع
            summary_filename = f"gold_summary_{timestamp}.json"
            summary = {
                'timestamp': results.get('timestamp'),
                'status': results.get('status'),
                'signal': results.get('gold_analysis', {}).get('signal'),
                'confidence': results.get('gold_analysis', {}).get('confidence'),
                'price': results.get('gold_analysis', {}).get('current_price'),
                'recommendation': results.get('gold_analysis', {}).get('action_recommendation')
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"📋 تم حفظ الملخص في: {summary_filename}")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    """الدالة الرئيسية لتشغيل المحلل المحسن"""
    print("🚀 تشغيل محلل الذهب الاحترافي V3.0 - GitHub Enhanced")
    print("=" * 60)
    
    try:
        analyzer = ProfessionalGoldAnalyzerV3()
        
        # تشغيل التحليل بشكل غير متزامن
        result = asyncio.run(analyzer.run_analysis_v3())
        
        if result.get('status') == 'success':
            print("\n🎉 تم التحليل بنجاح!")
            
            # طباعة ملخص سريع
            ga = result.get('gold_analysis', {})
            if ga and 'error' not in ga:
                print(f"📊 الإشارة: {ga.get('signal')} | الثقة: {ga.get('confidence')}")
                print(f"💰 السعر: ${ga.get('current_price')} | النقاط: {ga.get('total_score')}")
                
                if ga.get('ml_prediction', {}).get('probability'):
                    ml_prob = ga['ml_prediction']['probability']
                    print(f"🤖 احتمالية ML: {ml_prob:.1%}")
        else:
            print(f"\n❌ فشل التحليل: {result.get('error', 'خطأ غير معروف')}")
            
    except Exception as e:
        print(f"\n💥 خطأ في التشغيل الرئيسي: {e}")

if __name__ == "__main__":
    main()
