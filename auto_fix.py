#!/usr/bin/env python3
"""إصلاح أوتوماتيكي كامل لملف main_analyzer_v3.py"""

import re
import sys

def auto_fix_script():
    print("🔧 بدء الإصلاح الأوتوماتيكي...")
    
    try:
        # قراءة الملف
        with open('main_analyzer_v3.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # إصلاح 1: _get_future_prices يجب أن يكون داخل DatabaseManager
        content = re.sub(
            r'\ndef _get_future_pricesKATEX_INLINE_OPENself, analysis_dateKATEX_INLINE_CLOSE:',
            r'    def _get_future_prices(self, analysis_date):',
            content
        )
        
        # إصلاح 2: fetch_multi_timeframe_data يجب أن يكون داخل ProfessionalGoldAnalyzerV3
        content = re.sub(
            r'\ndef fetch_multi_timeframe_dataKATEX_INLINE_OPENselfKATEX_INLINE_CLOSE:',
            r'    def fetch_multi_timeframe_data(self):',
            content
        )
        
        # إصلاح 3: generate_professional_signals_v3 يجب أن يكون داخل ProfessionalGoldAnalyzerV3
        content = re.sub(
            r'\ndef generate_professional_signals_v3KATEX_INLINE_OPENself,',
            r'    def generate_professional_signals_v3(self,',
            content
        )
        
        # إصلاح 4: إصلاح المسافات البادئة للمحتوى داخل _get_future_prices
        def fix_get_future_prices(match):
            lines = match.group(0).split('\n')
            fixed_lines = [lines[0]]  # احتفظ بسطر التعريف
            for line in lines[1:]:
                if line.strip():  # إذا كان السطر غير فارغ
                    # أضف 8 مسافات للمحتوى داخل الدالة
                    fixed_lines.append('        ' + line.lstrip())
                else:
                    fixed_lines.append(line)
            return '\n'.join(fixed_lines)
        
        content = re.sub(
            r'    def _get_future_pricesKATEX_INLINE_OPENself, analysis_dateKATEX_INLINE_CLOSE:.*?(?=\n    def|\nclass|\Z)',
            fix_get_future_prices,
            content,
            flags=re.DOTALL
        )
        
        # إصلاح 5: إصلاح التكرار في fetch_multi_timeframe_data
        # إزالة أي تكرار
        content = re.sub(
            r'(def fetch_multi_timeframe_dataKATEX_INLINE_OPENselfKATEX_INLINE_CLOSE:.*?return None\s*\n)(?=.*def fetch_multi_timeframe_data)',
            '',
            content,
            flags=re.DOTALL
        )
        
        # إصلاح 6: التأكد من أن جميع الدوال داخل الكلاسات لها المسافات الصحيحة
        lines = content.split('\n')
        fixed_lines = []
        in_class = False
        class_name = ""
        
        for i, line in enumerate(lines):
            # تحديد بداية الكلاس
            if line.strip().startswith('class '):
                in_class = True
                class_name = line.strip().split()[1].split('(')[0].rstrip(':')
                fixed_lines.append(line)
                continue
            
            # إذا كنا داخل كلاس
            if in_class:
                # إذا كان سطر تعريف دالة
                if line.strip().startswith('def ') and not line.startswith('    '):
                    line = '    ' + line.lstrip()
                # إذا وصلنا لكلاس جديد أو نهاية الملف
                elif line.strip().startswith('class ') or (i == len(lines) - 1):
                    in_class = False
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # إصلاح 7: إصلاح أي مشاكل في المسافات البادئة للتقاطعات الذهبية
        content = re.sub(
            r'\n\s+# التقاطعات الذهبية',
            r'\n            # التقاطعات الذهبية',
            content
        )
        
        # حفظ الملف المصحح
        output_file = 'main_analyzer_v3_fixed.py'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ تم إصلاح الملف بنجاح!")
        print(f"📄 الملف المصحح: {output_file}")
        print("\n🚀 لتشغيل الملف المصحح:")
        print(f"   python {output_file}")
        
        # اختبار بسيط للتأكد من صحة الكود
        print("\n🔍 فحص الملف المصحح...")
        try:
            compile(content, output_file, 'exec')
            print("✅ الكود صحيح ويمكن تشغيله!")
        except SyntaxError as e:
            print(f"⚠️ ما زال هناك خطأ في السطر {e.lineno}: {e.msg}")
            print("   قد تحتاج لمراجعة يدوية بسيطة")
        
        return True
        
    except FileNotFoundError:
        print("❌ لم يتم العثور على ملف main_analyzer_v3.py")
        print("   تأكد من وجود الملف في نفس المجلد")
        return False
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        return False

def create_run_script():
    """إنشاء سكربت لتشغيل الملف المصحح"""
    run_script = """#!/usr/bin/env python3
import subprocess
import sys

print("🚀 تشغيل محلل الذهب المصحح...")
subprocess.run([sys.executable, "main_analyzer_v3_fixed.py"])
"""
    
    with open('run_analyzer.py', 'w', encoding='utf-8') as f:
        f.write(run_script)
    
    print("\n📝 تم إنشاء سكربت التشغيل: run_analyzer.py")

if __name__ == "__main__":
    if auto_fix_script():
        create_run_script()
        print("\n✨ تم إكمال كل شيء!")
        print("\n🎯 الخطوات التالية:")
        print("   1. شغل المحلل: python main_analyzer_v3_fixed.py")
        print("   2. أو استخدم: python run_analyzer.py")
