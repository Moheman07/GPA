#!/usr/bin/env python3
"""إصلاح مشاكل المسافات البادئة في main_analyzer_v3.py"""

def fix_indentation():
    with open('main_analyzer_v3.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_database_manager = False
    in_professional_analyzer = False
    class_indent = ""
    
    for i, line in enumerate(lines):
        # تحديد بداية الكلاسات
        if 'class DatabaseManager' in line:
            in_database_manager = True
            class_indent = "    "
        elif 'class ProfessionalGoldAnalyzerV3' in line:
            in_professional_analyzer = True
            class_indent = "    "
        
        # إصلاح _get_future_prices
        if 'def _get_future_prices' in line and in_database_manager:
            line = "    " + line.lstrip()
        
        # إصلاح fetch_multi_timeframe_data
        if 'def fetch_multi_timeframe_data' in line and in_professional_analyzer:
            line = "    " + line.lstrip()
        
        # إصلاح generate_professional_signals_v3
        if 'def generate_professional_signals_v3' in line and in_professional_analyzer:
            line = "    " + line.lstrip()
        
        fixed_lines.append(line)
    
    # حفظ الملف المصحح
    with open('main_analyzer_v3_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ تم إصلاح المسافات البادئة وحفظ الملف: main_analyzer_v3_fixed.py")

if __name__ == "__main__":
    fix_indentation()
