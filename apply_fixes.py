#!/usr/bin/env python3
"""
سكربت لتطبيق جميع الإصلاحات على main_analyzer_v3.py
"""

import re

def apply_all_fixes():
    # قراءة الملف الأصلي
    with open('main_analyzer_v3.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. إضافة auto_adjust=True لجميع yf.download
    content = re.sub(
        r'yf\.downloadKATEX_INLINE_OPEN(.*?)KATEX_INLINE_CLOSE',
        lambda m: f"yf.download({m.group(1)}, auto_adjust=True)" if 'auto_adjust' not in m.group(0) else m.group(0),
        content
    )
    
    # 2. إضافة import pandas as pd إذا لم يكن موجود
    if 'pd.notna' not in content and 'import pandas as pd' in content:
        content = content.replace(
            'import pandas as pd',
            'import pandas as pd\nimport pandas as pd  # للتأكد من توفر pd.notna'
        )
    
    # 3. حفظ النسخة المصلحة
    with open('main_analyzer_v3_fixed.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ تم إنشاء النسخة المصلحة: main_analyzer_v3_fixed.py")
    print("📌 الآن شغل: python main_analyzer_v3_fixed.py")

if __name__ == "__main__":
    apply_all_fixes()
