#!/usr/bin/env python3
"""
Simple Python syntax checker
"""

import ast
import sys

def check_syntax(filename):
    """Check Python syntax"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the AST
        ast.parse(content, filename=filename)
        print(f"✅ Syntax is valid for {filename}")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax Error in {filename}:")
        print(f"  Line {e.lineno}: {e.text.strip() if e.text else 'N/A'}")
        print(f"  Error: {e.msg}")
        print(f"  Position: {' ' * (e.offset - 1 if e.offset else 0)}^")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "enhanced_gold_analyzer.py"
    
    check_syntax(filename)
