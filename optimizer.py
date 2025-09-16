# -*- coding: utf-8 -*-
import pandas as pd
import itertools
import time
import json
import os
import random
from typing import Dict, Any, List

# --- استيراد الوحدات الأساسية ---
# هذا يوضح قوة تقسيم الكود إلى وحدات قابلة لإعادة الاستخدام
from gold_analyzer import (
    fetch_market_data,
    calc_indicators,
    compute_regime,
    backtest_engine,
    log # To use the same logger
)

# --- الإعدادات الرئيسية للمُحسِّن المتكيف ---

# اسم الملف الذي سيحفظ "ذاكرة" المُحسِّن بين كل تشغيل
OPTIMIZER_STATE_FILE = "optimizer_state.json"

# الفترة التاريخية التي سيتم الاختبار عليها
HISTORICAL_PERIOD = "5y"

# نطاق البحث الافتراضي (يُستخدم فقط في أول تشغيل على الإطلاق)
DEFAULT_SEARCH_SPACE = {
    "adx_min": [20, 25, 30],
    "rsi_buy": [25, 30, 35],
    "rsi_sell": [65, 70, 75],
    "atr_mult_sl": [1.5, 2.0, 2.5, 3.0]
}

# --- دوال إدارة الحالة والبحث الذكي ---

def load_optimizer_state() -> Dict[str, Any]:
    """تحميل آخر حالة وصل إليها المُحسِّن من ملف الذاكرة."""
    if os.path.exists(OPTIMIZER_STATE_FILE):
        try:
            with open(OPTIMIZER_STATE_FILE, 'r') as f:
                log.info(f"Loading state from {OPTIMIZER_STATE_FILE}")
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load state file, starting fresh. Error: {e}")
    return {}

def save_optimizer_state(state: Dict[str, Any]):
    """حفظ الحالة الجديدة للمُحسِّن في ملف الذاكرة للتشغيل القادم."""
    try:
        with open(OPTIMIZER_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
            log.info(f"Successfully saved new state to {OPTIMIZER_STATE_FILE}")
    except Exception as e:
        log.error(f"Could not save state file: {e}")

def generate_new_search_space(last_best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """الجزء الذكي: يولد نطاق بحث جديد بناءً على أفضل النتائج السابقة."""
    if not last_best_params:
        log.info("No previous state found. Using default search space.")
        return DEFAULT_SEARCH_SPACE

    log.info(f"Generating new adaptive search space around previous best: {last_best_params}")
    new_space = {}
    
    for param, best_value in last_best_params.items():
        if param not in DEFAULT_SEARCH_SPACE: continue

        # 1. التركيز: إنشاء "مجاورة" حول أفضل قيمة سابقة
        if isinstance(best_value, float):
            step = best_value * 0.10 # خطوة بنسبة 10%
            neighborhood = [round(best_value - step, 2), round(best_value, 2), round(best_value + step, 2)]
        else: # for integers
            step = max(1, int(best_value * 0.10))
            neighborhood = [best_value - step, best_value, best_value + step]

        # 2. الطفرة: إضافة قيمة عشوائية من النطاق الأصلي لاستكشاف مناطق جديدة
        mutation = [random.choice(DEFAULT_SEARCH_SPACE[param])]

        # 3. التجميع: دمج كل القيم مع التأكد من عدم التكرار
        combined_values = sorted(list(set(neighborhood + mutation)))
        new_space[param] = combined_values

    log.info(f"Generated new search space: {new_space}")
    return new_space


def run_optimizer():
    """Runs the adaptive grid search optimization for the trading strategy."""
    log.info("--- Starting Adaptive Strategy Optimization ---")
    start_time = time.time()

    # 1. تحميل الحالة السابقة وتوليد نطاق بحث جديد وذكي
    optimizer_state = load_optimizer_state()
    last_best_params = optimizer_state.get("last_best_params")
    search_space = generate_new_search_space(last_best_params)

    # 2. جلب البيانات مرة واحدة فقط
    log.info(f"Fetching market data for period: {HISTORICAL_PERIOD}")
    market_data = fetch_market_data(HISTORICAL_PERIOD)
    gold_df = market_data.get("GC=F")
    if gold_df.empty:
        log.critical("Cannot run optimizer without gold data.")
        return

    gold_df = gold_df.copy()
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)

    # 3. حساب المؤشرات مرة واحدة فقط
    indicators = calc_indicators(gold_df)
    regime = compute_regime(indicators)

    # 4. إنشاء كل التوليفات الممكنة من المعلمات
    param_keys = search_space.keys()
    param_values = search_space.values()
    param_combinations = list(itertools.product(*param_values))
    total_runs = len(param_combinations)
    log.info(f"Generated {total_runs} unique parameter combinations to test.")

    results = []
    # 5. حلقة الاختبار الخلفي لكل توليفة
    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(param_keys, combo))
        
        from gold_analyzer import CONFIG
        full_params = CONFIG["backtest_params"].copy()
        full_params.update(current_params)

        log.info(f"Running backtest {i+1}/{total_runs} with params: {current_params}")
        backtest_result = backtest_engine(gold_df, indicators, regime, full_params)
        performance = backtest_result.get("performance", {})

        result_row = {**current_params, **performance}
        results.append(result_row)

    # 6. تحليل النتائج النهائية
    log.info("--- Optimization Complete ---")
    end_time = time.time()
    log.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    if not results:
        log.warning("No results were generated.")
        return

    results_df = pd.DataFrame(results)
    results_df.sort_values(by="final_equity", ascending=False, inplace=True)

    output_file = "optimization_results.csv"
    results_df.to_csv(output_file, index=False)
    log.info(f"Full optimization results saved to {output_file}")

    log.info("--- Top 10 Performing Parameter Sets ---")
    print(results_df.head(10).to_string())

    # 7. حفظ أفضل المعلمات والذاكرة للمستقبل
    best_run_results = results_df.iloc[0].to_dict()
    params_to_save_for_prod = {k: v for k, v in best_run_results.items() if k in search_space}

    # حفظ أفضل المعلمات للاستخدام في التحليل الحي
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(params_to_save_for_prod, f, indent=2)
    log.info(f"*** Best parameters saved to best_params.json for production use. ***")

    # حفظ الحالة (أفضل معلمات هذا التشغيل) للتشغيل القادم
    new_state = {"last_best_params": params_to_save_for_prod}
    save_optimizer_state(new_state)

if __name__ == "__main__":
    run_optimizer()
