
import pandas as pd
import itertools
import time
from typing import Dict, Any, List

# نستورد الدوال الأساسية من السكربت الرئيسي
# هذا يوضح قوة تقسيم الكود إلى وحدات قابلة لإعادة الاستخدام
from gold_analyzer import (
    fetch_market_data,
    calc_indicators,
    compute_regime,
    backtest_engine,
    log # To use the same logger
)

# --- إعدادات التحسين ---
# هنا نحدد نطاق القيم التي نريد تجربتها لكل معلمة
# تحذير: كلما زادت القيم، زاد وقت التشغيل بشكل كبير
OPTIMIZATION_PARAMS = {
    "period": "5y", # الفترة التاريخية التي سيتم الاختبار عليها
    "search_space": {
        "adx_min": [20, 25],
        "rsi_buy": [25, 30, 35],
        "rsi_sell": [65, 70, 75],
        "atr_mult_sl": [1.5, 2.0, 2.5] # مضاعف وقف الخسارة
        # يمكنك إضافة معلمات أخرى هنا مثل "atr_mult_tp"
    }
}

def run_optimizer():
    """ 
    Runs the grid search optimization for the trading strategy.
    """
    log.info("--- Starting Strategy Optimization ---")
    start_time = time.time()

    # 1. جلب البيانات مرة واحدة فقط لتوفير الوقت
    log.info(f"Fetching market data for period: {OPTIMIZATION_PARAMS['period']}")
    market_data = fetch_market_data(OPTIMIZATION_PARAMS['period'])
    gold_df = market_data.get("GC=F")
    if gold_df.empty:
        log.critical("Cannot run optimizer without gold data.")
        return

    gold_df = gold_df.copy()
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)
    
    # 2. حساب المؤشرات مرة واحدة فقط
    indicators = calc_indicators(gold_df)
    regime = compute_regime(indicators)

    # 3. إنشاء كل التوليفات الممكنة من المعلمات
    param_space = OPTIMIZATION_PARAMS["search_space"]
    param_keys = param_space.keys()
    param_values = param_space.values()
    param_combinations = list(itertools.product(*param_values))
    total_runs = len(param_combinations)
    log.info(f"Generated {total_runs} unique parameter combinations to test.")

    results = []
    # 4. حلقة الاختبار الخلفي لكل توليفة
    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(param_keys, combo))
        
        # نضيف المعلمات التي لا نريد تحسينها من الإعدادات الافتراضية
        from gold_analyzer import CONFIG
        full_params = CONFIG["backtest_params"].copy()
        full_params.update(current_params)

        log.info(f"Running backtest {i+1}/{total_runs} with params: {current_params}")
        
        # تشغيل محرك الاختبار الخلفي
        backtest_result = backtest_engine(gold_df, indicators, regime, full_params)
        performance = backtest_result.get("performance", {})
        
        # حفظ النتائج
        result_row = {
            **current_params,
            "final_equity": performance.get("final_equity"),
            "profit_factor": performance.get("profit_factor"),
            "win_rate": performance.get("win_rate"),
            "trades": performance.get("trades"),
            "max_drawdown": performance.get("max_drawdown")
        }
        results.append(result_row)

    # 5. تحليل النتائج النهائية
    log.info("--- Optimization Complete ---")
    end_time = time.time()
    log.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    if not results:
        log.warning("No results were generated.")
        return

    # تحويل النتائج إلى DataFrame لسهولة التحليل
    results_df = pd.DataFrame(results)
    
    # فرز النتائج حسب أعلى ربح نهائي
    results_df.sort_values(by="final_equity", ascending=False, inplace=True)
    
    # حفظ كل النتائج في ملف CSV
    output_file = "optimization_results.csv"
    results_df.to_csv(output_file, index=False)
    log.info(f"Full optimization results saved to {output_file}")

    # طباعة أفضل 10 نتائج على الشاشة
    log.info("--- Top 10 Performing Parameter Sets ---")
    print(results_df.head(10).to_string())

    # --- حفظ أفضل المعلمات تلقائيًا ---
    best_params = results_df.iloc[0].to_dict()
    # نقوم بإزالة حقول النتائج ونحتفظ فقط بالمعلمات
    params_to_save = {k: v for k, v in best_params.items() if k in OPTIMIZATION_PARAMS["search_space"]}
    
    best_params_file = "best_params.json"
    with open(best_params_file, "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2)
    log.info(f"*** Best parameters automatically saved to {best_params_file} for future use. ***")


if __name__ == "__main__":
    run_optimizer()
