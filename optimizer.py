# -*- coding: utf-8 -*-
import pandas as pd
import itertools
import time
import json
import os
import random
import numpy as np
from typing import Dict, Any, List

from gold_analyzer import (
    fetch_market_data,
    calc_indicators,
    compute_regime,
    backtest_engine,
    CONFIG,
    log
)

OPTIMIZER_STATE_FILE = "optimizer_state.json"
HISTORICAL_PERIOD = "5y"

DEFAULT_SEARCH_SPACE = {
    "adx_min": [20, 25, 30],
    "rsi_buy": [25, 30, 35],
    "rsi_sell": [65, 70, 75],
    "atr_mult_sl": [1.5, 2.0, 2.5, 3.0]
}

def load_optimizer_state() -> Dict[str, Any]:
    if os.path.exists(OPTIMIZER_STATE_FILE):
        try:
            with open(OPTIMIZER_STATE_FILE, 'r', encoding="utf-8") as f:
                log.info(f"Loading state from {OPTIMIZER_STATE_FILE}")
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load state file, starting fresh. Error: {e}")
    return {}

def save_optimizer_state(state: Dict[str, Any]):
    try:
        with open(OPTIMIZER_STATE_FILE, 'w', encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            log.info(f"Successfully saved new state to {OPTIMIZER_STATE_FILE}")
    except Exception as e:
        log.error(f"Could not save state file: {e}")

def generate_new_search_space(last_best_params: Dict[str, Any]) -> Dict[str, List[Any]]:
    if not last_best_params:
        log.info("No previous state found. Using default search space.")
        return DEFAULT_SEARCH_SPACE
    log.info(f"Generating new adaptive search space around previous best: {last_best_params}")
    new_space = {}
    for param, best_value in last_best_params.items():
        if param not in DEFAULT_SEARCH_SPACE:
            continue
        if isinstance(best_value, float):
            step = max(0.1, round(best_value * 0.10, 2))
            neighborhood = [round(best_value - step, 2), round(best_value, 2), round(best_value + step, 2)]
        else:
            step = max(1, int(best_value * 0.10))
            neighborhood = [best_value - step, best_value, best_value + step]
        mutation = [random.choice(DEFAULT_SEARCH_SPACE[param])]
        combined_values = sorted(list(set(neighborhood + mutation)))
        new_space[param] = combined_values
    log.info(f"Generated new search space: {new_space}")
    return new_space

def walk_forward_validate(gold_df, base_params, n_splits: int = 4) -> Dict[str, Any]:
    df = gold_df.copy()
    n = len(df)
    if n < 252*2:
        log.warning("Not enough data for walk-forward; need at least ~2y.")
        return {"wf_trades": 0, "wf_final_equity": None}
    split_points = np.linspace(int(n*0.5), n-1, n_splits+1, dtype=int)
    eq = 1.0; all_trades = 0
    for i in range(len(split_points)-1):
        train_end = split_points[i]
        test_end  = split_points[i+1]
        train_df = df.iloc[:train_end].copy()
        test_df  = df.iloc[train_end:test_end].copy()
        ind_tr = calc_indicators(train_df); reg_tr = compute_regime(ind_tr)
        best_eq = -1; best_params = base_params.copy()
        for adx_try in [max(10, base_params.get("adx_min",20)-5), base_params.get("adx_min",20), base_params.get("adx_min",20)+5]:
            p = base_params.copy(); p["adx_min"] = adx_try
            rtr = backtest_engine(train_df, ind_tr, reg_tr, p)
            feq = rtr["performance"].get("final_equity", 1.0)
            if feq > best_eq: best_eq, best_params = feq, p
        ind_ts = calc_indicators(test_df); reg_ts = compute_regime(ind_ts)
        rts = backtest_engine(test_df, ind_ts, reg_ts, best_params)
        eq *= rts["performance"].get("final_equity", 1.0)
        all_trades += rts["performance"].get("trades", 0)
    return {"wf_trades": int(all_trades), "wf_final_equity": float(eq)}

def run_optimizer():
    log.info("--- Starting Adaptive Strategy Optimization ---")
    start_time = time.time()

    optimizer_state = load_optimizer_state()
    last_best_params = optimizer_state.get("last_best_params")
    search_space = generate_new_search_space(last_best_params)

    log.info(f"Fetching market data for period: {HISTORICAL_PERIOD}")
    market_data = fetch_market_data(HISTORICAL_PERIOD)
    gold_df = market_data.get("GC=F")
    if gold_df.empty:
        log.critical("Cannot run optimizer without gold data.")
        return

    gold_df = gold_df.copy()
    gold_df.index = pd.to_datetime(gold_df.index).tz_localize(None)

    indicators = calc_indicators(gold_df)
    regime = compute_regime(indicators)

    param_keys = list(search_space.keys())
    param_values = list(search_space.values())
    param_combinations = list(itertools.product(*param_values))
    total_runs = len(param_combinations)
    log.info(f"Generated {total_runs} unique parameter combinations to test.")

    results = []
    for i, combo in enumerate(param_combinations):
        current_params = dict(zip(param_keys, combo))
        full_params = CONFIG["backtest_params"].copy()
        full_params.update(current_params)
        log.info(f"Running backtest {i+1}/{total_runs} with params: {current_params}")
        backtest_result = backtest_engine(gold_df, indicators, regime, full_params)
        performance = backtest_result.get("performance", {})
        result_row = {**current_params, **performance}
        results.append(result_row)

    log.info("--- Optimization Complete ---")
    end_time = time.time()
    log.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    if not results:
        log.warning("No results were generated.")
        return

    results_df = pd.DataFrame(results).sort_values(by="final_equity", ascending=False)
    results_df.to_csv("optimization_results.csv", index=False)
    log.info(f"Full optimization results saved to optimization_results.csv")

    print(results_df.head(10).to_string())

    best_run_results = results_df.iloc[0].to_dict()
    params_to_save_for_prod = {k: v for k, v in best_run_results.items() if k in search_space}
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(params_to_save_for_prod, f, indent=2)
    log.info(f"*** Best parameters saved to best_params.json for production use. ***")

    # Walk-Forward
    wf = walk_forward_validate(gold_df, {**CONFIG["backtest_params"], **params_to_save_for_prod}, n_splits=4)
    log.info(f"Walk-Forward result: trades={wf['wf_trades']}, final_equity={wf['wf_final_equity']}")

    new_state = {"last_best_params": params_to_save_for_prod}
    save_optimizer_state(new_state)

if __name__ == "__main__":
    run_optimizer()
