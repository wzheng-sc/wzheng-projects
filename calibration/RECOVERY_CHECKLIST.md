# Recovery checklist – analysis changes from conversation

Use this when redoing the full analysis. **helpers.py** is already updated; the notebook may need cells re-added or re-run.

---

## helpers.py (recovered)

- **calibration()**: Returns `(0, 0)` when `mean_d == 0` or not finite; returns `(calibration, 0.0)` when `len(conversion) <= 1` (avoids NaN from `np.cov`).
- **calculate_percent_revenue_in_sla()**: `calibration = np.where(conversion != 0, pred/conversion, 0)`; returns `(0, 0)` when `mean_d == 0`; returns `(perc_rev_slo, 0.0)` when `len(conversion) <= 1`.
- **winsorization_revenue_ci_summary()**: Adds `conversion_cut_out`, `conversion_cut_out_pct` (and `total_conv_full` / `conv_kept` in the loop).
- **_calibration_analysis_one()**: Adds `cali_d_ci_width`, `cali_d_ci_width_rel`, `perc_rev_sla_d_ci_width`, `perc_rev_sla_d_ci_width_rel`.
- **all_traffic_data()**: `week` = `DATE(served_date_pst)` (not WEEK(MONDAY)).

---

## Notebook (eb_heuristic.ipynb) – what to have

If you re-create or reset the notebook, ensure these exist (order is approximate):

1. **Variance (empirical vs delta)**  
   After the cell that defines `calculate_mean_and_half_ci_percentile` and uses `df_sampled_1201_agg`: one cell that computes empirical variance of `pct_rev_sla` and `calibration` across sims, and delta-method variance per sim (e.g. `groupby('sim_id')`, call `calibration` and `calculate_percent_revenue_in_sla` per group), then prints both and ratio. Optional: `MAX_SIMS_FOR_DELTA = 100` and only append when `np.isfinite(cal_var)` and `np.isfinite(perc_var)`.

2. **Calibration 0 when conversion 0**  
   Wherever you build agg calibration:  
   `calibration = np.where(conversions == 0, 0, conversions_pred / conversions)`  
   (and for population totals, guard division by zero when total conversions is 0).

3. **Winsorization revenue/CI**  
   - Use `winsorization_revenue_ci_summary`; table will include `conversion_cut_out`, `conversion_cut_out_pct`.
   - Before plotting, check `'conversion_cut_out_pct' in df_rev_ci.columns` (or raise a clear error).
   - Plots: Figure 1 = revenue_cut_out_pct + conversion_cut_out_pct (left), cali_d_ci_reduction_pct (right). Figure 2 = revenue_cut_out (left), conversion_cut_out_pct + perc_rev_sla_d_ci_reduction_pct (right, one or two axes).

4. **Scatter (impression / log revenue)**  
   Cap points per plot, e.g. `MAX_SCATTER_POINTS = 50_000` and `.sample(n=MAX_SCATTER_POINTS)` before plotting, to avoid timeouts.

5. **Scatter: log(revenue) vs calibration (and log(calibration))**  
   - Outliers defined by **raw (pre-log) calibration** only: `is_outlier_iqr(df_plot['calibration'])` for row 0; for row 1 use the same flags reindexed: `out_orig.reindex(df_log.index).fillna(False)` (and same for app).
   - Axes: **x = log(revenue)**, **y = calibration** (row 0) or **y = log(calibration)** (row 1).
   - Fix: use `&` and parentheses for element-wise conditions, not `and`.

6. **AB metrics by advertiser**  
   - Wide table from `df_ab_advertiser_agg` with index `['ad_account_id', 'adv_size_by_revenue', 'optimization_goal', 'treatment_group']`.
   - Loop over `adv_size_by_revenue`; for each, run `calibration_analysis` on the subset (with `line_item_id` = ad_account_id), filter to same goals (e.g. Overall, APP_INSTALLS), add column `adv_size_by_revenue`, then `pd.concat` and display.

7. **Time series (df_ts)**  
   - One cell that builds `df_ts_by_acct`, `df_ts_by_acct_agg`, `df_ts_overall`, `df_ts_overall_agg` (and `pct_rev_sla`), and ends with `df_ts_overall_agg.head()`. Depends on `df_ts` existing (e.g. from `all_traffic_data` or similar).

8. **pct_rev_sla by week**  
   - One chart: x = week, y = pct_rev_sla; one line from `df_ts_overall_agg`, one line per `adv_size_by_revenue` from `df_ts_by_acct_agg`. Optional: x-tick rotation 45°.

9. **Coefficient of variation (CV)**  
   - For each label (Overall + each adv_size): CV = std(pct_rev_sla) / mean(pct_rev_sla) over weeks; display a small table (label, mean, std, cv, cv_pct).

10. **pct revenue by adv_size over time**  
    - From `df_ts_by_acct_agg`: per week, pct_revenue = revenue / total revenue that week (use a copy, e.g. `df_plot_rev`); plot week vs pct_revenue, one line per `adv_size_by_revenue`.

---

## Quick verification

- Run `calibration_analysis` once and check the result has columns: `cali_d_ci_width`, `cali_d_ci_width_rel`, `perc_rev_sla_d_ci_width`, `perc_rev_sla_d_ci_width_rel`.
- Run `winsorization_revenue_ci_summary` and check for `conversion_cut_out`, `conversion_cut_out_pct`.
- In helpers, search for `DATE(served_date_pst)` to confirm week truncation.

If you tell me which part you’re redoing first (e.g. “variance cell” or “advertiser AB”), I can give the exact code for that cell.
