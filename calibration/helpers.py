"""
Helper functions for calibration analysis.
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from scipy.stats import norm

# Default BigQuery client setup
project_id = 'sc-bq-ds-adhoc'
client = bigquery.Client(project=project_id)


def calculate_sla_metrics(table_name, gbb_filter=None, is_sample=False, client=client):
    """
    Calculates SLA metrics for either the whole population or simulation samples.

    Parameters:
    - table_name: Name of the BigQuery table to query.
    - gbb_filter: Specific optimization goal to filter by.
    - is_sample: If True, uses the seeded/simulated table and groups by sim_id.
    - client: BigQuery client instance.
    
    Returns:
    - DataFrame with line item level SLA metrics.
    """

    # Determine grouping based on mode
    sim_col = "sim_id," if is_sample else ""
    sim_group = "sim_id," if is_sample else ""

    condition = f"and optimization_goal = '{gbb_filter}'" if gbb_filter else ""

    query = f"""
    with agg_li as (
        select
            {sim_col}
            line_item_id,
            count(serve_item_id) as impressions,
            sum(conversions) as conversions,
            sum(predicted_conversions) as conversions_pred,
            sum(revenue) as revenue
        from `sc-analytics-privacy.ad_hoc_analysis.{table_name}`
        where 1=1
        {condition}
        group by all
    )
    select
        *,
        # case 
        #     when conversions = 0 then 0
        # else (conversions_pred / conversions) end as calibration,
        case
            when conversions > 0
                 and (conversions_pred / conversions) > 0.8
                 and (conversions_pred / conversions) < 1.2
            then 1 else 0
        end as sla_flag,
    from agg_li
    """

    df_li = client.query(query).to_dataframe()

    # Pre-calculate Revenue weighted SLA value row-wise
    df_li['rev_sla'] = df_li['sla_flag'] * df_li['revenue']

    return df_li



def all_traffic_data(start_date, end_date):
    query = f"""
          SELECT
          line_item_id,
          model_type,
          bid_strategy,
          optimization_goal,
          winner_bgid,
          SUM(impressions) AS impressions,
          COALESCE(SUM(
              CASE
                  -- 7-0 conversions for non app deep funnel 7-0 products
                  WHEN (optimization_goal NOT IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                      AND (model_type LIKE '%7_0%')
                  THEN IF(optimization_goal LIKE '%_VO', conversions_value_7_0, conversions_7_0)

                  -- lp v2 conversions (2-0) for app deep funnel 7-0 products and 2-0 conversion for app reengage purchase 7-0
                  WHEN (optimization_goal IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                      AND (model_type LIKE '%7_0%')
                  THEN IF(optimization_goal LIKE '%_VO', conversions_value_2_0, conversions_2_0)

                  -- lp v2 conversions (2-1) for app deep funnel 28-1 products and 2-1 conversion for app reengage purchase 28-1
                  WHEN (optimization_goal IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                      AND (model_type NOT LIKE '%7_0%')
                  THEN IF(optimization_goal LIKE '%_VO', conversions_value_2_1, conversions_2_1)

                  -- other 28-1 products
                  ELSE IF(optimization_goal LIKE '%_VO', conversions_value_1_1, IF(optimization_goal LIKE 'PIXEL%', conversions_1_1, conversions_2_1))
              END
          ),0) AS conversions,
          COALESCE(SUM(
              IF(optimization_goal LIKE '%_VO' OR optimization_goal LIKE '%ROAS',
                predicted_conversions_value,
                predicted_conversions)
          ),0) AS predicted_conversions,
          SUM(revenue) AS revenue
      FROM
          `snap-advanced-matching-data.monetization_ab.final_metrics_global`
      WHERE
          served_date_pst BETWEEN '{start_date}' AND '{end_date}'
          -- AND is_pixel_spammy = FALSE # pixel severity
          AND NOT (SELECT 2 IN UNNEST(app_anomaly_severity))
          AND ad_account_is_house_ad_account is False
          AND optimization_goal != 'IMPRESSIONS'
      GROUP BY ALL
      """
    query_job = client.query(
            query
        )
    df = query_job.to_dataframe()

    return df


def calibration_ab_data(start_date, end_date, macrostate, treatment_bgid, control_bgid, gbb_filter =None, spammy_filter=True, house_ads_filter=True):
    """
    Compute calibration and spend weighted calibration for Pixel GBBs in AB analysis
    """
    # Check for valid macrostate user entry
    macrostate_upper = macrostate.upper()
    macrostate = macrostate.lower()
    if macrostate not in ['classical', 'pixel_star', 'app_star', 'amalgam', 'swipe']:
        print('\n Invalid macrostate entered as a parameter. Valid macrostate values: \'classical\', \'pixel_star\', \'app_star\', \'amalgam\', \'swipe\' \n')
        sys.exit('Execution halted due to invalid input.')

    if spammy_filter:
       spammy_bq_filter = """
       AND is_pixel_spammy = FALSE # pixel severity
       AND NOT (SELECT 2 IN UNNEST(app_anomaly_severity)) # app severity (new version, as of March 2024)
       """
    else:
       spammy_bq_filter = ''

    if house_ads_filter:
       house_ads_bq_filter = '  AND ad_account_is_house_ad_account is False # exclude house ads'
    else:
       house_ads_bq_filter = ''

    if gbb_filter is not None:
       gbb_filter  = f"  AND optimization_goal = '{gbb_filter}'"
    else:
       gbb_filter = ''

    query = f"""
    WITH account_size_label AS
    (
      SELECT
      ad_account_id,
      adv_size_by_revenue,
      ROW_NUMBER() OVER (PARTITION BY ad_account_id ORDER BY account_updated_at desc) as ranked
      FROM `snap-advanced-matching-data.monetization_ab.final_metrics_global`
      WHERE
      served_date_pst BETWEEN DATE_SUB('{start_date}', INTERVAL 7 DAY) AND '{start_date}' --may not be necessary as the field seems to be updated daily
    )
    SELECT
        line_item_id,
        main.ad_account_id,
        coalesce(acct.adv_size_by_revenue, 'Unknown') AS adv_size_by_revenue,
        model_type,
        bid_strategy,
        optimization_goal,
        IF(bgid = '{control_bgid}', 'control', 'treatment') AS treatment_group,
        SUM(impressions) AS impressions,
        COALESCE(SUM(
            CASE
                -- 7-0 conversions for non app deep funnel 7-0 products
                WHEN (optimization_goal NOT IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                    AND (model_type LIKE '%7_0%')
                THEN IF(optimization_goal LIKE '%_VO', conversions_value_7_0, conversions_7_0)

                -- lp v2 conversions (2-0) for app deep funnel 7-0 products and 2-0 conversion for app reengage purchase 7-0
                WHEN (optimization_goal IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                    AND (model_type LIKE '%7_0%')
                THEN IF(optimization_goal LIKE '%_VO', conversions_value_2_0, conversions_2_0)

                -- lp v2 conversions (2-1) for app deep funnel 28-1 products and 2-1 conversion for app reengage purchase 28-1
                WHEN (optimization_goal IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                    AND (model_type NOT LIKE '%7_0%')
                THEN IF(optimization_goal LIKE '%_VO', conversions_value_2_1, conversions_2_1)

                -- other 28-1 products
                ELSE IF(optimization_goal LIKE '%_VO', conversions_value_1_1, IF(optimization_goal LIKE 'PIXEL%', conversions_1_1, conversions_2_1))
            END
        ),0) AS conversions,
        COALESCE(SUM(
            IF(optimization_goal LIKE '%_VO' OR optimization_goal LIKE '%ROAS',
               predicted_conversions_value,
               predicted_conversions)
        ),0) AS predicted_conversions,
        SUM(revenue) AS revenue
    FROM
        `snap-advanced-matching-data.monetization_ab.final_metrics_macrostate_{macrostate_upper}` AS main
    LEFT JOIN
        account_size_label AS acct
    ON
        main.ad_account_id = acct.ad_account_id
        AND acct.ranked = 1
    WHERE
        served_date_pst BETWEEN '{start_date}' AND '{end_date}'
        AND bgid IN ('{control_bgid}', '{treatment_bgid}')
        {spammy_bq_filter}
        {house_ads_bq_filter}
        {gbb_filter}
        AND optimization_goal != 'IMPRESSIONS'
    GROUP BY ALL
    """

    query_job = client.query(
            query
        )
    df = query_job.to_dataframe()

    return df



def weighted_field_calibration(field_weight, field_pred, field_conversion):
    """
    Compute the weighted field level calibration with the standard error estimation using delta method
    ref: https://wiki.sc-corp.net/pages/viewpage.action?pageId=93422574

    Parameters:
    - field_weight: a list of field level weight, usually using line item as field and spend as weight
    - field_pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - field_conversion: a list of field level sum conversions

    Returns:
    - weighted_calibration: computed value of spend weighted calibration
    - weighted_calibration_var: the computed variance of spend weighted calibration
    - use +-1.96*sqrt(var) to construct the 95% confidence interval
    """

    field_weight = np.array(field_weight)
    field_calibration = np.array(field_pred)/ np.array(field_conversion)

    mean_n = np.mean(field_weight*field_calibration)
    mean_d = np.mean(field_weight)
    var_n = np.var(field_weight*field_calibration)
    var_d =np.var(field_weight)
    cov_nd = np.cov(field_weight*field_calibration, field_weight)[0][1]

    weighted_calibration = mean_n/mean_d
    weighted_calibration_var = (var_n/mean_d**2+var_d*mean_n**2/mean_d**4-2*cov_nd*mean_n/mean_d**3)/len(field_conversion)

    return weighted_calibration, weighted_calibration_var

def weighted_field_calibration_smoothed(field_weight, field_pred, field_conversion):
    """
    Compute the weighted field level calibration with add one smoothing

    Parameters:
    - field_weight: a list of field level weight, usually using line item as field and spend as weight
    - field_pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - field_conversion: a list of field level sum conversions

    Returns:
    - weighted_calibration: computed value of spend weighted calibration with add one smoothing
    """
    field_pred = np.array(field_pred)
    field_conversion = np.array(field_conversion)
    field_weight = np.array(field_weight)

    ind = field_conversion == 0

    field_pred[ind] += 1
    field_conversion[ind] += 1
    field_calibration = field_pred/field_conversion

    mean_n = np.mean(field_weight*field_calibration)
    mean_d = np.mean(field_weight)

    weighted_calibration = mean_n/mean_d

    return weighted_calibration

def weighted_absolute_calibration_error(field_weight, field_pred, field_conversion):
    """
    Compute the weighted absolute calibration error

    Parameters:
    - field_weight: a list of field level weight, usually using line item as field and spend as weight
    - field_pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - field_conversion: a list of field level sum conversions

    Returns:
    - weighted_absolute_calibration_error: computed value of spend weighted absolute calibration error
    - weighted_absolute_calibration_error_var: the computed variance of spend weighted absolute calibration error
    - weighted_absolute_calibration_error_symmetric: computed value of spend weighted absolute calibration error (symmetric version)
    - weighted_absolute_calibration_error_symmetric_var: the computed variance of spend weighted absolute calibration error (symmetric version)
    - use +-1.96*sqrt(var) to construct the 95% confidence interval
    """

    field_weight = np.array(field_weight)
    field_calibration = np.array(field_pred)/ np.array(field_conversion)
    field_error = abs(field_calibration - 1)
    field_error_symmetric = np.where(field_calibration > 1, 1 - 1/field_calibration, 1 - field_calibration)

    mean_n = np.mean(field_weight*field_error)
    mean_d = np.mean(field_weight)
    var_n = np.var(field_weight*field_error)
    var_d =np.var(field_weight)
    cov_nd = np.cov(field_weight*field_error, field_weight)[0][1]

    weighted_absolute_calibration_error = mean_n/mean_d
    weighted_absolute_calibration_error_var = (var_n/mean_d**2+var_d*mean_n**2/mean_d**4-2*cov_nd*mean_n/mean_d**3)/len(field_conversion)

    mean_n_symmetric = np.mean(field_weight*field_error_symmetric)
    var_n_symmetric = np.var(field_weight*field_error_symmetric)
    cov_nd_symmetric = np.cov(field_weight*field_error_symmetric, field_weight)[0][1]

    weighted_absolute_calibration_error_symmetric = mean_n_symmetric/mean_d
    weighted_absolute_calibration_error_symmetric_var = (var_n_symmetric/mean_d**2+var_d*mean_n_symmetric**2/mean_d**4-2*cov_nd_symmetric*mean_n_symmetric/mean_d**3)/len(field_conversion)

    return weighted_absolute_calibration_error, weighted_absolute_calibration_error_var, weighted_absolute_calibration_error_symmetric, weighted_absolute_calibration_error_symmetric_var


def calibration(pred, conversion):
    """
    Compute calibration with the standard error estimation using delta method
    ref: https://wiki.sc-corp.net/pages/viewpage.action?pageId=93422574

    Parameters:
    - pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - conversion: a list of field level sum conversions

    Returns:
    - calibration: computed value of calibration
    - calibration_var: the computed variance of calibration
    - use +-1.96*sqrt(var) to construct the 95% confidence interval
    """

    mean_n = np.mean(pred)
    mean_d = np.mean(conversion)
    if mean_d == 0 or not np.isfinite(mean_d):
        return 0.0, 0.0
    calibration = mean_n / mean_d
    # With <=1 observation we cannot estimate variance (np.cov uses ddof=1 and returns NaN)
    if len(conversion) <= 1:
        return calibration, 0.0
    var_n = np.var(pred)
    var_d = np.var(conversion)
    cov_nd = np.cov(pred, conversion)[0][1]
    calibration_var = (var_n/mean_d**2+var_d*mean_n**2/mean_d**4-2*cov_nd*mean_n/mean_d**3)/len(conversion)

    return calibration, calibration_var


def calculate_percent_revenue_in_sla(pred, conversion, revenue):
    """
    Compute % Rev in SLO with the standard error estimation using delta method
    ref: https://wiki.sc-corp.net/pages/viewpage.action?pageId=93422574

    Parameters:
    - pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - conversion: a list of field level sum conversions
    - revenue: a list of field level sum revenue

    Returns:
    - perc_rev_slo: computed value of % Rev in SLO
    - perc_rev_slo_var: the computed variance of % Rev in SLO
    - use +-1.96*sqrt(var) to construct the 95% confidence interval
    """

    # Compute calibration
    calibration = pred/conversion

    # Compute rev in / out of SLO
    in_slo_boolean = np.zeros_like(calibration)  # Initialize series with zeros
    in_slo_boolean[(calibration >= 0.8) & (calibration <= 1.2)] = 1  # In SLA range
    in_slo_rev = in_slo_boolean * revenue

    # numerator / denominator
    mean_n = np.mean(in_slo_rev)
    mean_d = np.mean(revenue)
    var_n = np.var(in_slo_rev)
    var_d = np.var(revenue)
    cov_nd = np.cov(in_slo_rev, revenue)[0][1]

    perc_rev_slo = mean_n/mean_d
    perc_rev_slo_var = (var_n/mean_d**2+var_d*mean_n**2/mean_d**4-2*cov_nd*mean_n/mean_d**3)/len(conversion)

    return perc_rev_slo, perc_rev_slo_var


def _calibration_analysis_one(df, z_score):
    """Run calibration metrics on a single df (one row per line_item_id). Returns one-row DataFrame."""
    cali_t, cali_t_var = calibration(df['predicted_conversions_treatment'], df['conversions_treatment'])
    cali_ci_t_l, cali_ci_t_r = cali_t - z_score * np.sqrt(cali_t_var), cali_t + z_score * np.sqrt(cali_t_var)
    cali_c, cali_c_var = calibration(df['predicted_conversions_control'], df['conversions_control'])
    cali_ci_c_l, cali_ci_c_r = cali_c - z_score * np.sqrt(cali_c_var), cali_c + z_score * np.sqrt(cali_c_var)
    cali_d = abs(cali_t - 1) - abs(cali_c - 1)
    cali_d_se = np.sqrt(cali_t_var + cali_c_var)
    cali_ci_d_l, cali_ci_d_r = cali_d - z_score * cali_d_se, cali_d + z_score * cali_d_se
    perc_rev_sla_t, perc_rev_sla_t_var = calculate_percent_revenue_in_sla(df['predicted_conversions_treatment'], df['conversions_treatment'], df['revenue_treatment'])
    perc_rev_sla_ci_t_l, perc_rev_sla_ci_t_r = perc_rev_sla_t - z_score * np.sqrt(perc_rev_sla_t_var), perc_rev_sla_t + z_score * np.sqrt(perc_rev_sla_t_var)
    perc_rev_sla_c, perc_rev_sla_c_var = calculate_percent_revenue_in_sla(df['predicted_conversions_control'], df['conversions_control'], df['revenue_control'])
    perc_rev_sla_ci_c_l, perc_rev_sla_ci_c_r = perc_rev_sla_c - z_score * np.sqrt(perc_rev_sla_c_var), perc_rev_sla_c + z_score * np.sqrt(perc_rev_sla_c_var)
    perc_rev_sla_d = perc_rev_sla_t - perc_rev_sla_c
    perc_rev_sla_d_se = np.sqrt(perc_rev_sla_t_var + perc_rev_sla_c_var)
    perc_rev_sla_ci_d_l, perc_rev_sla_ci_d_r = perc_rev_sla_d - z_score * perc_rev_sla_d_se, perc_rev_sla_d + z_score * perc_rev_sla_d_se

    # CI width and relative CI width (width / |point estimate|)
    cali_d_ci_width = cali_ci_d_r - cali_ci_d_l
    cali_d_ci_width_rel = (cali_d_ci_width / abs(cali_d)) if cali_d != 0 else np.nan
    perc_rev_sla_d_ci_width = perc_rev_sla_ci_d_r - perc_rev_sla_ci_d_l
    perc_rev_sla_d_ci_width_rel = (perc_rev_sla_d_ci_width / abs(perc_rev_sla_d)) if perc_rev_sla_d != 0 else np.nan

    return pd.DataFrame({
        'cali_t': ['{:.3f} ({:.3f}, {:.3f})'.format(cali_t, cali_ci_t_l, cali_ci_t_r)],
        'cali_c': ['{:.3f} ({:.3f}, {:.3f})'.format(cali_c, cali_ci_c_l, cali_ci_c_r)],
        'cali_d': [round(cali_d, 3)],
        'cali_d_se': [round(cali_d_se, 3)],
        'cali_d_ci': [(round(cali_ci_d_l, 3), round(cali_ci_d_r, 3))],
        'cali_d_ci_width': [round(cali_d_ci_width, 4)],
        'cali_d_ci_width_rel': [round(cali_d_ci_width_rel, 4) if np.isfinite(cali_d_ci_width_rel) else np.nan],
        'perc_rev_sla_t': ['{:.2%} ({:.2%}, {:.2%})'.format(perc_rev_sla_t, perc_rev_sla_ci_t_l, perc_rev_sla_ci_t_r)],
        'perc_rev_sla_c': ['{:.2%} ({:.2%}, {:.2%})'.format(perc_rev_sla_c, perc_rev_sla_ci_c_l, perc_rev_sla_ci_c_r)],
        'perc_rev_sla_d': ['{:.2%}'.format(perc_rev_sla_d)],
        'perc_rev_sla_d_se': ['{:.2%}'.format(perc_rev_sla_d_se)],
        'perc_rev_sla_d_ci': [('{:.2%}'.format(perc_rev_sla_ci_d_l), '{:.2%}'.format(perc_rev_sla_ci_d_r))],
        'perc_rev_sla_d_ci_width': [round(perc_rev_sla_d_ci_width, 4)],
        'perc_rev_sla_d_ci_width_rel': [round(perc_rev_sla_d_ci_width_rel, 4) if np.isfinite(perc_rev_sla_d_ci_width_rel) else np.nan],
    })


def calibration_analysis(df, threshold, confidence_interval):
    z_score = norm.ppf(1 - (1 - confidence_interval / 100) / 2)
    metric_cols = ['predicted_conversions_treatment', 'conversions_treatment', 'revenue_treatment',
                   'predicted_conversions_control', 'conversions_control', 'revenue_control']

    if 'optimization_goal' in df.columns:
        # Overall row first, then one row per optimization_goal
        df_agg_all = df.groupby('line_item_id', as_index=False)[metric_cols].sum()
        overall = _calibration_analysis_one(df_agg_all, z_score)
        overall.insert(0, 'optimization_goal', 'Overall')
        results = [overall]
        group_cols = ['line_item_id', 'optimization_goal']
        df_agg = df.groupby(group_cols, as_index=False)[metric_cols].sum()
        for g, sub in df_agg.groupby('optimization_goal', sort=False):
            sub_li = sub.drop(columns=['optimization_goal']).groupby('line_item_id', as_index=False)[metric_cols].sum()
            one = _calibration_analysis_one(sub_li, z_score)
            one.insert(0, 'optimization_goal', g)
            results.append(one)
        return pd.concat(results, ignore_index=True)

    # No optimization_goal: single aggregated row
    df_agg = df.groupby('line_item_id', as_index=False)[metric_cols].sum()
    return _calibration_analysis_one(df_agg, z_score)


def winsorize_by_dimension(df_wide, dimension_col, pct, direction='left'):
    """
    Winsorize a wide-format df by a dimension column (e.g. control's impression count).
    Drops line items in the left, right, or both tails of the dimension distribution.

    Parameters:
    - df_wide: DataFrame with one row per line_item_id (and optionally optimization_goal).
    - dimension_col: Column name used for the cut (e.g. 'impressions_control').
    - pct: Percentile for the cut (0-100). E.g. 5 means "cut 5% from that side" (or each side for 'both').
    - direction: 'left' = drop bottom pct%% (keep rows with dimension >= pct percentile);
                 'right' = drop top pct%% (keep rows with dimension <= (100-pct) percentile);
                 'both' = drop bottom pct%% and top pct%% (keep rows between pct and (100-pct) percentiles).

    Returns:
    - Filtered df_wide with same columns.
    """
    if dimension_col not in df_wide.columns:
        raise ValueError(f"dimension_col '{dimension_col}' not in df. Columns: {list(df_wide.columns)}")
    vals = df_wide[dimension_col].dropna()
    if len(vals) == 0:
        return df_wide.copy()
    if direction == 'left':
        threshold = np.percentile(vals, pct)
        return df_wide[df_wide[dimension_col] >= threshold].copy()
    elif direction == 'right':
        threshold = np.percentile(vals, 100 - pct)
        return df_wide[df_wide[dimension_col] <= threshold].copy()
    elif direction == 'both':
        low = np.percentile(vals, pct)
        high = np.percentile(vals, 100 - pct)
        return df_wide[(df_wide[dimension_col] >= low) & (df_wide[dimension_col] <= high)].copy()
    else:
        raise ValueError("direction must be 'left', 'right', or 'both'")


def _ci_width_from_tuple(ci_tuple):
    """CI as (low, high); return width. Handles numeric or percentage strings."""
    low, high = ci_tuple
    if isinstance(low, str):
        low = float(str(low).replace('%', '')) / 100.0
        high = float(str(high).replace('%', '')) / 100.0
    return high - low


def winsorization_revenue_ci_summary(df_wide, dimension_cols, directions, pct_levels, goals_to_keep=None, threshold=0, confidence_interval=95):
    """
    For each winsorization (dimension, direction, pct): revenue cut out and CI width / reduction vs baseline.
    Supports multiple optimization_goal segments (e.g. Overall and APP_INSTALLS).

    Parameters:
    - df_wide: Wide-format DataFrame (e.g. df_ab_agg_wide with optional optimization_goal).
    - dimension_cols: List of column names to winsorize by (e.g. ['impressions_control', 'conversions_control']).
    - directions: List of 'left', 'right', 'both'.
    - pct_levels: Array of cut percentiles (e.g. np.arange(0, 11) for 0-10%).
    - goals_to_keep: List of optimization_goal values to report (e.g. ['Overall', 'APP_INSTALLS']).
                    If None, uses Overall only when calibration_analysis returns optimization_goal.
    - threshold: Passed to calibration_analysis.
    - confidence_interval: Passed to calibration_analysis.

    Returns:
    - DataFrame with columns: dimension, direction, pct_cut, optimization_goal, revenue_cut_out,
      revenue_cut_out_pct, conversion_cut_out, conversion_cut_out_pct, cali_d_ci_width,
      cali_d_ci_reduction, cali_d_ci_reduction_pct, perc_rev_sla_d_ci_width, perc_rev_sla_d_ci_reduction,
      perc_rev_sla_d_ci_reduction_pct.
    """
    if goals_to_keep is None:
        goals_to_keep = ['Overall']
    rows = []
    for dim_col in dimension_cols:
        if dim_col not in df_wide.columns:
            continue
        df_dim = df_wide.dropna(subset=[dim_col])
        total_rev_full = (df_dim['revenue_control'] + df_dim['revenue_treatment']).sum()
        total_conv_full = (df_dim['conversions_control'] + df_dim['conversions_treatment']).sum()
        m0 = calibration_analysis(df_dim, threshold=threshold, confidence_interval=confidence_interval)
        if 'optimization_goal' not in m0.columns:
            m0 = m0.copy()
            m0['optimization_goal'] = 'Overall'
        m0 = m0[m0['optimization_goal'].isin(goals_to_keep)]
        baselines = {}
        for _, row0 in m0.iterrows():
            g = row0['optimization_goal']
            baselines[g] = {
                'cali_ci_width': _ci_width_from_tuple(row0['cali_d_ci']),
                'rev_ci_width': _ci_width_from_tuple(row0['perc_rev_sla_d_ci']),
            }
        for direction in directions:
            for pct in pct_levels:
                w = winsorize_by_dimension(df_dim, dim_col, pct, direction=direction)
                if len(w) < 2:
                    continue
                rev_kept = (w['revenue_control'] + w['revenue_treatment']).sum()
                revenue_cut_out = total_rev_full - rev_kept
                revenue_cut_out_pct = (revenue_cut_out / total_rev_full * 100) if total_rev_full > 0 else 0
                conv_kept = (w['conversions_control'] + w['conversions_treatment']).sum()
                conversion_cut_out = total_conv_full - conv_kept
                conversion_cut_out_pct = (conversion_cut_out / total_conv_full * 100) if total_conv_full > 0 else 0
                m = calibration_analysis(w, threshold=threshold, confidence_interval=confidence_interval)
                if 'optimization_goal' not in m.columns:
                    m = m.copy()
                    m['optimization_goal'] = 'Overall'
                m = m[m['optimization_goal'].isin(goals_to_keep)]
                for _, r in m.iterrows():
                    goal = r['optimization_goal']
                    if goal not in baselines:
                        continue
                    bl = baselines[goal]
                    cali_ci_w = _ci_width_from_tuple(r['cali_d_ci'])
                    rev_ci_w = _ci_width_from_tuple(r['perc_rev_sla_d_ci'])
                    rows.append({
                        'dimension': dim_col,
                        'direction': direction,
                        'pct_cut': pct,
                        'optimization_goal': goal,
                        'revenue_cut_out': revenue_cut_out,
                        'revenue_cut_out_pct': round(revenue_cut_out_pct, 2),
                        'conversion_cut_out': conversion_cut_out,
                        'conversion_cut_out_pct': round(conversion_cut_out_pct, 2),
                        'cali_d_ci_width': round(cali_ci_w, 4),
                        'cali_d_ci_reduction': round(bl['cali_ci_width'] - cali_ci_w, 4),
                        'cali_d_ci_reduction_pct': round((bl['cali_ci_width'] - cali_ci_w) / bl['cali_ci_width'] * 100, 2) if bl['cali_ci_width'] > 0 else 0,
                        'perc_rev_sla_d_ci_width': round(rev_ci_w, 4),
                        'perc_rev_sla_d_ci_reduction': round(bl['rev_ci_width'] - rev_ci_w, 4),
                        'perc_rev_sla_d_ci_reduction_pct': round((bl['rev_ci_width'] - rev_ci_w) / bl['rev_ci_width'] * 100, 2) if bl['rev_ci_width'] > 0 else 0,
                    })
    return pd.DataFrame(rows)


def weighted_field_calibration(field_weight, field_pred, field_conversion):
    """
    Compute the weighted field level calibration with the standard error estimation using delta method
    ref: https://wiki.sc-corp.net/pages/viewpage.action?pageId=93422574

    Parameters:
    - field_weight: a list of field level weight, usually using line item as field and spend as weight
    - field_pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - field_conversion: a list of field level sum conversions

    Returns:
    - weighted_calibration: computed value of spend weighted calibration
    - weighted_calibration_var: the computed variance of spend weighted calibration
    - use +-1.96*sqrt(var) to construct the 95% confidence interval
    """

    field_weight = np.array(field_weight)
    field_calibration = np.array(field_pred)/ np.array(field_conversion)

    mean_n = np.mean(field_weight*field_calibration)
    mean_d = np.mean(field_weight)
    var_n = np.var(field_weight*field_calibration)
    var_d =np.var(field_weight)
    cov_nd = np.cov(field_weight*field_calibration, field_weight)[0][1]

    weighted_calibration = mean_n/mean_d
    weighted_calibration_var = (var_n/mean_d**2+var_d*mean_n**2/mean_d**4-2*cov_nd*mean_n/mean_d**3)/len(field_conversion)

    return weighted_calibration, weighted_calibration_var

def weighted_field_calibration_smoothed(field_weight, field_pred, field_conversion):
    """
    Compute the weighted field level calibration with add one smoothing

    Parameters:
    - field_weight: a list of field level weight, usually using line item as field and spend as weight
    - field_pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - field_conversion: a list of field level sum conversions

    Returns:
    - weighted_calibration: computed value of spend weighted calibration with add one smoothing
    """
    field_pred = np.array(field_pred)
    field_conversion = np.array(field_conversion)
    field_weight = np.array(field_weight)

    ind = field_conversion == 0

    field_pred[ind] += 1
    field_conversion[ind] += 1
    field_calibration = field_pred/field_conversion

    mean_n = np.mean(field_weight*field_calibration)
    mean_d = np.mean(field_weight)

    weighted_calibration = mean_n/mean_d

    return weighted_calibration

def weighted_absolute_calibration_error(field_weight, field_pred, field_conversion):
    """
    Compute the weighted absolute calibration error

    Parameters:
    - field_weight: a list of field level weight, usually using line item as field and spend as weight
    - field_pred: a list of field level sum pevents, the order of the fields should be the same as other parameters
    - field_conversion: a list of field level sum conversions

    Returns:
    - weighted_absolute_calibration_error: computed value of spend weighted absolute calibration error
    - weighted_absolute_calibration_error_var: the computed variance of spend weighted absolute calibration error
    - weighted_absolute_calibration_error_symmetric: computed value of spend weighted absolute calibration error (symmetric version)
    - weighted_absolute_calibration_error_symmetric_var: the computed variance of spend weighted absolute calibration error (symmetric version)
    - use +-1.96*sqrt(var) to construct the 95% confidence interval
    """

    field_weight = np.array(field_weight)
    field_calibration = np.array(field_pred)/ np.array(field_conversion)
    field_error = abs(field_calibration - 1)
    field_error_symmetric = np.where(field_calibration > 1, 1 - 1/field_calibration, 1 - field_calibration)

    mean_n = np.mean(field_weight*field_error)
    mean_d = np.mean(field_weight)
    var_n = np.var(field_weight*field_error)
    var_d =np.var(field_weight)
    cov_nd = np.cov(field_weight*field_error, field_weight)[0][1]

    weighted_absolute_calibration_error = mean_n/mean_d
    weighted_absolute_calibration_error_var = (var_n/mean_d**2+var_d*mean_n**2/mean_d**4-2*cov_nd*mean_n/mean_d**3)/len(field_conversion)

    mean_n_symmetric = np.mean(field_weight*field_error_symmetric)
    var_n_symmetric = np.var(field_weight*field_error_symmetric)
    cov_nd_symmetric = np.cov(field_weight*field_error_symmetric, field_weight)[0][1]

    weighted_absolute_calibration_error_symmetric = mean_n_symmetric/mean_d
    weighted_absolute_calibration_error_symmetric_var = (var_n_symmetric/mean_d**2+var_d*mean_n_symmetric**2/mean_d**4-2*cov_nd_symmetric*mean_n_symmetric/mean_d**3)/len(field_conversion)

    return weighted_absolute_calibration_error, weighted_absolute_calibration_error_var, weighted_absolute_calibration_error_symmetric, weighted_absolute_calibration_error_symmetric_var

def all_traffic_data(start_date, end_date):
    query = f"""
          SELECT
          CAST(DATETIME_TRUNC(served_date_pst, WEEK(MONDAY)) AS DATE) AS week,
          line_item_id,
          model_type,
          bid_strategy,
          optimization_goal,
          winner_bgid,
          ad_account_id,
          adv_size_by_revenue,
          SUM(impressions) AS impressions,
          COALESCE(SUM(
              CASE
                  -- 7-0 conversions for non app deep funnel 7-0 products
                  WHEN (optimization_goal NOT IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                      AND (model_type LIKE '%7_0%')
                  THEN IF(optimization_goal LIKE '%_VO', conversions_value_7_0, conversions_7_0)

                  -- lp v2 conversions (2-0) for app deep funnel 7-0 products and 2-0 conversion for app reengage purchase 7-0
                  WHEN (optimization_goal IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                      AND (model_type LIKE '%7_0%')
                  THEN IF(optimization_goal LIKE '%_VO', conversions_value_2_0, conversions_2_0)

                  -- lp v2 conversions (2-1) for app deep funnel 28-1 products and 2-1 conversion for app reengage purchase 28-1
                  WHEN (optimization_goal IN ('APP_REENGAGE_PURCHASE', 'APP_PURCHASE', 'APP_PURCHASE_VO', 'APP_ADD_TO_CART', 'APP_SIGNUP', 'APP_AD_VIEW', 'APP_ACHIEVEMENT_UNLOCKED', 'APP_LEVEL_COMPLETE'))
                      AND (model_type NOT LIKE '%7_0%')
                  THEN IF(optimization_goal LIKE '%_VO', conversions_value_2_1, conversions_2_1)

                  -- other 28-1 products
                  ELSE IF(optimization_goal LIKE '%_VO', conversions_value_1_1, IF(optimization_goal LIKE 'PIXEL%', conversions_1_1, conversions_2_1))
              END
          ),0) AS conversions,
          COALESCE(SUM(
              IF(optimization_goal LIKE '%_VO' OR optimization_goal LIKE '%ROAS',
                predicted_conversions_value,
                predicted_conversions)
          ),0) AS predicted_conversions,
          SUM(revenue) AS revenue
      FROM
          `snap-advanced-matching-data.monetization_ab.final_metrics_global`
      WHERE
          served_date_pst BETWEEN '{start_date}' AND '{end_date}'
          -- AND is_pixel_spammy = FALSE # pixel severity
          AND NOT (SELECT 2 IN UNNEST(app_anomaly_severity))
          AND ad_account_is_house_ad_account is False
          AND optimization_goal != 'IMPRESSIONS'
      GROUP BY ALL
      """
    query_job = client.query(
            query
        )
    df = query_job.to_dataframe()

    return df