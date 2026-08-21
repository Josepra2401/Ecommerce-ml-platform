# =============================================================
#  pipeline.py
#  Orchestrates the full ML pipeline across all 3 modules.
#  Call run_pipeline() to execute all modules on a dataframe.
# =============================================================

import pandas as pd
from modules.data_loader import guess_columns
from modules.forecasting import prepare_for_forecast, fit_prophet_model
from modules.segmentation import prepare_for_rfm, segment_rfm
from modules.geo_analysis import try_geocode_cities


def run_pipeline(df: pd.DataFrame,
                 resample: str = "W",
                 horizon_days: int = 90,
                 n_clusters: int = 3,
                 city_col: str = "city",
                 value_col: str = "total_price") -> dict:
    """
    Full end-to-end pipeline:
      1. Auto-detect columns
      2. Sales forecast (Prophet)
      3. Customer segmentation (RFM + KMeans)
      4. Geo analysis (city aggregation + geocoding)

    Returns a dict with keys: forecast, rfm, geo, columns
    """
    # ── Step 1: Column detection ──────────────────────────────
    date_col, num_col, cust_col = guess_columns(df)
    print(f"[Pipeline] Detected → date: {date_col} | value: {num_col} | customer: {cust_col}")

    results = {"columns": {"date": date_col, "value": num_col, "customer": cust_col}}

    # ── Step 2: Sales Forecast ────────────────────────────────
    try:
        print("[Pipeline] Running forecasting module...")
        df_daily = prepare_for_forecast(df, date_col, num_col, resample)
        df_hash  = str(pd.util.hash_pandas_object(df_daily).sum())
        model    = fit_prophet_model(df_hash, df_daily,
                                     daily_s=False, weekly_s=True, yearly_s=True)
        freq_map = {"D": ("D", horizon_days),
                    "W": ("W", max(1, horizon_days // 7)),
                    "M": ("MS", max(1, horizon_days // 30))}
        freq_str, n_periods = freq_map.get(resample, ("D", horizon_days))
        future   = model.make_future_dataframe(periods=n_periods, freq=freq_str)
        forecast = model.predict(future)
        results["forecast"] = {
            "model":    model,
            "df_daily": df_daily,
            "forecast": forecast[["ds", "yhat"]],
        }
        print(f"[Pipeline] Forecast done — {n_periods} {resample} periods ahead.")
    except Exception as e:
        print(f"[Pipeline] Forecast failed: {e}")
        results["forecast"] = None

    # ── Step 3: Customer Segmentation ────────────────────────
    try:
        print("[Pipeline] Running segmentation module...")
        df_rfm = prepare_for_rfm(df, date_col, cust_col)
        rfm_df = segment_rfm(df_rfm, date_col, cust_col, n_clusters)
        results["rfm"] = rfm_df
        print(f"[Pipeline] Segmentation done — {len(rfm_df):,} customers, "
              f"{rfm_df['SegmentLabel'].nunique()} segments.")
    except Exception as e:
        print(f"[Pipeline] Segmentation failed: {e}")
        results["rfm"] = None

    # ── Step 4: Geo Analysis ──────────────────────────────────
    try:
        print("[Pipeline] Running geo module...")
        df_geo = df.copy()
        df_geo[value_col] = pd.to_numeric(df_geo[value_col], errors="coerce").fillna(0)
        df_geo[city_col]  = df_geo[city_col].astype(str).str.strip().str.title()
        agg = (df_geo.groupby(city_col)[value_col]
                     .sum().reset_index()
                     .rename(columns={city_col: "City", value_col: "Sales"}))
        agg = agg.sort_values("Sales", ascending=False).reset_index(drop=True)
        geo = try_geocode_cities(agg, "City")
        results["geo"] = {"agg": agg, "geocoded": geo}
        print(f"[Pipeline] Geo done — {len(agg)} cities, {len(geo)} geocoded.")
    except Exception as e:
        print(f"[Pipeline] Geo failed: {e}")
        results["geo"] = None

    print("[Pipeline] ✅ All modules complete.")
    return results


# ── Quick test when run directly ──────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        results = run_pipeline(df)
        print("\n=== Pipeline Results ===")
        if results["forecast"]:
            fc = results["forecast"]["forecast"]
            print(f"Forecast rows : {len(fc)}")
            print(fc.tail(3).to_string())
        if results["rfm"] is not None:
            rfm = results["rfm"]
            print(f"\nRFM customers : {len(rfm)}")
            print(rfm["SegmentLabel"].value_counts().to_string())
        if results["geo"]:
            print(f"\nTop 5 cities  :")
            print(results["geo"]["agg"].head(5).to_string())
    else:
        print("Usage: python pipeline.py <your_data.csv>")
