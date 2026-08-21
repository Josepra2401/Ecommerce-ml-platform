# =============================================================
#  forecasting.py
#  Sales forecasting using Facebook Prophet
#  FIXED: CV horizon capped to actual forecast horizon (7 days)
#         Period changed to 7 days so CV starts from day 4-5
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


# -------------------------------------------------------------
#  HELPER — compute correct CV parameters based on resample mode
#  THIS IS THE CORE FIX for the unit mismatch bug
# -------------------------------------------------------------

def _cv_params(n_rows: int, resample: str, forecast_horizon: int) -> tuple[str, str, str]:
    """
    Return (initial, period, horizon) strings for Prophet cross_validation().

    KEY DESIGN DECISIONS:
    1. CV horizon is capped to the actual forecast horizon the user chose.
       No point testing 146 days if you only forecast 7 days ahead.
       This keeps the CV table short, fast, and directly meaningful.

    2. Period is set to 7 days for all modes so the CV table starts
       from day 4-5 instead of day 15 (which happened with period=30 days).

    3. Unit conversion: n_rows is in weeks/months for W/M resample,
       so we multiply by unit_days to get the correct number of days.
    """
    if resample == "D":
        unit_days   = 1
        min_initial = 180
    elif resample == "W":
        unit_days   = 7
        min_initial = 180
    else:  # "M"
        unit_days   = 30
        min_initial = 365

    # Period = 7 days for all modes → CV starts from ~day 4-5
    # (previously 30 days → started from day 15, which was confusing)
    period = "7 days"

    initial_days = max(min_initial, int(n_rows * 0.5) * unit_days)

    # ✅ KEY FIX: cap CV horizon to the actual forecast horizon
    # CV should only test as far ahead as you actually forecast
    horizon_days = min(forecast_horizon, int(n_rows * 0.2) * unit_days)
    horizon_days = max(horizon_days, unit_days)          # at least 1 unit
    horizon_days = min(horizon_days, initial_days - 1)  # must be < initial

    return (
        f"{initial_days} days",
        period,
        f"{horizon_days} days",
    )


# -------------------------------------------------------------
#  BLOCK 1 — prepare_for_forecast
# -------------------------------------------------------------

def prepare_for_forecast(df: pd.DataFrame, date_col: str,
                         value_col: str, resample: str = "W") -> pd.DataFrame:
    """
    Clean and prepare raw transactions into a time series ready for Prophet.

    - Strips timezone from dates (Prophet requires tz-naive)
    - Aggregates duplicates on the same date (sum, not keep-first)
    - Fills date gaps with NaN instead of 0 (Prophet interpolates gaps)
    - Auto-detects closure days (isolated zeros) and marks as NaN
    - Uses 3-sigma rolling cap instead of hard 99th-percentile clip
      so real seasonal spikes (Black Friday / Diwali etc.) are preserved
    - Resamples AFTER cleaning so aggregation doesn't hide gaps
    - Shows a data quality report in the UI before training
    """
    df2 = df.copy()

    # Step 1 — parse dates, strip timezone, normalize to date only
    df2[date_col] = (
        pd.to_datetime(df2[date_col], errors="coerce")
          .dt.tz_localize(None)   # Prophet breaks on tz-aware timestamps
          .dt.floor("D")          # drop time component
    )
    df2[value_col] = pd.to_numeric(df2[value_col], errors="coerce")
    df2 = df2.dropna(subset=[date_col])

    # Step 2 — aggregate by date (handles duplicate rows for same date)
    daily = (
        df2.groupby(df2[date_col].dt.date)[value_col]
           .sum()
           .reset_index()
    )
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])

    # Step 3 — reindex to full date range, gaps become NaN (not 0)
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = (
        daily.set_index("ds")
             .reindex(full_range)
             .reset_index()
             .rename(columns={"index": "ds"})
    )

    # Step 4 — auto-detect closure days (isolated zeros) → NaN
    # Logic: if y=0 AND the 7-day rolling median is above the 10th percentile,
    # it's likely a closure day, not a real zero-sales period
    rolling_med = daily["y"].rolling(7, center=True, min_periods=1).median()
    is_closure  = (daily["y"] == 0) & (rolling_med > daily["y"].quantile(0.1))
    daily.loc[is_closure, "y"] = None

    # Step 5 — smart outlier cap: only flag true anomalies (3-sigma)
    # Does NOT cap real seasonal spikes like Black Friday / Diwali
    roll_mean = daily["y"].rolling(30, center=True, min_periods=7).mean()
    roll_std  = daily["y"].rolling(30, center=True, min_periods=7).std()
    upper_cap = roll_mean + 3 * roll_std
    anomaly   = daily["y"] > upper_cap
    daily.loc[anomaly, "y"] = upper_cap[anomaly]
    daily["y"] = daily["y"].clip(lower=0)

    # Step 6 — resample AFTER cleaning
    if resample != "D":
        freq_map = {"W": "W-MON", "M": "MS"}
        daily = (
            daily.set_index("ds")
                 .resample(freq_map[resample])["y"]
                 .sum()
                 .reset_index()
        )

    # Warn if too few rows for the chosen resample
    min_rows = {"D": 60, "W": 26, "M": 18}
    if len(daily) < min_rows.get(resample, 30):
        st.warning(
            f"⚠️ Only {len(daily)} {resample} periods found. "
            f"Minimum recommended is {min_rows[resample]}. "
            f"Consider using a finer resample or uploading more data."
        )

    # Data quality report shown in UI before training starts
    gap_count  = daily["y"].isna().sum()
    zero_count = (daily["y"] == 0).sum()
    st.info(
        f"📊 Training on **{len(daily)}** {resample} periods &nbsp;|&nbsp; "
        f"**{gap_count}** gaps (NaN) &nbsp;|&nbsp; "
        f"**{zero_count}** genuine zero-sales periods &nbsp;|&nbsp; "
        f"{daily['ds'].min().date()} → {daily['ds'].max().date()}"
    )

    return daily


# -------------------------------------------------------------
#  BLOCK 2 — fit_prophet_model
# -------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fit_prophet_model(
    df_hash: str,
    df_daily: pd.DataFrame,
    daily_s: bool,
    weekly_s: bool,
    yearly_s: bool,
    country: str = "US",
) -> Prophet:
    """
    Train Prophet model with improved accuracy settings.

    - Stable cache key (passed in as df_hash)
    - changepoint_prior_scale = 0.05 (smoother trend, less cliff at end)
    - seasonality_prior_scale = 1.5  (less seasonal overfitting)
    - Monthly seasonality component added (payday / month-end spikes)
    - Country holiday regressors prevent spikes bleeding into trend
    """
    model = Prophet(
        daily_seasonality=daily_s,
        weekly_seasonality=weekly_s,
        yearly_seasonality=yearly_s,
        changepoint_prior_scale=0.05,   # lowered from 0.15 → smoother trend
        seasonality_prior_scale=1.5,    # was 10 → less seasonal overfitting
        interval_width=0.80,
    )

    # Monthly seasonality — captures end-of-month / payday patterns
    # Needs 12+ months of data to be meaningful
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    # Country holidays — stops spike days corrupting the trend baseline
    try:
        model.add_country_holidays(country_name=country)
    except Exception:
        pass  # silently skip if country code is invalid

    model.fit(df_daily)
    return model


# -------------------------------------------------------------
#  BLOCK 3 — run_forecast_tab
# -------------------------------------------------------------

def run_forecast_tab(raw_df: pd.DataFrame, date_guess: str, num_guess: str):
    """Streamlit UI for the Sales Forecast tab."""
    st.header("📈 Sales Forecast — Prophet")
    st.markdown("Configure columns and hit **Run Forecast** to generate predictions.")

    # Column selectors
    col1, col2, col3 = st.columns(3)
    with col1:
        date_col = st.selectbox(
            "Date column", raw_df.columns,
            index=raw_df.columns.get_loc(date_guess) if date_guess in raw_df.columns else 0,
            key="fc_date",
        )
    with col2:
        value_col = st.selectbox(
            "Sales / Value column",
            [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])],
            key="fc_val",
        )
    with col3:
        resample = st.selectbox(
            "Resample frequency", ["W", "M", "D"], index=0, key="fc_res"
        )

    st.info("💡 **Tip:** Weekly (W) gives the smoothest forecast. Use Monthly (M) for 3+ years of data.")

    # Forecast settings
    st.subheader("Forecast settings")
    c1, c2, c3, c4, c5 = st.columns(5)
    horizon  = c1.slider("Horizon (days)", 7, 365, 7)  # default 7 days
    daily_s  = c2.checkbox("Daily seasonality",  value=False)
    weekly_s = c3.checkbox("Weekly seasonality", value=True)
    yearly_s = c4.checkbox("Yearly seasonality", value=True)

    country = c5.selectbox(
        "Holidays",
        ["US", "GB", "IN", "AU", "CA", "DE", "FR", "SG", "None"],
        index=2,   # default to IN (India)
        key="fc_country",
    )

    run_cv = st.checkbox(
        "Run cross-validation after forecast (measures actual MAPE / RMSE)",
        value=False,
        key="fc_cv",
    )

    if st.button("🚀 Run Forecast", key="btn_forecast"):
        try:
            # --- Data prep ---
            with st.spinner("Preparing and cleaning data…"):
                df_daily = prepare_for_forecast(raw_df, date_col, value_col, resample)

            st.subheader("Time-series preview")
            st.dataframe(df_daily.tail(10), use_container_width=True)

            # --- Model training ---
            with st.spinner("Training Prophet model…"):
                y_sum   = round(float(df_daily["y"].sum()), 2)
                df_hash = (
                    f"{len(df_daily)}_{df_daily['ds'].min()}_"
                    f"{df_daily['ds'].max()}_{y_sum}_{resample}"
                )

                model = fit_prophet_model(
                    df_hash, df_daily,
                    daily_s, weekly_s, yearly_s,
                    country=country if country != "None" else "US",
                )

                freq_map = {
                    "D": ("D",  horizon),
                    "W": ("W",  max(1, horizon // 7)),
                    "M": ("MS", max(1, horizon // 30)),
                }
                freq_str, n_periods = freq_map.get(resample, ("D", horizon))
                future   = model.make_future_dataframe(periods=n_periods, freq=freq_str)
                forecast = model.predict(future)

            st.success("✅ Forecast completed!")

            # --- KPIs ---
            last_actual = df_daily["y"].dropna().iloc[-1]
            next_pred   = forecast["yhat"].iloc[-n_periods]
            k1, k2, k3 = st.columns(3)
            k1.metric("Last Actual",      f"{last_actual:,.0f}")
            k2.metric("Next Period Pred", f"{next_pred:,.0f}")
            k3.metric("Forecast Horizon", f"{horizon} days ({n_periods} {resample} periods)")

            # --- Plotly chart ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_daily["ds"], y=df_daily["y"],
                mode="lines", name="Actual",
                line=dict(color="#00d4aa", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                mode="lines", name="Forecast",
                line=dict(color="#a78bfa", width=1.5, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(167,139,250,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% interval", showlegend=True,
            ))
            fig.update_layout(
                title="Sales Forecast",
                xaxis_title="Date", yaxis_title="Sales",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=0),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Forecast table ---
            st.subheader("Forecast table (last 10 rows)")
            st.dataframe(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
                    "ds": "Date",
                    "yhat": "Predicted Sales",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound",
                }).tail(10).round(2),
                use_container_width=True,
            )

            # --- Cross-validation (FIXED unit mismatch) ---
            if run_cv:
                with st.spinner("Running cross-validation — this may take a minute…"):
                    n_rows              = len(df_daily)

                    # ✅ FIX: pass forecast horizon so CV only tests 7 days ahead
                    initial_str, period_str, horizon_str = _cv_params(
                        n_rows, resample, horizon
                    )

                    st.caption(
                        f"🔍 CV window — initial: **{initial_str}** | "
                        f"period: **{period_str}** | "
                        f"horizon: **{horizon_str}** "
                        f"*(matches your {horizon}-day forecast)*"
                    )

                    df_cv   = cross_validation(
                        model,
                        initial=initial_str,
                        period=period_str,
                        horizon=horizon_str,
                    )
                    metrics = performance_metrics(df_cv)

                st.subheader("Cross-validation metrics")
                st.dataframe(
                    metrics[["horizon", "mape", "rmse", "mae"]].round(3),
                    use_container_width=True,
                )
                st.caption(
                    f"MAPE = mean absolute % error. Lower is better. "
                    f"This table shows accuracy only for your **{horizon}-day** "
                    f"forecast window — the range you actually use for decisions."
                )

                # Warn if MAPE degrades sharply
                if len(metrics) > 1:
                    first_mape = metrics["mape"].iloc[0]
                    last_mape  = metrics["mape"].iloc[-1]
                    if last_mape > first_mape * 2:
                        st.warning(
                            f"⚠️ MAPE more than doubled from "
                            f"{first_mape:.3f} → {last_mape:.3f} across the horizon. "
                            f"Treat longer-range predictions as rough directional signals only."
                        )

            # --- Download ---
            csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
                "ds": "Date", "yhat": "Predicted_Sales",
                "yhat_lower": "Lower_Bound", "yhat_upper": "Upper_Bound",
            }).to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Forecast CSV", csv,
                "forecast_output.csv", "text/csv",
            )

        except Exception as e:
            st.error(f"Forecast error: {e}")
            st.exception(e)
