# =============================================================
#  segmentation.py
#  Customer segmentation using RFM analysis + KMeans clustering
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def prepare_for_rfm(df: pd.DataFrame, date_col: str, cust_col: str,
                    qty_col=None, price_col=None) -> pd.DataFrame:
    """
    Clean and prepare dataframe for RFM computation.
    Builds a 'sales' column from qty*price or falls back to a total column.
    Strips time from date so recency is calculated in whole days.
    """
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce").dt.floor("D")

    # Build sales column
    if qty_col and price_col and qty_col in df2.columns and price_col in df2.columns:
        df2["sales"] = (
            pd.to_numeric(df2[qty_col],   errors="coerce").fillna(0) *
            pd.to_numeric(df2[price_col], errors="coerce").fillna(0)
        )
    else:
        total_col = next(
            (c for c in df2.columns if any(
                k in c for k in ["total", "amount", "sales", "revenue"]
            )),
            None,
        )
        if total_col:
            df2["sales"] = pd.to_numeric(df2[total_col], errors="coerce").fillna(0)
        else:
            st.warning("⚠️ No sales column found — Monetary = transaction count.")
            df2["sales"] = 1

    df2 = df2.dropna(subset=[date_col, cust_col])
    return df2


def segment_rfm(df_transactions: pd.DataFrame, date_col: str,
                cust_col: str, n_clusters: int = 3) -> pd.DataFrame:
    """
    Compute RFM metrics and apply KMeans clustering.
    Uses direct dataframe — no JSON serialisation (avoids timestamp corruption).
    """
    snapshot_date = df_transactions[date_col].max() + pd.Timedelta(days=1)

    # Aggregate per customer
    grp       = df_transactions.groupby(cust_col)
    recency   = grp[date_col].apply(lambda x: (snapshot_date - x.max()).days)
    frequency = grp[date_col].count()
    monetary  = grp["sales"].sum()

    rfm = pd.DataFrame({
        "Recency":   recency,
        "Frequency": frequency,
        "Monetary":  monetary,
    }).reset_index().rename(columns={cust_col: "CustomerID"})

    # R / F / M quartile scores (rank avoids ties)
    r_labels = pd.qcut(rfm["Recency"].rank(method="first"),   4, labels=[4, 3, 2, 1])
    f_labels = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    m_labels = pd.qcut(rfm["Monetary"].rank(method="first"),  4, labels=[1, 2, 3, 4])
    rfm["R"], rfm["F"], rfm["M"] = (
        r_labels.astype(int), f_labels.astype(int), m_labels.astype(int)
    )
    rfm["RFM_Score"] = rfm["R"] * 100 + rfm["F"] * 10 + rfm["M"]

    # KMeans — manual normalisation (mean / std)
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X = (X - X.mean()) / (X.std().replace(0, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X)

    # Segment labels
    def label_customer(row):
        r, f, m = row["R"], row["F"], row["M"]
        if r >= 3 and f >= 3 and m >= 3:   return "💎 Loyal"
        elif r <= 2 and f <= 2:             return "😴 Churned"
        elif r >= 3 and f <= 2 and m <= 2: return "🆕 New"
        else:                               return "🔄 Others"

    rfm["SegmentLabel"] = rfm.apply(label_customer, axis=1)
    return rfm


def run_rfm_tab(raw_df: pd.DataFrame, date_guess: str, cust_guess: str):
    """Streamlit UI for the Customer Segmentation tab."""
    st.header("⚙️ Customer Segmentation — RFM + KMeans")
    st.markdown("Segments customers by **Recency, Frequency, Monetary** value using KMeans.")

    c1, c2, c3, c4 = st.columns(4)
    date_col  = c1.selectbox(
        "Date column", raw_df.columns,
        index=raw_df.columns.get_loc(date_guess) if date_guess in raw_df.columns else 0,
        key="rfm_date",
    )
    cust_col  = c2.selectbox(
        "Customer column", raw_df.columns,
        index=raw_df.columns.get_loc(cust_guess) if cust_guess in raw_df.columns else 0,
        key="rfm_cust",
    )
    qty_col   = c3.selectbox("Qty column (opt)",   [None] + list(raw_df.columns), index=0, key="rfm_qty")
    price_col = c4.selectbox("Price column (opt)", [None] + list(raw_df.columns), index=0, key="rfm_price")
    n_clusters = st.slider("K (clusters)", 2, 6, 3, key="rfm_k")

    if st.button("🚀 Run RFM Segmentation", key="btn_rfm"):
        try:
            with st.spinner("Computing RFM…"):
                df_ready = prepare_for_rfm(raw_df, date_col, cust_col, qty_col, price_col)
                rfm_df   = segment_rfm(df_ready, date_col, cust_col, n_clusters)

            st.success("✅ Segmentation done!")

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Customers",    f"{len(rfm_df):,}")
            k2.metric("Avg Recency (days)", f"{rfm_df['Recency'].mean():.0f}")
            k3.metric("Avg Frequency",      f"{rfm_df['Frequency'].mean():.1f}")
            k4.metric("Avg Monetary",       f"{rfm_df['Monetary'].mean():,.0f}")

            # Pie chart
            counts = rfm_df["SegmentLabel"].value_counts()
            colors = ["#6c3bdb", "#00d4aa", "#f97316", "#e11d48", "#facc15"]
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#0f0f1a")
            ax.set_facecolor("#0f0f1a")
            labels_pct = [f"{lbl} ({round(100*v/counts.sum())}%)" for lbl, v in counts.items()]
            ax.pie(
                counts.values, labels=labels_pct, startangle=90,
                colors=colors[:len(counts)],
                textprops={"color": "white", "fontsize": 10},
            )
            ax.set_title("Customer Segments", color="white", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Per-segment tables + download
            st.subheader("Segments")
            for label in rfm_df["SegmentLabel"].unique():
                sub = rfm_df[rfm_df["SegmentLabel"] == label]
                with st.expander(f"{label}  —  {len(sub)} customers"):
                    st.dataframe(sub.head(10), use_container_width=True)
                    csv_b = sub.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"⬇️ Download {label} CSV", csv_b,
                        f"{label.split()[-1].lower()}_customers.csv", "text/csv",
                        key=f"dl_{label}",
                    )

        except Exception as e:
            st.error(f"RFM error: {e}")
