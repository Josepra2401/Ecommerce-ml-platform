# =============================================================
#  app.py  —  E-Commerce ML Platform
#  Entry point — imports all modules, renders 3-tab Streamlit UI
#  Run: streamlit run app.py
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from modules.data_loader  import load_raw, guess_columns
from modules.forecasting  import run_forecast_tab
from modules.segmentation import run_rfm_tab
from modules.geo_analysis import run_geo_tab

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce ML Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e; border-radius: 8px;
        color: #aaa; padding: 8px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #6c3bdb !important; color: white !important;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='padding:18px 0 8px 0'>
        <span style='font-size:11px;color:#845EF7;letter-spacing:4px;text-transform:uppercase'>
            Final Year Project
        </span><br>
        <span style='font-size:28px;font-weight:800;color:white'>
            📦 E-Commerce ML Platform
        </span><br>
        <span style='font-size:13px;color:#666'>
            Sales Forecast · Customer Segmentation · Geo Analysis
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # File upload — stored in session_state so it persists across tab switches
    if "raw_df" not in st.session_state:
        uploaded_file = st.file_uploader(
            "📂 Upload your transactions file (CSV or Excel)",
            type=["csv", "xls", "xlsx"],
        )
        if uploaded_file:
            try:
                raw_df = load_raw(uploaded_file)
                st.session_state["raw_df"] = raw_df
                st.success(f"✅ File loaded — {len(raw_df):,} rows × {len(raw_df.columns)} columns")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.info("👆 Upload a CSV or Excel file to get started.")
        return

    raw_df = st.session_state["raw_df"]
    date_g, num_g, cust_g = guess_columns(raw_df)

    info_col, clear_col = st.columns([5, 1])
    info_col.info(
        f"📄 **{len(raw_df):,} rows** loaded  |  "
        f"Auto-detected → date: `{date_g}` · value: `{num_g}` · customer: `{cust_g}`"
    )
    if clear_col.button("🔄 New file"):
        del st.session_state["raw_df"]
        st.rerun()

    st.divider()

    with st.expander("🔍 Data preview (first 5 rows)"):
        st.dataframe(raw_df.head(), use_container_width=True)

    # 3 tabs — each calls its own module
    tab1, tab2, tab3 = st.tabs([
        "📈 Forecast (Prophet)",
        "⚙️ Customer Segments (RFM)",
        "🗺 Geo Analysis",
    ])

    with tab1:
        run_forecast_tab(raw_df, date_g, num_g)

    with tab2:
        run_rfm_tab(raw_df, date_g, cust_g)

    with tab3:
        run_geo_tab(raw_df, num_g)


if __name__ == "__main__":
    main()
