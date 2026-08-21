# =============================================================
#  geo_analysis.py
#  Geospatial sales analysis — vertical bar chart + Folium map
#  FIXED: expanded city dictionary + geopy fallback so ALL cities appear
# =============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

try:
    import folium
    from streamlit_folium import st_folium
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False


# ── Built-in city geocoder (expanded — 150+ locations) ──────
CITY_COORDS = {
    # ── Tamil Nadu (expanded) ────────────────────────────────
    "chennai":          (13.08, 80.27),
    "coimbatore":       (11.02, 76.97),
    "madurai":          (9.93,  78.12),
    "tiruchirappalli":  (10.79, 78.70),
    "trichy":           (10.79, 78.70),   # alias
    "salem":            (11.65, 78.16),
    "tirunelveli":      (8.73,  77.70),
    "vellore":          (12.92, 79.13),
    "erode":            (11.34, 77.73),
    "tiruppur":         (11.10, 77.34),
    "thanjavur":        (10.79, 79.14),
    "tanjore":          (10.79, 79.14),   # alias
    "dindigul":         (10.36, 77.97),
    "thoothukudi":      (8.76,  78.13),
    "tuticorin":        (8.76,  78.13),   # alias
    "kanchipuram":      (12.83, 79.70),
    "cuddalore":        (11.75, 79.77),
    "hosur":            (12.74, 77.83),
    "nagercoil":        (8.18,  77.43),
    "kumbakonam":       (10.96, 79.39),
    "karur":            (10.96, 78.08),
    "namakkal":         (11.22, 78.17),
    "sivakasi":         (9.45,  77.80),
    "pudukkottai":      (10.38, 78.82),
    "krishnagiri":      (12.52, 78.21),
    "dharmapuri":       (12.13, 78.16),
    "ramanathapuram":   (9.37,  78.83),
    "virudhunagar":     (9.58,  77.96),
    "theni":            (10.01, 77.48),
    "nagapattinam":     (10.76, 79.84),
    "villupuram":       (11.94, 79.49),
    "tiruvannamalai":   (12.23, 79.07),
    "perambalur":       (11.23, 78.88),
    "ariyalur":         (11.14, 79.08),
    "kallakurichi":     (11.74, 78.96),
    "ranipet":          (12.92, 79.33),
    "chengalpattu":     (12.69, 79.97),
    "tenkasi":          (8.96,  77.31),
    "tirupathur":       (12.50, 78.57),
    "mayiladuthurai":   (11.10, 79.65),

    # ── Rest of India ────────────────────────────────────────
    "mumbai":           (19.08, 72.88),
    "delhi":            (28.61, 77.21),
    "new delhi":        (28.61, 77.21),
    "bangalore":        (12.97, 77.59),
    "bengaluru":        (12.97, 77.59),   # alias
    "hyderabad":        (17.38, 78.47),
    "kolkata":          (22.57, 88.36),
    "pune":             (18.52, 73.86),
    "ahmedabad":        (23.02, 72.57),
    "jaipur":           (26.91, 75.79),
    "surat":            (21.17, 72.83),
    "lucknow":          (26.85, 80.95),
    "kanpur":           (26.45, 80.33),
    "nagpur":           (21.15, 79.09),
    "indore":           (22.72, 75.86),
    "bhopal":           (23.26, 77.41),
    "visakhapatnam":    (17.69, 83.22),
    "vizag":            (17.69, 83.22),   # alias
    "patna":            (25.59, 85.14),
    "vadodara":         (22.31, 73.19),
    "kochi":            (9.93,  76.26),
    "cochin":           (9.93,  76.26),   # alias
    "agra":             (27.18, 78.01),
    "nashik":           (19.99, 73.79),
    "rajkot":           (22.30, 70.80),
    "chandigarh":       (30.73, 76.78),
    "guwahati":         (26.14, 91.74),
    "mysuru":           (12.30, 76.64),
    "mysore":           (12.30, 76.64),   # alias
    "hubli":            (15.35, 75.13),
    "mangalore":        (12.87, 74.88),
    "mangaluru":        (12.87, 74.88),   # alias
    "thiruvananthapuram":(8.52, 76.94),
    "trivandrum":       (8.52,  76.94),   # alias
    "kozhikode":        (11.25, 75.77),
    "calicut":          (11.25, 75.77),   # alias
    "thrissur":         (10.52, 76.21),
    "warangal":         (17.97, 79.60),
    "vijayawada":       (16.51, 80.62),
    "guntur":           (16.30, 80.44),
    "nellore":          (14.44, 79.99),
    "kurnool":          (15.83, 78.04),
    "rajahmundry":      (17.00, 81.78),
    "tirupati":         (13.63, 79.42),
    "amritsar":         (31.63, 74.87),
    "ludhiana":         (30.90, 75.85),
    "jalandhar":        (31.33, 75.58),
    "jodhpur":          (26.29, 73.02),
    "udaipur":          (24.58, 73.68),
    "kota":             (25.18, 75.85),
    "ajmer":            (26.45, 74.64),
    "bikaner":          (28.02, 73.31),
    "dehradun":         (30.32, 78.03),
    "haridwar":         (29.95, 78.16),
    "meerut":           (28.98, 77.71),
    "ghaziabad":        (28.67, 77.44),
    "noida":            (28.54, 77.34),
    "faridabad":        (28.41, 77.31),
    "gurugram":         (28.46, 77.03),
    "gurgaon":          (28.46, 77.03),   # alias
    "allahabad":        (25.45, 81.84),
    "prayagraj":        (25.45, 81.84),   # alias
    "varanasi":         (25.32, 83.00),
    "benaras":          (25.32, 83.00),   # alias
    "agartala":         (23.83, 91.28),
    "imphal":           (24.82, 93.95),
    "shillong":         (25.58, 91.88),
    "aizawl":           (23.73, 92.72),
    "kohima":           (25.67, 94.11),
    "gangtok":          (27.33, 88.62),
    "bhubaneswar":      (20.30, 85.84),
    "cuttack":          (20.46, 85.88),
    "raipur":           (21.25, 81.63),
    "bilaspur":         (22.09, 82.15),
    "ranchi":           (23.34, 85.31),
    "jamshedpur":       (22.80, 86.20),
    "dhanbad":          (23.80, 86.45),
    "bokaro":           (23.67, 86.15),
    "shimla":           (31.10, 77.17),
    "manali":           (32.24, 77.19),
    "srinagar":         (34.09, 74.80),
    "jammu":            (32.74, 74.87),
    "leh":              (34.16, 77.58),
    "panaji":           (15.49, 73.83),
    "goa":              (15.30, 74.00),
    "india":            (20.59, 78.96),

    # ── UK ───────────────────────────────────────────────────
    "london":           (51.51, -0.13),
    "manchester":       (53.48, -2.24),
    "birmingham":       (52.48, -1.90),
    "leeds":            (53.80, -1.55),
    "glasgow":          (55.86, -4.25),
    "liverpool":        (53.41, -2.98),
    "bristol":          (51.45, -2.59),
    "sheffield":        (53.38, -1.47),
    "edinburgh":        (55.95, -3.19),
    "cardiff":          (51.48, -3.18),
    "leicester":        (52.64, -1.13),
    "nottingham":       (52.95, -1.15),
    "coventry":         (52.41, -1.51),
    "bradford":         (53.80, -1.75),
    "belfast":          (54.60, -5.93),
    "newcastle":        (54.97, -1.61),
    "oxford":           (51.75, -1.26),
    "cambridge":        (52.20,  0.12),
    "brighton":         (50.82, -0.14),
    "plymouth":         (50.37, -4.14),
    "uk":               (54.00, -2.00),
    "united kingdom":   (54.00, -2.00),
    "england":          (52.50, -1.50),
    "scotland":         (56.49, -4.20),
    "wales":            (52.13, -3.78),

    # ── Europe ───────────────────────────────────────────────
    "paris":            (48.85,  2.35),
    "berlin":           (52.52, 13.40),
    "madrid":           (40.42, -3.70),
    "rome":             (41.90, 12.50),
    "amsterdam":        (52.37,  4.90),
    "brussels":         (50.85,  4.35),
    "vienna":           (48.21, 16.37),
    "stockholm":        (59.33, 18.07),
    "oslo":             (59.91, 10.75),
    "copenhagen":       (55.68, 12.57),
    "lisbon":           (38.72, -9.14),
    "athens":           (37.98, 23.73),
    "warsaw":           (52.23, 21.01),
    "prague":           (50.08, 14.44),
    "budapest":         (47.50, 19.04),
    "zurich":           (47.38,  8.54),

    # ── Americas ─────────────────────────────────────────────
    "new york":         (40.71, -74.01),
    "los angeles":      (34.05, -118.24),
    "chicago":          (41.88, -87.63),
    "houston":          (29.76, -95.37),
    "toronto":          (43.65, -79.38),
    "vancouver":        (49.25, -123.12),
    "montreal":         (45.50, -73.57),
    "mexico city":      (19.43, -99.13),
    "sao paulo":        (-23.55, -46.63),
    "buenos aires":     (-34.60, -58.38),
    "usa":              (37.09, -95.71),
    "canada":           (56.13, -106.35),

    # ── Asia / Middle East / Africa / Oceania ────────────────
    "tokyo":            (35.69, 139.69),
    "beijing":          (39.91, 116.39),
    "shanghai":         (31.23, 121.47),
    "singapore":        (1.35,  103.82),
    "sydney":           (-33.87, 151.21),
    "melbourne":        (-37.81, 144.96),
    "dubai":            (25.20,  55.27),
    "hong kong":        (22.32, 114.17),
    "seoul":            (37.57, 126.98),
    "kuala lumpur":     (3.14,  101.69),
    "jakarta":          (-6.21, 106.85),
    "nairobi":          (-1.29,  36.82),
    "lagos":            (6.45,    3.39),
    "cairo":            (30.04,  31.24),
    "china":            (35.86, 104.20),
    "japan":            (36.20, 138.25),
    "australia":        (-25.27, 133.78),
}


# ── Geocoding with dict + geopy fallback ─────────────────────

def try_geocode_cities(df: pd.DataFrame, city_col: str) -> tuple[pd.DataFrame, list]:
    """
    Map city names to lat/lon.
    Step 1 — fast hardcoded dictionary lookup
    Step 2 — geopy Nominatim fallback for any city not found in dict
    Returns (geocoded_df, list_of_failed_city_names)
    """
    df2 = df.copy()
    df2["_city_lower"] = df2[city_col].astype(str).str.strip().str.lower()

    # Step 1 — dict lookup
    df2["lat"] = df2["_city_lower"].map(
        lambda c: CITY_COORDS.get(c, (None, None))[0]
    )
    df2["lon"] = df2["_city_lower"].map(
        lambda c: CITY_COORDS.get(c, (None, None))[1]
    )

    # Step 2 — geopy fallback for misses
    missing_mask = df2["lat"].isna()
    failed_cities = []

    if missing_mask.any() and GEOPY_AVAILABLE:
        geolocator = Nominatim(user_agent="ecommerce_ml_platform", timeout=5)
        for idx, row in df2[missing_mask].iterrows():
            city_name = row["_city_lower"]
            try:
                # Try with ", India" suffix first (since data is India-focused)
                loc = geolocator.geocode(f"{city_name}, India")
                if loc is None:
                    loc = geolocator.geocode(city_name)
                if loc:
                    df2.at[idx, "lat"] = loc.latitude
                    df2.at[idx, "lon"] = loc.longitude
                else:
                    failed_cities.append(row[city_col])
                time.sleep(1)  # Nominatim rate limit: 1 request/second
            except (GeocoderTimedOut, GeocoderServiceError):
                failed_cities.append(row[city_col])

    elif missing_mask.any() and not GEOPY_AVAILABLE:
        # geopy not installed — record all misses
        failed_cities = df2.loc[missing_mask, city_col].tolist()

    df2 = df2.drop(columns=["_city_lower"])
    return df2.dropna(subset=["lat", "lon"]), failed_cities


# ── Main tab ─────────────────────────────────────────────────

def run_geo_tab(raw_df: pd.DataFrame, num_guess: str):
    """Streamlit UI for the Geo Analysis tab."""
    st.header("🗺 Geo Sales Analysis")
    st.markdown("Visualise **sales by city** — vertical bar chart + interactive map.")

    if not GEOPY_AVAILABLE:
        st.warning(
            "⚠️ `geopy` is not installed. Cities not in the built-in dictionary "
            "will be skipped. Run `pip install geopy` to enable auto-geocoding."
        )

    col1, col2 = st.columns(2)
    city_col  = col1.selectbox("City column",  raw_df.columns, key="geo_city")
    value_col = col2.selectbox(
        "Sales / Value column",
        [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])],
        key="geo_val",
    )
    top_n = st.slider("Show top N cities", 5, 20, 10, key="geo_topn")

    if st.button("🚀 Generate Geo Analysis", key="btn_geo"):
        try:
            df_geo = raw_df.copy()
            df_geo[value_col] = pd.to_numeric(df_geo[value_col], errors="coerce").fillna(0)
            df_geo[city_col]  = df_geo[city_col].astype(str).str.strip().str.title()
            df_geo = df_geo[
                df_geo[city_col].notna() &
                (df_geo[city_col] != "") &
                (df_geo[city_col].str.lower() != "nan")
            ]

            if df_geo.empty:
                st.error("No valid city data found.")
                return

            # Aggregate
            agg = (
                df_geo.groupby(city_col)[value_col]
                      .sum().reset_index()
                      .rename(columns={city_col: "City", value_col: "Sales"})
            )
            agg = agg.sort_values("Sales", ascending=False).reset_index(drop=True)

            total_sales  = agg["Sales"].sum()
            top_city     = agg.iloc[0]["City"]
            top_city_val = agg.iloc[0]["Sales"]

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Cities",   f"{len(agg):,}")
            k2.metric("Total Sales",    f"{total_sales:,.0f}")
            k3.metric("Top City",       top_city)
            k4.metric("Top City Sales", f"{top_city_val:,.0f}")

            top_df = agg.head(top_n)

            # ── Vertical bar chart ──────────────────────────
            st.subheader(f"Top {top_n} Cities by Sales")
            bar_colors = ["#00d4aa" if i == 0 else "#6c3bdb" for i in range(len(top_df))]
            fig, ax = plt.subplots(figsize=(max(8, top_n * 0.9), 5))
            fig.patch.set_facecolor("#0f0f1a")
            ax.set_facecolor("#0f0f1a")
            bars = ax.bar(top_df["City"], top_df["Sales"], color=bar_colors, width=0.6)
            for bar, val in zip(bars, top_df["Sales"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:,.0f}",
                    ha="center", va="bottom", color="#aaa", fontsize=8,
                )
            ax.set_title(f"Sales by City — Top {top_n}", color="white", fontsize=13, pad=10)
            ax.set_ylabel("Total Sales", color="#aaa")
            ax.tick_params(colors="#aaa")
            ax.set_xticklabels(top_df["City"], rotation=35, ha="right", color="#ccc", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.set_ylim(0, top_df["Sales"].max() * 1.15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Folium map ──────────────────────────────────
            st.subheader(f"🗺 Interactive Sales Map — Top {top_n} Cities")
            if GEO_AVAILABLE:
                with st.spinner("Geocoding cities…"):
                    geo_agg, failed = try_geocode_cities(top_df, "City")

                # ✅ Show which cities were successfully mapped
                mapped   = len(geo_agg)
                total_c  = len(top_df)
                st.info(
                    f"📍 Mapped **{mapped} of {total_c}** cities on the map."
                    + (
                        f" Could not find: **{', '.join(failed)}**. "
                        f"{'Install `geopy` to auto-geocode missing cities.' if not GEOPY_AVAILABLE else 'These cities were not recognised by the geocoder.'}"
                        if failed else ""
                    )
                )

                if not geo_agg.empty:
                    max_val = geo_agg["Sales"].max() or 1
                    center  = [geo_agg["lat"].mean(), geo_agg["lon"].mean()]
                    m = folium.Map(
                        location=center, zoom_start=6,
                        tiles="CartoDB dark_matter",
                    )
                    cmap = plt.cm.get_cmap("RdYlGn_r")
                    for _, row in geo_agg.iterrows():
                        norm_val = row["Sales"] / max_val
                        hex_col  = mcolors.to_hex(cmap(norm_val))
                        radius   = max(6, norm_val * 25)
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=radius,
                            color=hex_col,
                            fill=True,
                            fill_color=hex_col,
                            fill_opacity=0.8,
                            popup=folium.Popup(
                                f"<b>{row['City']}</b><br>Sales: {row['Sales']:,.0f}",
                                max_width=200,
                            ),
                            tooltip=row["City"],
                        ).add_to(m)
                    st_folium(m, width=900, height=480, returned_objects=[])
                else:
                    st.warning("Could not geocode any cities. Check city column values.")
            else:
                st.info("Install folium + streamlit-folium: `pip install folium streamlit-folium`")

            # Full table
            with st.expander("📋 All cities data"):
                st.dataframe(agg, use_container_width=True)

            csv_b = agg.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Geo Summary CSV", csv_b,
                "geo_analysis.csv", "text/csv",
            )

        except Exception as e:
            st.error(f"Geo error: {e}")
            st.exception(e)
