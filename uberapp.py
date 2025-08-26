import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Vehicle Rides App Data Analysis", layout="wide")
sns.set_style("darkgrid")

# =======================
# Load & normalize schema
# =======================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Renomeia colunas para um padrão sem espaços/maiúsculas
    col_map = {
        "Date": "DATE",
        "Time": "TIME",
        "Booking ID": "BOOKING_ID",
        "Booking Status": "BOOKING_STATUS",
        "Customer ID": "CUSTOMER_ID",
        "Vehicle Type": "VEHICLE_TYPE",
        "Pickup Location": "PICKUP_LOCATION",
        "Drop Location": "DROP_LOCATION",
        "Avg VTAT": "AVG_VTAT",
        "Avg CTAT": "AVG_CTAT",
        "Cancelled Rides by Customer": "CANCELLED_RIDES_BY_CUSTOMER",
        "Reason for cancelling by Customer": "REASON_FOR_CANCELLING_BY_CUSTOMER",
        "Cancelled Rides by Driver": "CANCELLED_RIDES_BY_DRIVER",
        "Driver Cancellation Reason": "DRIVER_CANCELLATION_REASON",
        "Incomplete Rides": "INCOMPLETE_RIDES",
        "Incomplete Rides Reason": "INCOMPLETE_RIDES_REASON",
        "Booking Value": "BOOKING_VALUE",
        "Ride Distance": "RIDE_DISTANCE",
        "Driver Ratings": "DRIVER_RATINGS",
        "Customer Rating": "CUSTOMER_RATING",
        "Payment Method": "PAYMENT_METHOD",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Tipagem
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    if "TIME" in df.columns:
        dt_time = pd.to_datetime(df["TIME"], format="%H:%M:%S", errors="coerce")
        df["HOUR"] = dt_time.dt.hour.astype("Int64")  # preserva NaN

    # Converte possíveis numéricos
    numeric_candidates = [
        "AVG_VTAT", "AVG_CTAT",
        "CANCELLED_RIDES_BY_CUSTOMER", "CANCELLED_RIDES_BY_DRIVER", "INCOMPLETE_RIDES",
        "BOOKING_VALUE", "RIDE_DISTANCE", "DRIVER_RATINGS", "CUSTOMER_RATING"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # PRICE_PER_KM seguro (sem divisão por zero)
    if {"BOOKING_VALUE", "RIDE_DISTANCE"}.issubset(df.columns):
        denom = df["RIDE_DISTANCE"].replace({0: np.nan})
        df["PRICE_PER_KM"] = df["BOOKING_VALUE"] / denom
        df["PRICE_PER_KM"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Mês (para agregações)
    if "DATE" in df.columns:
        df["MONTH"] = df["DATE"].dt.to_period("M")

    return df


# ===========
# Load data
# ===========
df = load_data("ncr_ride_bookings.csv")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.image("my_picture.jpg", width=150) 
st.sidebar.markdown(
"""
**Hey! I'm Aaron.**  
I clean the mess, find the signal, and build analytics that actually get used.
"""
)
st.sidebar.subheader("Portfolio Links")
st.sidebar.markdown("""
[![GitHub](https://img.shields.io/badge/GitHub-AaronProgramas-black?logo=github)](https://github.com/AaronProgramas)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-aaron--albrecht-black?logo=linkedin)](https://www.linkedin.com/in/aaron-albrecht-32692b259/)  
[![Kaggle](https://img.shields.io/badge/Kaggle-AaronAlbrecht-black?logo=kaggle)](https://www.kaggle.com/aaronalbrecht)
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["EDA", "Data Visualization"])

# ==============
# TAB 1 - EDA
# ==============
with tab1:
    st.title("Uber Rides Analytics Hub with Streamlit")
    date_min = pd.to_datetime(df.get("DATE"), errors="coerce").min()
    date_max = pd.to_datetime(df.get("DATE"), errors="coerce").max()
    n_rows = len(df)
    n_cities = df["PICKUP_LOCATION"].nunique() if "PICKUP_LOCATION" in df.columns else None
    n_vehicle = df["VEHICLE_TYPE"].nunique() if "VEHICLE_TYPE" in df.columns else None
    
    facts = [
        f"{n_rows:,} rides",
        f"{n_vehicle} vehicle types" if n_vehicle else None,
        f"{n_cities} pickup locations" if n_cities else None,
        f"{date_min:%b %Y} — {date_max:%b %Y}" if pd.notna(date_min) and pd.notna(date_max) else None,
    ]
    facts = " · ".join([f for f in facts if f])
    
    st.caption(f"Interactive Streamlit Panel with a **public Kaggle dataset**. {facts}")
    # KPIs
    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", f"{n_cols}")
    c3.metric("Total Missing", f"{total_missing:,}")
    c4.metric("Duplicate Rows", f"{dup_rows:,}")

    # Head
    st.subheader("Head of the dataframe")
    st.dataframe(df.head())

    # =======================
    # INFO  |  HEATMAP (lado a lado, compacto)
    # =======================
    st.subheader("EDA")
    col_info, col_heat = st.columns(2)

    with col_info:
        st.markdown("**DataFrame.info()**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue())

    with col_heat:
        st.markdown("**Correlation Heatmap (compact)**")

        # base numérica sem constantes
        num_base = df.select_dtypes(include=[np.number])
        num_base = num_base.loc[:, num_base.std(numeric_only=True) > 0]

        # controles
        cA, cB, cC = st.columns(3)
        default_target = "BOOKING_VALUE" if "BOOKING_VALUE" in num_base.columns else num_base.columns[0]
        with cA:
            target_col = st.selectbox(
                "Target", options=list(num_base.columns),
                index=list(num_base.columns).index(default_target)
            )
        with cB:
            max_cols = st.slider("Max cols", 4, min(20, len(num_base.columns)),
                                 min(8, len(num_base.columns)))
        with cC:
            min_abs_corr = st.slider("Min |corr|", 0.0, 1.0, 0.10, 0.01)

        corr_full = num_base.corr()
        ranked = (corr_full[target_col].drop(labels=[target_col])
                  .abs().sort_values(ascending=False))

        selected = [target_col] + ranked[ranked >= min_abs_corr].index.tolist()
        if len(selected) < 2:
            selected = [target_col] + list(ranked.index[:max_cols - 1])
        else:
            selected = selected[:max_cols]

        corr = corr_full.loc[selected, selected]
        n = len(selected)

        # figura compacta (não escalada)
        fig, ax = plt.subplots(figsize=(5.6, 3.9), dpi=150)
        sns.heatmap(
            corr, cmap="viridis", annot=(n <= 8), fmt=".2f",
            ax=ax, center=0, cbar_kws={"shrink": 0.7, "aspect": 20},
            annot_kws={"size": 8}
        )
        ax.set_title(f"Correlation (top {n} cols vs {target_col})", fontsize=11, pad=6)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # =======================
    # Describe (ABAIXO das colunas)
    # =======================
    st.subheader("Describe (all)")
    st.dataframe(df.describe(include="all").transpose())

    # =======================
    # Extra EDA (duas colunas, abaixo)
    # =======================
    st.subheader("Extra EDA (side-by-side)")
    colA, colB = st.columns(2)

    with colA:
        # Missing values %
        st.markdown("**Missing Values (%)**")
        miss = (df.isna().mean() * 100).sort_values(ascending=False).rename("missing_%").to_frame()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=miss.index[:15], y=miss["missing_%"][:15], palette="viridis", ax=ax)
        ax.set_ylabel("Missing (%)"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
        st.pyplot(fig)

        # Correlação com alvo (top 10)
        num_base2 = df.select_dtypes(include=[np.number])
        num_base2 = num_base2.loc[:, num_base2.std(numeric_only=True) > 0]
        if not num_base2.empty:
            tgt = "BOOKING_VALUE" if "BOOKING_VALUE" in num_base2.columns else num_base2.columns[0]
            st.markdown(f"**Correlation with `{tgt}` (top 10)**")
            corr_top = num_base2.corr()[tgt].drop(tgt).sort_values(
                key=lambda s: s.abs(), ascending=False
            ).head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=corr_top.index, y=corr_top.values, palette="viridis", ax=ax)
            ax.set_ylabel("Pearson r"); ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

    with colB:
        # Cardinalidade das categóricas
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.markdown("**Categorical Cardinality (top 15)**")
            card = df[cat_cols].nunique().sort_values(ascending=False).rename("unique_values").to_frame()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=card.index[:15], y=card["unique_values"][:15], palette="viridis", ax=ax)
            ax.set_ylabel("Unique values"); ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)

        # Taxa de outliers (IQR)
        st.markdown("**Outlier Rate by Numeric Column (IQR)**")
        def outlier_rate(s: pd.Series) -> float:
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return 0.0
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                return 0.0
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return ((s < lower) | (s > upper)).mean()

        num_base3 = df.select_dtypes(include=[np.number])
        num_base3 = num_base3.loc[:, num_base3.std(numeric_only=True) > 0]
        out_rates = num_base3.apply(outlier_rate).sort_values(ascending=False).rename("outlier_rate").to_frame()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=out_rates.index[:12], y=out_rates["outlier_rate"][:12], palette="viridis", ax=ax)
        ax.set_ylabel("Outlier rate"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
        st.pyplot(fig)

    # Distribuições rápidas (segunda fileira)
    dist_cols = [c for c in ["DRIVER_RATINGS", "CUSTOMER_RATING"] if c in df.columns]
    if dist_cols:
        st.subheader("Ratings Distribution")
        d1, d2 = st.columns(2)
        if len(dist_cols) > 0:
            with d1:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[dist_cols[0]].dropna(), bins=30, kde=True,
                             color=sns.color_palette("viridis", as_cmap=True)(0.6), ax=ax)
                ax.set_title(dist_cols[0]); st.pyplot(fig)
        if len(dist_cols) > 1:
            with d2:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df[dist_cols[1]].dropna(), bins=30, kde=True,
                             color=sns.color_palette("viridis", as_cmap=True)(0.6), ax=ax)
                ax.set_title(dist_cols[1]); st.pyplot(fig)

# =========================
# TAB 2 - Visualizations
# =========================
with tab2:
    st.title("Visualization")

    # Garantias (caso o CSV mude)
    if "DATE" in df.columns and "MONTH" not in df.columns:
        df["MONTH"] = pd.to_datetime(df["DATE"], errors="coerce").dt.to_period("M")
    if "TIME" in df.columns and "HOUR" not in df.columns:
        df["HOUR"] = pd.to_datetime(df["TIME"], format="%H:%M:%S", errors="coerce").dt.hour
    if {"BOOKING_VALUE", "RIDE_DISTANCE"}.issubset(df.columns) and "PRICE_PER_KM" not in df.columns:
        df["PRICE_PER_KM"] = df["BOOKING_VALUE"] / df["RIDE_DISTANCE"].replace({0: np.nan})

    col1, col2 = st.columns(2)

    with col1:
        # Monthly Rides by Vehicle Type
        if {"MONTH", "VEHICLE_TYPE", "BOOKING_ID"}.issubset(df.columns):
            st.subheader("Monthly Rides by Vehicle Type")
            rides_by_vehicle = (
                df.groupby(["MONTH", "VEHICLE_TYPE"])["BOOKING_ID"]
                  .count()
                  .reset_index()
            )
            rides_by_vehicle["MONTH"] = rides_by_vehicle["MONTH"].astype(str)
            fig, ax = plt.subplots(figsize=(14, 7))
            sns.lineplot(
                data=rides_by_vehicle,
                x="MONTH", y="BOOKING_ID", hue="VEHICLE_TYPE",
                palette="viridis", marker="o", linewidth=2.5, ax=ax
            )
            ax.set_xlabel("Month"); ax.set_ylabel("Number of Rides")
            ax.tick_params(axis="x", rotation=90)
            ax.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Scatter: Booking Value vs Ride Distance
        if {"RIDE_DISTANCE", "BOOKING_VALUE", "VEHICLE_TYPE"}.issubset(df.columns):
            st.subheader("Booking Value vs Ride Distance")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(
                data=df, x="RIDE_DISTANCE", y="BOOKING_VALUE",
                hue="VEHICLE_TYPE", palette="viridis", alpha=0.6, ax=ax
            )
            ax.set_title("Booking Value vs Ride Distance", fontsize=16, weight="bold")
            ax.set_xlabel("Ride Distance (km)")
            ax.set_ylabel("Booking Value ($)")
            ax.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Cancellation Breakdown
        if {"CANCELLED_RIDES_BY_CUSTOMER", "CANCELLED_RIDES_BY_DRIVER"}.issubset(df.columns):
            st.subheader("Cancellation Breakdown")
            cancel_data = {
                "Cancelled by Customer": float(df["CANCELLED_RIDES_BY_CUSTOMER"].sum()),
                "Cancelled by Driver": float(df["CANCELLED_RIDES_BY_DRIVER"].sum())
            }
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = sns.color_palette("viridis", n_colors=2)
            ax.pie(cancel_data.values(), labels=cancel_data.keys(),
                   autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title("Cancellation Breakdown", fontsize=16, weight="bold")
            st.pyplot(fig)

        # Ride Demand by Hour
        if "HOUR" in df.columns:
            st.subheader("Ride Demand Distribution by Hour of the Day")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(
                data=df.dropna(subset=["HOUR"]),
                x="HOUR", bins=24, kde=True, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.6)
            )
            ax.set_title("Ride Demand Distribution by Hour of the Day", fontsize=16, weight="bold")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Number of Rides (density overlay)")
            ax.set_xticks(range(0, 24))
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    with col2:
        # Average Price per Km by Hour
        if {"HOUR", "PRICE_PER_KM"}.issubset(df.columns):
            st.subheader("Average Price per Km Throughout the Day")
            avg_price_hour = (
                df.dropna(subset=["HOUR", "PRICE_PER_KM"])
                  .groupby("HOUR")["PRICE_PER_KM"]
                  .mean()
                  .reset_index()
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=avg_price_hour, x="HOUR", y="PRICE_PER_KM",
                marker="o", linewidth=2.5, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.7)
            )
            ax.set_title("Average Price per Km Throughout the Day", fontsize=16, weight="bold")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Average Price per Km ($/km)")
            ax.set_xticks(range(0, 24))
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Distribution of Booking Value
        if "BOOKING_VALUE" in df.columns:
            st.subheader("Distribution of Booking Value")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(
                df["BOOKING_VALUE"].dropna(), bins=50, kde=True, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.7)
            )
            ax.set_title("Distribution of Booking Value", fontsize=16, weight="bold")
            ax.set_xlabel("Booking Value ($)"); ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # Total Revenue per Month
        if {"MONTH", "BOOKING_VALUE"}.issubset(df.columns):
            st.subheader("Total Revenue per Month")
            revenue_month = df.groupby("MONTH")["BOOKING_VALUE"].sum().reset_index()
            revenue_month["MONTH"] = revenue_month["MONTH"].astype(str)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=revenue_month, x="MONTH", y="BOOKING_VALUE",
                marker="o", linewidth=2.5, ax=ax,
                color=sns.color_palette("viridis", as_cmap=True)(0.6)
            )
            ax.set_title("Total Revenue per Month", fontsize=16, weight="bold")
            ax.set_xlabel("Month"); ax.set_ylabel("Total Revenue ($)")
            ax.tick_params(axis="x", rotation=90)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        # Violin: Price per Km by Vehicle Type
        if {"PRICE_PER_KM", "VEHICLE_TYPE"}.issubset(df.columns):
            st.subheader("Distribution of Price per Km by Vehicle Type")
            cap = df["PRICE_PER_KM"].quantile(0.80)
            df_cap = df.copy()
            df_cap.loc[df_cap["PRICE_PER_KM"] > cap, "PRICE_PER_KM"] = cap
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(
                data=df_cap, x="VEHICLE_TYPE", y="PRICE_PER_KM",
                palette="viridis", cut=0, ax=ax
            )
            ax.set_title("Distribution of Price per Km by Vehicle Type", fontsize=16, weight="bold")
            ax.set_xlabel("Vehicle Type"); ax.set_ylabel("Price per Km ($/km)")
            ax.tick_params(axis="x", rotation=90)
            st.pyplot(fig)


