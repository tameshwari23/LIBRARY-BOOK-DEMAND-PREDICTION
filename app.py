
import os
import calendar
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Library Demand Predictor",
    page_icon="📚",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "library_demand_model.pkl")
DATA_PATH = os.path.join(
    BASE_DIR, "library_checkouts_2025_2026_clean.csv"
)

PURPLE = "#8b5cf6"
CYAN = "#22d3ee"
PINK = "#f472b6"
GREEN = "#34d399"
AMBER = "#fbbf24"
COLORS = [PURPLE, CYAN, PINK, GREEN, AMBER, "#60a5fa"]

MONTHS = list(calendar.month_name)[1:]
MONTH_ABBR = list(calendar.month_abbr)[1:]

# ===================== CUSTOM DESIGN =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
      radial-gradient(900px 500px at 0% 0%,
      rgba(139,92,246,.20), transparent 60%),
      radial-gradient(900px 500px at 100% 0%,
      rgba(34,211,238,.13), transparent 60%),
      #0b1020;
    color: #e5e7eb;
}

#MainMenu, footer {visibility: hidden;}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#11162b,#0d1224);
    border-right: 1px solid rgba(255,255,255,.08);
}

.hero {
    padding: 30px;
    border-radius: 24px;
    margin-bottom: 22px;
    background: linear-gradient(
        115deg,
        rgba(139,92,246,.34),
        rgba(34,211,238,.16),
        rgba(244,114,182,.20)
    );
    border: 1px solid rgba(255,255,255,.13);
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
}

.hero h1 {
    color: white;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #cbd5e1;
    margin-top: 10px;
}

.pill {
    display: inline-block;
    padding: 5px 12px;
    margin: 12px 7px 0 0;
    border-radius: 999px;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.13);
    color: #e2e8f0;
    font-size: .8rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.09);
    padding: 16px;
    border-radius: 17px;
}

div[data-testid="stMetricLabel"] {
    color: #a5b4fc !important;
}

div[data-testid="stMetricValue"] {
    color: white !important;
    font-weight: 750;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 750;
    color: white;
    margin: 8px 0 5px;
}

.section-sub {
    color: #94a3b8;
    font-size: .88rem;
    margin-bottom: 12px;
}

.result {
    padding: 25px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid rgba(255,255,255,.15);
    min-height: 190px;
}

.high {
    background: linear-gradient(
        135deg, rgba(52,211,153,.25),
        rgba(34,211,238,.13)
    );
}

.low {
    background: linear-gradient(
        135deg, rgba(251,113,133,.25),
        rgba(251,191,36,.12)
    );
}

.result-icon {font-size: 2.8rem;}
.result-label {font-size: 1.8rem; font-weight: 800; color: white;}
.result-prob {font-size: 2.7rem; font-weight: 800; color: #ddd6fe;}
.result-period {color: #cbd5e1;}

.stButton > button {
    border: 0;
    border-radius: 12px;
    color: white;
    font-weight: 700;
    padding: .7rem 1rem;
    background: linear-gradient(90deg,#8b5cf6,#0891b2);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,.035);
    padding: 6px;
    border-radius: 14px;
}

.book-row {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 10px 13px;
    margin: 7px 0;
    border-radius: 13px;
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.07);
}

.book-rank {
    min-width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#8b5cf6,#0891b2);
    color: white;
    font-weight: 800;
}

.book-title {color: #f8fafc; flex: 1; font-size: .88rem;}
.book-author {color: #94a3b8; font-size: .75rem; margin-top: 3px;}
.book-count {color: #67e8f9; font-weight: 750;}

</style>
""", unsafe_allow_html=True)


# ===================== LOAD MODEL AND DATA =====================
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: library_demand_model.pkl")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found: library_checkouts_2025_2026_clean.csv")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required = {
        "checkoutyear", "checkoutmonth", "checkouts", "title"
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset missing columns: {', '.join(sorted(missing))}"
        )

    for col in ["checkoutyear", "checkoutmonth", "checkouts"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=["checkoutyear", "checkoutmonth", "checkouts"]
    ).copy()

    data["checkoutyear"] = data["checkoutyear"].astype(int)
    data["checkoutmonth"] = data["checkoutmonth"].astype(int)
    data["checkouts"] = data["checkouts"].clip(lower=0)

    for col in [
        "usageclass", "materialtype", "title",
        "creator", "subjects"
    ]:
        if col not in data.columns:
            data[col] = "Unknown"

        data[col] = data[col].fillna("Unknown").astype(str).str.strip()

    data["materialtype"] = data["materialtype"].str.upper()

    data["title_clean"] = (
        data["title"]
        .str.split(" / ").str[0]
        .str.strip()
    )

    data["period"] = pd.to_datetime(
        dict(
            year=data["checkoutyear"],
            month=data["checkoutmonth"],
            day=1
        ),
        errors="coerce"
    )

    # Derive the target from this dataset, not from a missing CSV column.
    median_value = data["checkouts"].median()
    data["demand"] = (
        data["checkouts"] > median_value
    ).astype(int)

    return data


try:
    model = load_model()
    df_all = load_data()
except Exception as error:
    st.error(f"Could not load files: {error}")
    st.stop()


# ===================== HELPER FUNCTIONS =====================
def style_chart(fig, height=370, legend=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        height=height,
        margin=dict(l=10, r=10, t=55, b=15),
        title_font=dict(size=16, color="white"),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.2, x=0)
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


def predict_period(year, month):
    inputs = pd.DataFrame({
        "checkoutyear": [int(year)],
        "checkoutmonth": [int(month)]
    })

    probability = float(model.predict_proba(inputs)[0][1])
    predicted_class = int(probability >= 0.5)

    return probability, predicted_class


def show_books(table, limit=5):
    for rank, (_, row) in enumerate(table.head(limit).iterrows(), 1):
        title = str(row["title_clean"])
        author = str(row["creator"])
        material = str(row["materialtype"])
        count = int(row["checkouts"])

        if len(title) > 75:
            title = title[:75] + "..."

        st.markdown(
            f'<div class="book-row">'
            f'<div class="book-rank">{rank}</div>'
            f'<div class="book-title">{title}'
            f'<div class="book-author">{author} • {material}</div>'
            f'</div><div class="book-count">{count:,}</div></div>',
            unsafe_allow_html=True
        )


# ===================== SIDEBAR FILTERS =====================
with st.sidebar:
    st.markdown("## 🎛️ Dashboard filters")

    usage_options = sorted(df_all["usageclass"].unique().tolist())
    material_options = sorted(df_all["materialtype"].unique().tolist())
    year_options = sorted(df_all["checkoutyear"].unique().tolist())

    selected_usage = st.multiselect(
        "Usage class", usage_options, default=usage_options
    )
    selected_materials = st.multiselect(
        "Material type", material_options, default=material_options
    )
    selected_years = st.multiselect(
        "Year", year_options, default=year_options
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("Source: Seattle Public Library — Checkouts by Title.")
    st.caption(
        "The model uses year and month only. Predictions are an "
        "educational demonstration, not reliable future forecasts."
    )

df = df_all[
    df_all["usageclass"].isin(selected_usage)
    & df_all["materialtype"].isin(selected_materials)
    & df_all["checkoutyear"].isin(selected_years)
].copy()

if df.empty:
    st.warning("No records match these filters. Change the sidebar filters.")
    st.stop()


# ===================== HERO AND KPI CARDS =====================
st.markdown(
    f"""
    <div class="hero">
      <h1>📚 Library Book Demand Predictor</h1>
      <p>Predict demand categories and explore checkout patterns
      across the Seattle Public Library dataset.</p>
      <span class="pill">🤖 Random Forest</span>
      <span class="pill">📊 {len(df_all):,} records</span>
      <span class="pill">📅 {df_all['checkoutyear'].min()}–{df_all['checkoutyear'].max()}</span>
      <span class="pill">📖 {df_all['title_clean'].nunique():,} title labels</span>
    </div>
    """,
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 Total checkouts", f"{int(df['checkouts'].sum()):,}")
k2.metric("🧾 Title records", f"{len(df):,}")
k3.metric("📚 Unique title labels", f"{df['title_clean'].nunique():,}")
k4.metric("⚖️ Median checkouts", f"{df_all['checkouts'].median():.0f}")


# ===================== TABS =====================
tab_predict, tab_trends, tab_explore, tab_model = st.tabs([
    "🔮 Predict Demand",
    "📈 Monthly Trends",
    "🧭 Explore Books",
    "🤖 Model Insights"
])


# ===================== TAB 1: PREDICTION =====================
with tab_predict:
    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">🎯 Select a period</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-sub">Choose the year and month.</div>',
            unsafe_allow_html=True
        )

        available_years = sorted(
            set(df_all["checkoutyear"].tolist())
        )

        year = st.selectbox(
            "Checkout year",
            available_years,
            index=available_years.index(2026)
            if 2026 in available_years else 0
        )

        month_name = st.selectbox("Checkout month", MONTHS)
        month = MONTHS.index(month_name) + 1

        clicked = st.button(
            "✨ Predict Demand", use_container_width=True
        )

        st.caption(
            "This model uses year and month only; it does not accept "
            "a specific book title as an input."
        )

    probability, prediction = predict_period(year, month)
    is_high = prediction == 1
    result_probability = probability if is_high else 1 - probability

    label = "HIGH DEMAND" if is_high else "LOW DEMAND"
    emoji = "🔥" if is_high else "🧊"
    result_class = "high" if is_high else "low"

    period_exists = (
        (df_all["checkoutyear"] == year)
        & (df_all["checkoutmonth"] == month)
    ).any()

    with right:
        st.markdown(
            '<div class="section-title">✨ Prediction result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result {result_class}">
              <div class="result-icon">{emoji}</div>
              <div class="result-label">{label}</div>
              <div class="result-period">{month_name} {year}</div>
              <div class="result-prob">{result_probability * 100:.1f}%</div>
              <div style="color:#cbd5e1;font-size:.82rem;">
                Model probability for the predicted class
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not period_exists:
            st.warning(
                "This month is not represented in the dataset. "
                "The output is an extrapolation."
            )

        if clicked:
            st.toast(f"{label.title()} predicted for {month_name} {year}")

    c1, c2 = st.columns(2)

    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Probability of High Demand"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PURPLE},
                "steps": [
                    {"range": [0, 33], "color": "#51202c"},
                    {"range": [33, 66], "color": "#59451a"},
                    {"range": [66, 100], "color": "#164e3c"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "value": 50
                }
            }
        ))
        st.plotly_chart(
            style_chart(gauge, 320, False),
            use_container_width=True
        )

    with c2:
        probs = [
            predict_period(year, m)[0] * 100
            for m in range(1, 13)
        ]

        fig = go.Figure(go.Bar(
            x=MONTH_ABBR,
            y=probs,
            marker_color=[
                PURPLE if m == month else CYAN
                for m in range(1, 13)
            ],
            text=[f"{p:.0f}%" for p in probs],
            textposition="outside"
        ))
        fig.add_hline(
            y=50, line_dash="dash", line_color="white", opacity=.6
        )
        fig.update_layout(title=f"Model output across {year}")
        fig.update_yaxes(title="Probability (%)", range=[0, 115])

        st.plotly_chart(
            style_chart(fig, 320, False),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">📖 Actual dataset context</div>',
        unsafe_allow_html=True
    )

    context = df[
        (df["checkoutyear"] == year)
        & (df["checkoutmonth"] == month)
    ]

    if context.empty:
        st.info(
            "No actual records exist for this period under the current filters."
        )
    else:
        a, b, c, d = st.columns(4)
        a.metric("Actual checkouts", f"{int(context['checkouts'].sum()):,}")
        b.metric("Title records", f"{len(context):,}")
        c.metric("Average per record", f"{context['checkouts'].mean():.2f}")
        d.metric("High-class share", f"{context['demand'].mean() * 100:.1f}%")

        top = (
            context.groupby(
                ["title_clean", "creator", "materialtype"],
                as_index=False
            )["checkouts"]
            .sum()
            .sort_values("checkouts", ascending=False)
        )

        p1, p2 = st.columns(2)

        with p1:
            st.markdown("**🏆 Most borrowed titles in this period**")
            show_books(top, 5)

        with p2:
            material_data = context.groupby(
                "materialtype", as_index=False
            )["checkouts"].sum()

            fig = px.pie(
                material_data,
                names="materialtype",
                values="checkouts",
                hole=.55,
                color_discrete_sequence=COLORS,
                title="Checkout share by material"
            )
            st.plotly_chart(
                style_chart(fig, 330, False),
                use_container_width=True
            )


# ===================== TAB 2: MONTHLY TRENDS =====================
with tab_trends:
    st.markdown(
        '<div class="section-title">📈 Month-wise checkout trends</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">Explore checkout activity using the sidebar filters.</div>',
        unsafe_allow_html=True
    )

    monthly = (
        df.groupby("period", as_index=False)
        .agg(
            total_checkouts=("checkouts", "sum"),
            records=("title", "size"),
            average_checkouts=("checkouts", "mean")
        )
        .sort_values("period")
    )

    monthly["month_label"] = monthly["period"].dt.strftime("%b %Y")

    fig = px.line(
        monthly,
        x="month_label",
        y="total_checkouts",
        markers=True,
        title="Total checkouts by month"
    )
    fig.update_traces(
        line=dict(color=CYAN, width=3),
        marker=dict(color=PURPLE, size=8),
        fill="tozeroy",
        fillcolor="rgba(34,211,238,.10)"
    )
    fig.update_yaxes(title="Total checkouts")
    st.plotly_chart(
        style_chart(fig, 400, False),
        use_container_width=True
    )

    p1, p2 = st.columns(2)

    with p1:
        comparison = df.groupby(
            ["checkoutyear", "checkoutmonth"], as_index=False
        )["checkouts"].sum()

        comparison["year"] = comparison["checkoutyear"].astype(str)
        comparison["month"] = comparison["checkoutmonth"].map(
            lambda m: calendar.month_abbr[int(m)]
        )

        fig = px.bar(
            comparison,
            x="month",
            y="checkouts",
            color="year",
            barmode="group",
            category_orders={"month": MONTH_ABBR},
            color_discrete_sequence=[PURPLE, CYAN],
            title="2025 vs 2026 — checkouts by month"
        )
        fig.update_yaxes(title="Total checkouts")
        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True
        )

    with p2:
        usage = df.groupby(
            ["period", "usageclass"], as_index=False
        )["checkouts"].sum()

        usage["month_label"] = usage["period"].dt.strftime("%b %Y")

        fig = px.bar(
            usage,
            x="month_label",
            y="checkouts",
            color="usageclass",
            barmode="group",
            color_discrete_sequence=[PINK, GREEN],
            title="Digital vs Physical checkouts"
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Checkouts")
        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True
        )

    p3, p4 = st.columns(2)

    with p3:
        fig = px.line(
            monthly,
            x="month_label",
            y="average_checkouts",
            markers=True,
            title="Average checkouts per record"
        )
        fig.update_traces(line=dict(color=AMBER, width=3))
        fig.update_yaxes(title="Average checkouts")
        st.plotly_chart(
            style_chart(fig, 360, False),
            use_container_width=True
        )

    with p4:
        material_month = df.groupby(
            ["period", "materialtype"], as_index=False
        )["checkouts"].sum()
        material_month["month_label"] = (
            material_month["period"].dt.strftime("%b %Y")
        )

        fig = px.area(
            material_month,
            x="month_label",
            y="checkouts",
            color="materialtype",
            color_discrete_sequence=COLORS,
            title="Monthly checkouts by material type"
        )
        st.plotly_chart(
            style_chart(fig, 360),
            use_container_width=True
        )

    heatmap_data = df.pivot_table(
        index="materialtype",
        columns="period",
        values="checkouts",
        aggfunc="sum",
        fill_value=0
    )

    if not heatmap_data.empty:
        fig = go.Figure(go.Heatmap(
            z=np.log1p(heatmap_data.values),
            x=[p.strftime("%b %Y") for p in heatmap_data.columns],
            y=heatmap_data.index.tolist(),
            customdata=heatmap_data.values,
            colorscale="Viridis",
            colorbar={"title": "Log scale"},
            hovertemplate=(
                "%{y}<br>%{x}<br>Checkouts: %{customdata:,.0f}<extra></extra>"
            )
        ))
        fig.update_layout(
            title="Checkout heatmap — material type × month"
        )
        st.plotly_chart(
            style_chart(fig, 400, False),
            use_container_width=True
        )

    with st.expander("📋 View monthly summary table"):
        st.dataframe(
            monthly.rename(columns={
                "month_label": "Month",
                "total_checkouts": "Total checkouts",
                "records": "Title records",
                "average_checkouts": "Average checkouts"
            }),
            use_container_width=True,
            hide_index=True
        )


# ===================== TAB 3: EXPLORE BOOKS =====================
with tab_explore:
    st.markdown(
        '<div class="section-title">🧭 Explore the collection</div>',
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:
        material_totals = df.groupby(
            "materialtype", as_index=False
        )["checkouts"].sum().sort_values("checkouts", ascending=False)

        fig = px.pie(
            material_totals,
            names="materialtype",
            values="checkouts",
            hole=.55,
            color_discrete_sequence=COLORS,
            title="Checkouts by material type"
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(
            style_chart(fig, 390, False),
            use_container_width=True
        )

    with p2:
        if "subjects" in df.columns:
            subject_data = df[
                df["subjects"].notna()
                & (df["subjects"] != "Unknown")
            ][["subjects", "checkouts"]].copy()

            if not subject_data.empty:
                subject_data["subject"] = subject_data["subjects"].str.split(", ")
                subject_data = subject_data.explode("subject")

                top_subjects = (
                    subject_data.groupby("subject", as_index=False)["checkouts"]
                    .sum()
                    .nlargest(15, "checkouts")
                )

                fig = px.bar(
                    top_subjects.sort_values("checkouts"),
                    x="checkouts",
                    y="subject",
                    orientation="h",
                    color="checkouts",
                    color_continuous_scale=["#8b5cf6", "#f472b6"],
                    title="Popular subjects / genres"
                )
                fig.update_layout(
                    coloraxis_showscale=False,
                    yaxis_title=""
                )
                st.plotly_chart(
                    style_chart(fig, 390, False),
                    use_container_width=True
                )

    p3, p4 = st.columns(2)

    with p3:
        authors = (
            df[df["creator"] != "Unknown"]
            .groupby("creator", as_index=False)["checkouts"]
            .sum()
            .nlargest(12, "checkouts")
        )

        if not authors.empty:
            fig = px.bar(
                authors.sort_values("checkouts"),
                x="checkouts",
                y="creator",
                orientation="h",
                color="checkouts",
                color_continuous_scale=["#22d3ee", "#34d399"],
                title="Popular authors / creators"
            )
            fig.update_layout(
                coloraxis_showscale=False,
                yaxis_title=""
            )
            st.plotly_chart(
                style_chart(fig, 410, False),
                use_container_width=True
            )

    with p4:
        cutoff = df["checkouts"].quantile(.97)
        distribution = df[df["checkouts"] <= cutoff]

        fig = px.histogram(
            distribution,
            x="checkouts",
            color="usageclass",
            nbins=40,
            barmode="overlay",
            opacity=.75,
            color_discrete_sequence=[PINK, GREEN],
            title="Checkout distribution (up to 97th percentile)"
        )
        fig.update_xaxes(title="Checkouts per record")
        fig.update_yaxes(title="Number of records")
        st.plotly_chart(
            style_chart(fig, 410),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">🏆 Most borrowed titles</div>',
        unsafe_allow_html=True
    )

    periods = ["All selected months"] + sorted(
        df["period"].dt.strftime("%b %Y").unique().tolist()
    )
    selected_period = st.selectbox("Choose a month", periods)

    title_data = (
        df if selected_period == "All selected months"
        else df[df["period"].dt.strftime("%b %Y") == selected_period]
    )

    top_titles = (
        title_data.groupby(
            ["title_clean", "creator", "materialtype"],
            as_index=False
        )["checkouts"]
        .sum()
        .sort_values("checkouts", ascending=False)
    )

    a, b = st.columns(2)

    with a:
        show_books(top_titles, 10)

    with b:
        chart_titles = top_titles.head(10).copy()
        if not chart_titles.empty:
            chart_titles["short_title"] = (
                chart_titles["title_clean"].str.slice(0, 32)
            )
            fig = px.bar(
                chart_titles.sort_values("checkouts"),
                x="checkouts",
                y="short_title",
                orientation="h",
                color="materialtype",
                color_discrete_sequence=COLORS,
                title="Top 10 titles"
            )
            fig.update_layout(yaxis_title="")
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True
            )

    st.markdown(
        '<div class="section-title">🔎 Search for a book or author</div>',
        unsafe_allow_html=True
    )

    query = st.text_input(
        "Search title or author",
        placeholder="Enter a book title or creator name..."
    )

    if query:
        matches = df[
            df["title_clean"].str.contains(
                query, case=False, na=False, regex=False
            )
            | df["creator"].str.contains(
                query, case=False, na=False, regex=False
            )
        ]

        if matches.empty:
            st.info("No matching books or authors found.")
        else:
            results = (
                matches.groupby(
                    ["title_clean", "creator"], as_index=False
                )["checkouts"]
                .sum()
                .sort_values("checkouts", ascending=False)
                .head(25)
            )
            results.columns = ["Title", "Author / Creator", "Total checkouts"]
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            first_title = results.iloc[0]["Title"]
            history = (
                matches[matches["title_clean"] == first_title]
                .groupby("period", as_index=False)["checkouts"]
                .sum()
            )

            fig = px.line(
                history,
                x="period",
                y="checkouts",
                markers=True,
                title=f"Checkout history — {str(first_title)[:55]}"
            )
            fig.update_traces(line=dict(color=AMBER, width=3))
            st.plotly_chart(
                style_chart(fig, 320, False),
                use_container_width=True
            )


# ===================== TAB 4: MODEL INSIGHTS =====================
with tab_model:
    st.markdown(
        '<div class="section-title">🤖 Model insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div style="padding:20px;border-radius:18px;
        background:rgba(255,255,255,.045);
        border:1px solid rgba(255,255,255,.09);">
        <b>Algorithm:</b> Random Forest Classifier<br>
        <b>Input features:</b> checkoutyear, checkoutmonth<br>
        <b>Target:</b> High (1) when record checkouts are above the
        dataset median ({df_all['checkouts'].median():.0f});
        otherwise Low (0).<br>
        <b>Interpretation:</b> the model estimates a demand class from
        the period, not the demand of a specific title.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        features = list(
            getattr(model, "feature_names_in_", [
                "checkoutyear", "checkoutmonth"
            ])
        )
        importance = getattr(model, "feature_importances_", None)

        if importance is not None and len(importance) == len(features):
            importance_df = pd.DataFrame({
                "Feature": [
                    "Year" if "year" in f.lower() else "Month"
                    for f in features
                ],
                "Importance": importance
            })

            fig = px.bar(
                importance_df,
                x="Feature",
                y="Importance",
                color="Feature",
                color_discrete_sequence=[PURPLE, CYAN],
                title="Model feature importance"
            )
            st.plotly_chart(
                style_chart(fig, 350, False),
                use_container_width=True
            )
        else:
            st.info("Feature importance is not available for this model.")

    with right:
        actual = (
            df_all.groupby("period", as_index=False)["demand"]
            .mean()
            .sort_values("period")
        )
        actual["actual_share"] = actual["demand"] * 100
        actual["model_probability"] = [
            predict_period(row.period.year, row.period.month)[0] * 100
            for row in actual.itertuples()
        ]
        actual["month_label"] = actual["period"].dt.strftime("%b %Y")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual["month_label"],
            y=actual["actual_share"],
            name="Actual high-class share",
            mode="lines+markers",
            line=dict(color=AMBER, width=3)
        ))
        fig.add_trace(go.Scatter(
            x=actual["month_label"],
            y=actual["model_probability"],
            name="Model probability",
            mode="lines+markers",
            line=dict(color=PURPLE, width=3, dash="dot")
        ))
        fig.update_layout(title="Actual share vs model probability")
        fig.update_yaxes(title="Percent", range=[0, 100])

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True
        )


# ===================== FOOTER =====================
st.markdown("""
<div style="text-align:center;color:#64748b;
            font-size:.8rem;margin-top:30px;">
Built with Streamlit • Plotly • scikit-learn
<br>Library Book Demand Prediction Project 📚
</div>
""", unsafe_allow_html=True)
