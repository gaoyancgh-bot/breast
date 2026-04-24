import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Breast Cancer Screening Simulation", layout="wide")
st.title("Breast Cancer Screening Simulation")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def next_weekday(d, weekday):
    """Return the next occurrence of weekday (0=Mon … 6=Sun) on or after d."""
    days_ahead = weekday - d.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)

def get_report_date(result_date, is_abnormal):
    """
    Abnormal → next Mon (0), Wed (2), or Fri (4), whichever is earliest on/after result_date.
    Normal   → next Tuesday (1) on/after result_date.
    """
    if is_abnormal:
        candidates = [next_weekday(result_date, wd) for wd in [0, 2, 4]]
        return min(candidates)
    else:
        return next_weekday(result_date, 1)

def apply_delay(base_date, day_of_week_str, delay_probs, rng):
    """
    Given a base date and the delay probability row for that day-of-week,
    sample a delay bucket and return the resulting date.
    day_of_week_str: 'Mon','Tue','Wed','Thu','Fri','Sat'
    delay_probs: dict with keys 'd+1','d+2','d+3','d+4+'
    """
    buckets = ['d+1', 'd+2', 'd+3', 'd+4','d+5+']
    probs = np.array([delay_probs[b] for b in buckets], dtype=float)
    total = probs.sum()
    if total == 0:
        probs = np.ones(5) / 5
    else:
        probs = probs / total

    bucket = rng.choice(buckets, p=probs)
    if bucket == 'd+1':
        delta = 1
    elif bucket == 'd+2':
        delta = 2
    elif bucket == 'd+3':
        delta = 3
    elif bucket == 'd+4':
        delta = 4
    else:
        delta = int(rng.integers(5, 51))  # uniform 5–50
    return base_date + timedelta(days=delta)

def simulate(
    start_date,
    monthly_demand,
    age_dist,          # dict: {'<50':p, '50-59':p, '60-69':p, '>=70':p}
    abnormal_rates,    # dict same keys
    third_read_pct,    # dict: {'normal': p, 'abnormal': p}
    day_pct,           # dict: {'Mon':p, ..., 'Sat':p}
    r1_delays,         # dict of dicts: day -> {d+1,d+2,d+3,d+4+}
    r2_delays,
    r3_delays,
    seed=42
):
    rng = np.random.default_rng(seed)
    days_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    day_weights = np.array([day_pct[d] for d in days_list], dtype=float)
    day_weights /= day_weights.sum()

    age_groups = ['<50', '50-59', '60-69', '>=70']
    age_weights = np.array([age_dist[a] for a in age_groups], dtype=float)
    age_weights /= age_weights.sum()

    records = []

    for month_idx, demand in enumerate(monthly_demand):
        n = int(round(demand))
        # Build list of dates in this month
        year = start_date.year + (start_date.month - 1 + month_idx) // 12
        month = (start_date.month - 1 + month_idx) % 12 + 1
        _, days_in_month = calendar.monthrange(year, month)
        month_start = date(year, month, 1)

        # Collect all Mon–Sat dates in the month
        valid_dates = [
            month_start + timedelta(days=i)
            for i in range(days_in_month)
            if (month_start + timedelta(days=i)).weekday() < 6  # 0–5 = Mon–Sat
        ]

        # Assign mammogram dates weighted by day-of-week percentages
        date_day_weights = np.array(
            [day_weights[(month_start + timedelta(days=i)).weekday()]
             for i in range(days_in_month)
             if (month_start + timedelta(days=i)).weekday() < 6],
            dtype=float
        )
        date_day_weights /= date_day_weights.sum()

        mammo_dates = rng.choice(valid_dates, size=n, p=date_day_weights)

        for mammo_date in mammo_dates:
            mammo_date = pd.Timestamp(mammo_date).date()
            dow = days_list[mammo_date.weekday()]  # Mon–Sat

            # Age group
            age_group = rng.choice(age_groups, p=age_weights)

            # Abnormal?
            is_abnormal = rng.random() < abnormal_rates[age_group]

            # Reading 1 date
            r1_date = apply_delay(mammo_date, dow, r1_delays[dow], rng)

            # Reading 2 date (based on R1 day-of-week)
            r1_dow = days_list[r1_date.weekday() % 6] if r1_date.weekday() < 6 else 'Mon'
            r2_date = apply_delay(r1_date, r1_dow, r2_delays[r1_dow], rng)

            # Third reading?
            third_read_key = 'abnormal' if is_abnormal else 'normal'
            needs_r3 = rng.random() < third_read_pct[third_read_key]

            if needs_r3:
                r2_dow = days_list[r2_date.weekday() % 6] if r2_date.weekday() < 6 else 'Mon'
                r3_date = apply_delay(r2_date, r2_dow, r3_delays[r2_dow], rng)
                final_read_date = r3_date
            else:
                r3_date = None
                final_read_date = r2_date

            report_date = get_report_date(final_read_date, is_abnormal)

            records.append({
                'mammo_date': mammo_date,
                'age_group': age_group,
                'is_abnormal': is_abnormal,
                'r1_date': r1_date,
                'r2_date': r2_date,
                'needs_r3': needs_r3,
                'r3_date': r3_date,
                'final_read_date': final_read_date,
                'report_date': report_date,
                'mammo_to_report_days': (report_date - mammo_date).days,
            })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────
# SIDEBAR – GLOBAL INPUTS
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation Settings")
    start_date = st.date_input("Start Month (pick any day in the start month)",
                               value=date(2025, 1, 1))
    start_date = date(start_date.year, start_date.month, 1)
    duration = st.number_input("Duration (months)", min_value=1, max_value=60, value=6)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Demand & Clinical", "📅 Scheduling", "▶️ Run Simulation", "📊 Results"]
)

# ══════════════════════════════════════════════
# TAB 1 – DEMAND & CLINICAL PARAMETERS
# ══════════════════════════════════════════════

with tab1:
    st.subheader("Monthly Mammogram Demand")

    # Build default demand table
    month_labels = []
    for i in range(int(duration)):
        y = start_date.year + (start_date.month - 1 + i) // 12
        m = (start_date.month - 1 + i) % 12 + 1
        month_labels.append(f"{calendar.month_abbr[m]} {y}")

    default_demand = pd.DataFrame(
        {"Month": month_labels, "Demand": [1000] * int(duration)}
    )
    demand_df = st.data_editor(
        default_demand, num_rows="fixed", use_container_width=True,
        column_config={"Demand": st.column_config.NumberColumn(min_value=0, step=1)}
    )

    st.subheader("Age Group Distribution (%)")
    age_groups = ['<50', '50-59', '60-69', '>=70']
    default_age = pd.DataFrame({
        "Age Group": age_groups,
        "Distribution (%)": [20.0, 40.0, 30.0, 10.0]
    })
    age_df = st.data_editor(
        default_age, num_rows="fixed", use_container_width=True,
        column_config={"Distribution (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1)}
    )
    age_total = age_df["Distribution (%)"].sum()
    if abs(age_total - 100) > 0.5:
        st.warning(f"Age distribution sums to {age_total:.1f}% (should be 100%)")

    st.subheader("Abnormal Rate per Age Group (%)")
    default_abnormal = pd.DataFrame({
        "Age Group": age_groups,
        "Abnormal Rate (%)": [8.0, 10.0, 6.0, 4.0]
    })
    abnormal_df = st.data_editor(
        default_abnormal, num_rows="fixed", use_container_width=True,
        column_config={"Abnormal Rate (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1)}
    )

    st.subheader("3rd Reading Requirements (%)")
    col1, col2 = st.columns(2)
    with col1:
        r3_normal_pct = st.number_input("% Normal cases requiring 3rd reading", 0.0, 100.0, 5.0, 0.1)
    with col2:
        r3_abnormal_pct = st.number_input("% Abnormal cases requiring 3rd reading", 0.0, 100.0, 20.0, 0.1)

# ══════════════════════════════════════════════
# TAB 2 – SCHEDULING PARAMETERS
# ══════════════════════════════════════════════

with tab2:
    st.subheader("Mammogram Day-of-Week Distribution (%)")
    days_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    default_day_pct = pd.DataFrame({
        "Day": days_list,
        "% of Mammograms": [18.0, 18.0, 18.0, 18.0, 18.0, 10.0]
    })
    day_pct_df = st.data_editor(
        default_day_pct, num_rows="fixed", use_container_width=True,
        column_config={"% of Mammograms": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1)}
    )
    day_total = day_pct_df["% of Mammograms"].sum()
    if abs(day_total - 100) > 0.5:
        st.warning(f"Day distribution sums to {day_total:.1f}% (should be 100%)")

    def make_delay_table(label, key, defaults=None):
        st.markdown(f"**{label}**")
        if defaults is None:
            defaults = [[20.0, 20.0, 20.0, 20.0, 20.0]] * 6
        df = pd.DataFrame({
            "Day":  days_list,
            "d+1 (%)": [r[0] for r in defaults],
            "d+2 (%)": [r[1] for r in defaults],
            "d+3 (%)": [r[2] for r in defaults],
            "d+4 (%)": [r[3] for r in defaults],
            "d+5+ (%)": [r[4] for r in defaults],
        })
        edited = st.data_editor(
            df, num_rows="fixed", use_container_width=True,
            key=key,
            column_config={
                c: st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1)
                for c in ["d+1 (%)", "d+2 (%)", "d+3 (%)","d+4 (%)",  "d+5+ (%)"]
            }
        )
        # Warn if any row doesn't sum to ~100
        row_sums = edited[["d+1 (%)", "d+2 (%)", "d+3 (%)","d+4 (%)",  "d+5+ (%)"]].sum(axis=1)
        for i, s in enumerate(row_sums):
            if abs(s - 100) > 0.5:
                st.warning(f"Row {days_list[i]} sums to {s:.1f}% (should be 100%)")
        return edited

    r1_defaults = [
        [40.0, 30.0, 20.0, 5.0, 5.0],
        [40.0, 30.0, 20.0, 5.0, 5.0],
        [40.0, 30.0, 20.0, 5.0, 5.0],
        [70.0, 5.0, 5.0, 15.0, 5],
        [5.0, 5.0, 70.0, 15.0, 5],
        [5.0, 70.0, 10.0, 10.0,5],
    ]

    st.subheader("Reading 1 Delay from Mammogram Date")
    r1_df = make_delay_table("Reading 1 delay probabilities by mammogram day-of-week", key="r1_delay", defaults=r1_defaults)

    st.subheader("Reading 2 Delay from Reading 1 Date")
    r2_df = make_delay_table("Reading 2 delay probabilities by Reading 1 day-of-week", key="r2_delay",defaults=r1_defaults)

    st.subheader("Reading 3 Delay from Reading 2 Date")
    r3_df = make_delay_table("Reading 3 delay probabilities by Reading 2 day-of-week", key="r3_delay", defaults=r1_defaults)
# ══════════════════════════════════════════════
# TAB 3 – RUN SIMULATION
# ══════════════════════════════════════════════

with tab3:
    st.subheader("Run the Simulation")
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        # Parse inputs
        monthly_demand = demand_df["Demand"].tolist()

        age_dist = dict(zip(age_df["Age Group"], age_df["Distribution (%)"] / 100))
        abnormal_rates = dict(zip(abnormal_df["Age Group"], abnormal_df["Abnormal Rate (%)"] / 100))
        third_read_pct = {
            'normal': r3_normal_pct / 100,
            'abnormal': r3_abnormal_pct / 100
        }
        day_pct = dict(zip(day_pct_df["Day"], day_pct_df["% of Mammograms"] / 100))

        def parse_delay_df(df):
            result = {}
            for _, row in df.iterrows():
                result[row["Day"]] = {
                    'd+1': row["d+1 (%)"] / 100,
                    'd+2': row["d+2 (%)"] / 100,
                    'd+3': row["d+3 (%)"] / 100,
                    'd+4': row["d+4 (%)"] / 100,
                    'd+5+': row["d+5+ (%)"] / 100,
                }
            return result

        r1_delays = parse_delay_df(r1_df)
        r2_delays = parse_delay_df(r2_df)
        r3_delays = parse_delay_df(r3_df)

        with st.spinner("Simulating…"):
            df = simulate(
                start_date=start_date,
                monthly_demand=monthly_demand,
                age_dist=age_dist,
                abnormal_rates=abnormal_rates,
                third_read_pct=third_read_pct,
                day_pct=day_pct,
                r1_delays=r1_delays,
                r2_delays=r2_delays,
                r3_delays=r3_delays,
                seed=int(seed)
            )

        st.session_state["sim_df"] = df
        st.success(f"Simulation complete — {len(df):,} mammograms simulated.")
        #st.dataframe(df, use_container_width=True)
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == object and isinstance(display_df[col].dropna().iloc[0], date):
                display_df[col] = display_df[col].apply(lambda d: d.strftime("%Y-%m-%d (%a)") if pd.notna(d) else "")
        st.dataframe(display_df, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – RESULTS & CHARTS
# ══════════════════════════════════════════════

with tab4:
    if "sim_df" not in st.session_state:
        st.info("Run the simulation first (Tab 3).")
    else:
        df = st.session_state["sim_df"]

        # ── Summary KPIs ──────────────────────────────
        st.subheader("Summary Statistics")
        total = len(df)
        n_abnormal = df["is_abnormal"].sum()
        n_r3 = df["needs_r3"].sum()
        avg_days = df["mammo_to_report_days"].mean()
        median_days = df["mammo_to_report_days"].median()
        within_2weeks = sum(df["mammo_to_report_days"]<=14)/total

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Mammograms", f"{total:,}")
        k2.metric("Abnormal Cases", f"{n_abnormal:,} ({n_abnormal/total*100:.1f}%)")
        k3.metric("3rd Reading Cases", f"{n_r3:,} ({n_r3/total*100:.1f}%)")
        k4.metric("Avg Mammo→Report (days)", f"{avg_days:.1f}")
        k5.metric("Median Mammo→Report (days)", f"{median_days:.1f}")
        k6.metric("Meet Target 2 weeks", f"{within_2weeks:.1f}")

        st.divider()

        # ── Turnaround distribution ───────────────────
        st.subheader("Turnaround Time Distribution (Mammogram → Report)")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_hist = px.histogram(
                df, x="mammo_to_report_days",
                color="is_abnormal",
                barmode="overlay",
                labels={"mammo_to_report_days": "Days", "is_abnormal": "Abnormal"},
                title="All Cases",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                opacity=0.75
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            fig_box = px.box(
                df, x="age_group", y="mammo_to_report_days",
                color="is_abnormal",
                labels={"mammo_to_report_days": "Days", "age_group": "Age Group", "is_abnormal": "Abnormal"},
                title="By Age Group",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                category_orders={"age_group": ["<50", "50-59", "60-69", ">=70"]}
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Monthly volume ────────────────────────────
        st.subheader("Monthly Volume")
        df["mammo_month"] = pd.to_datetime(df["mammo_date"]).dt.to_period("M").astype(str)
        monthly = df.groupby("mammo_month").agg(
            total=("mammo_date", "count"),
            abnormal=("is_abnormal", "sum"),
            avg_tat=("mammo_to_report_days", "mean")
        ).reset_index()

        fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vol.add_trace(go.Bar(x=monthly["mammo_month"], y=monthly["total"],
                                  name="Total", marker_color="#3498db"), secondary_y=False)
        fig_vol.add_trace(go.Bar(x=monthly["mammo_month"], y=monthly["abnormal"],
                                  name="Abnormal", marker_color="#e74c3c"), secondary_y=False)
        fig_vol.add_trace(go.Scatter(x=monthly["mammo_month"], y=monthly["avg_tat"],
                                      name="Avg TAT (days)", mode="lines+markers",
                                      line=dict(color="#f39c12", width=2)), secondary_y=True)
        fig_vol.update_layout(barmode="overlay", title="Monthly Mammogram Volume & Avg Turnaround",
                               xaxis_title="Month")
        fig_vol.update_yaxes(title_text="Count", secondary_y=False)
        fig_vol.update_yaxes(title_text="Avg TAT (days)", secondary_y=True)
        st.plotly_chart(fig_vol, use_container_width=True)

        # ── Reading workload by date ──────────────────
        st.subheader("Daily Reading Workload")
        r1_counts = df.groupby("r1_date").size().reset_index(name="R1")
        r2_counts = df.groupby("r2_date").size().reset_index(name="R2")
        r3_sub = df[df["needs_r3"]].copy()
        r3_counts = r3_sub.groupby("r3_date").size().reset_index(name="R3") if not r3_sub.empty else pd.DataFrame(columns=["r3_date", "R3"])

        workload = (
            r1_counts.rename(columns={"r1_date": "date"})
            .merge(r2_counts.rename(columns={"r2_date": "date"}), on="date", how="outer")
            .merge(r3_counts.rename(columns={"r3_date": "date"}), on="date", how="outer")
            .fillna(0)
            .sort_values("date")
        )
        workload["date"] = pd.to_datetime(workload["date"])

        fig_wl = go.Figure()
        fig_wl.add_trace(go.Scatter(x=workload["date"], y=workload["R1"],
                                     name="Reading 1", mode="lines", line=dict(color="#3498db")))
        fig_wl.add_trace(go.Scatter(x=workload["date"], y=workload["R2"],
                                     name="Reading 2", mode="lines", line=dict(color="#9b59b6")))
        fig_wl.add_trace(go.Scatter(x=workload["date"], y=workload["R3"],
                                     name="Reading 3", mode="lines", line=dict(color="#e67e22")))
        fig_wl.update_layout(title="Daily Reading Workload", xaxis_title="Date", yaxis_title="Cases")
        st.plotly_chart(fig_wl, use_container_width=True)

        # ── Age group breakdown ───────────────────────
        st.subheader("Age Group Breakdown")
        age_summary = df.groupby("age_group").agg(
            count=("mammo_date", "count"),
            abnormal_count=("is_abnormal", "sum"),
            avg_tat=("mammo_to_report_days", "mean")
        ).reset_index()
        age_summary["abnormal_rate"] = age_summary["abnormal_count"] / age_summary["count"] * 100
        age_summary["avg_tat"] = age_summary["avg_tat"].round(1)
        age_summary["abnormal_rate"] = age_summary["abnormal_rate"].round(1)
        st.dataframe(age_summary, use_container_width=True)

        # ── Report date distribution ──────────────────
        st.subheader("Report Generation Date Distribution")
        df["report_dow"] = pd.to_datetime(df["report_date"]).dt.day_name()
        report_dow_counts = df.groupby(["report_dow", "is_abnormal"]).size().reset_index(name="count")
        fig_rdow = px.bar(
            report_dow_counts, x="report_dow", y="count", color="is_abnormal",
            barmode="group",
            labels={"report_dow": "Day of Week", "count": "Reports", "is_abnormal": "Abnormal"},
            title="Reports by Day of Week",
            color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
            category_orders={"report_dow": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
        )
        st.plotly_chart(fig_rdow, use_container_width=True)

        # ── Raw data download ─────────────────────────
        st.subheader("Download Raw Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "simulation_results.csv", "text/csv")
