import os
import io
import json
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

import analytics as A  # local module

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Login / Activity Analytics + Gemini", layout="wide")
st.title("🔐 Login/Activity Analytics Toolkit (In-Memory)")
st.caption("Upload CSV/Excel, compute metrics & charts completely in memory, and generate insights with Gemini.")

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    tz = st.text_input("Timezone (e.g., UTC, US/Eastern)", value="UTC")
    session_timeout = st.number_input("Session timeout (minutes, 0=disable)", min_value=0, value=30, step=5)
    lookback_days = st.slider("Org activity lookback (days)", 7, 365, 30, step=1)

    st.markdown("---")
    st.subheader("🤖 Gemini")

    # BACKEND SOURCES for API key (safe):
    # 1) Streamlit Secrets: st.secrets["GOOGLE_API_KEY"]
    # 2) Environment variable: os.environ["GOOGLE_API_KEY"]
    # The sidebar field allows override at runtime (optional)
    default_secret = ""
    try:
        default_secret = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    default_env = os.environ.get("GOOGLE_API_KEY", "")
    prefill_key = default_secret or default_env

    gemini_api_key = st.text_input("Gemini API Key (optional)", type="password", value=prefill_key)
    gemini_model = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash"], index=0)
    max_tokens = st.slider("Max output tokens", 200, 2000, 700, 50)
    include_sample_rows = st.checkbox("Include a tiny, anonymized sample in Gemini context", value=False)

# -------------------------------
# File upload
# -------------------------------
uploaded = st.file_uploader("Upload CSV or Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
sheet = None
if uploaded and uploaded.name.lower().endswith((".xlsx", ".xls")):
    sheet = st.text_input("Excel sheet name or index (e.g., 0 or Sheet1)", value="0")

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

st.markdown("### 💬 Gemini Copilot")
st.caption("Ask anything about the computed metrics (e.g., 'compare DAU last 14 days vs previous 14').")
user_prompt = st.text_input("Your question", placeholder="e.g., What stands out in the last 30 days?")

# keep results between reruns
if "results" not in st.session_state:
    st.session_state["results"] = None


# -------------------------------
# Helpers
# -------------------------------
AUTO_MAP = {
    # common variants -> canonical
    "login_time": "logon_date",
    "logon_time": "logon_date",
    "login": "logon_date",
    "timestamp": "logon_date",
    "time": "logon_date",
    "logout_time": "logout_date",
    "signout_time": "logout_date",
    "userid": "user_id",
    "user": "user_id",
    "email": "email_address",
}

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    lower_map = {c.lower().strip(): c for c in df.columns}
    for k, v in AUTO_MAP.items():
        if k in lower_map and v not in df.columns:
            cols[lower_map[k]] = v
    df = df.rename(columns=cols)
    return df

def _load_df(uploaded_file, sheet_name: Optional[str]):
    if uploaded_file is None:
        return None, "No file uploaded."
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            sn = int(sheet_name) if isinstance(sheet_name, str) and sheet_name.isdigit() else sheet_name
            df = pd.read_excel(uploaded_file, sheet_name=sn)
        df = _auto_map_columns(df)
        return df, None
    except Exception as e:
        return None, f"Failed to read file: {e}"

def _dfs_to_csv_bytes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, bytes]:
    out = {}
    for name, d in dfs.items():
        buf = io.StringIO()
        try:
            d.to_csv(buf, index=False)
        except Exception:
            d.to_csv(buf)  # handle pivot/retention with index
        out[f"{name}.csv"] = buf.getvalue().encode("utf-8")
    return out

def _fig_bytes_zip(fig_bytes: Dict[str, bytes]) -> Dict[str, bytes]:
    return {f"{k}.png": v for k, v in fig_bytes.items() if v}

def _gemini_generate(prompt: str, metrics_json: dict, api_key: str, model_name: str, max_tokens: int) -> str:
    if not api_key:
        return "⚠️ Provide a Gemini API key in the sidebar or via Secrets/Env."
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"⚠️ google-generativeai not installed: {e}"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # Convert non-serializable types (Timestamp, etc.)
        safe_json = json.dumps(metrics_json, default=str)

        sys_prompt = (
            "You are an analytics copilot. Use the provided metrics to answer the user's question. "
            "Be precise, cite specific values, and keep responses under 250 words.\n\n"
            f"METRICS(JSON): {safe_json[:180000]}"
        )
        full_prompt = sys_prompt + "\n\nUSER QUERY:\n" + prompt
        resp = model.generate_content(full_prompt, generation_config={"max_output_tokens": max_tokens})
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        return text or "⚠️ Empty response from Gemini."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

def gemini_summary_and_plan(metrics_json: dict, api_key: str, model_name: str, max_tokens: int) -> str:
    """Executive summary + insights + risks + 2-week action plan."""
    if not api_key:
        return "⚠️ Provide a Gemini API key in the sidebar or via Secrets/Env."
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"⚠️ google-generativeai not installed: {e}"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        safe_json = json.dumps(metrics_json, default=str)
        prompt = (
            "You are a data/PM copilot. Produce:\n"
            "1) Executive summary (<=120 words)\n"
            "2) Key insights (bullet list w/ numbers)\n"
            "3) Risks/Anomalies\n"
            "4) 2-week action plan grouped by Security, Engagement, Product, and Ops. "
            "For each action: Owner role, expected impact, and 1 measurable KPI.\n"
            "Use crisp bullets. Only use the data provided.\n\n"
            f"DATA(JSON): {safe_json[:180000]}"
        )
        resp = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        return text or "⚠️ Empty response from Gemini."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


# -------------------------------
# Core analytics runner
# -------------------------------
def build_business_insights(df: pd.DataFrame, lookback_days: int) -> dict:
    now = pd.Timestamp.today().normalize()
    start = now - pd.Timedelta(days=lookback_days)
    scope = df.loc[(df["date"] >= start) & (df["date"] <= now)].copy()

    for c in ["Industry", "org_name", "browser", "operating_system", "hour"]:
        if c not in scope.columns:
            scope[c] = pd.NA

    by_industry = (scope.groupby("Industry")["user_id"].nunique()
                   .sort_values(ascending=False).reset_index(names=["Industry", "unique_users"]))
    by_org = (scope.groupby("org_name")["user_id"].nunique()
              .sort_values(ascending=False).reset_index(names=["org_name", "unique_users"]))
    by_browser = (scope.groupby("browser")["user_id"].nunique()
                  .sort_values(ascending=False).reset_index(names=["browser", "unique_users"]))
    by_os = (scope.groupby("operating_system")["user_id"].nunique()
             .sort_values(ascending=False).reset_index(names=["operating_system", "unique_users"]))
    by_hour = (scope.groupby("hour")["user_id"].nunique()
               .sort_values(ascending=False).reset_index(names=["hour", "unique_users"]))

    return {
        "lookback_days": lookback_days,
        "window_start": str(start.date()),
        "window_end": str(now.date()),
        "by_industry": by_industry,
        "by_org_top10": by_org.head(10),
        "by_browser": by_browser,
        "by_os": by_os,
        "by_hour": by_hour,
    }

def run_analytics(df: pd.DataFrame, tz: str, session_timeout: int, lookback_days: int):
    # Normalize / auto-map
    df = _auto_map_columns(df)
    for col in ["user_id", "org_id", "org_name", "browser", "browser_version",
                "operating_system", "Industry", "Domain"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Enrich
    df = A.derive_email_domain(df)
    df = A.add_time_parts(df, tz=tz)
    timeout = None if session_timeout == 0 else session_timeout
    df = A.compute_session_minutes(df, session_timeout_min=timeout)
    df["logout_date_effective"] = A.derive_session_end(df, timeout)

    # Metrics
    daily = A.daily_metrics(df)
    weekly = A.weekly_metrics(df)
    monthly = A.monthly_metrics(df)
    duration_stats = A.session_duration_stats(df)
    org_30d = A.org_activity_last_30d(df)
    bshare = A.browser_os_share(df)
    ret_tbl = A.retention_table(df)
    concurrency = A.estimate_daily_peak_concurrency(df)

    # Charts
    figs = {}
    st.markdown("#### Core Trends")
    if not daily.empty:
        fig, png = A.make_time_series_fig(daily, "date", "dau", "Daily Active Users (DAU)")
        figs["dau_daily"] = png; st.pyplot(fig)

        fig, png = A.make_time_series_fig(daily, "date", "sessions", "Daily Sessions")
        figs["sessions_daily"] = png; st.pyplot(fig)

        daily_plt = daily.copy()
        daily_plt["stickiness_smoothed"] = daily_plt["stickiness_dau_over_mau"].rolling(7, min_periods=1).mean()
        fig, png = A.make_time_series_fig(daily_plt, "date", "stickiness_smoothed", "Stickiness (DAU/MAU, 7d avg)")
        figs["stickiness_daily"] = png; st.pyplot(fig)

    if df["session_minutes"].notna().any():
        st.markdown("#### Session Duration")
        fig, png = A.make_hist_fig(df["session_minutes"], "Session Duration (trimmed at 99th pct)", "minutes", bins=60)
        if fig: figs["session_duration_hist"] = png; st.pyplot(fig)

    if df["hour"].notna().any() and df["dow"].notna().any():
        st.markdown("#### Hour x Day Heatmap")
        fig, png = A.make_heatmap_hour_dow_fig(df)
        figs["heatmap_hour_dow"] = png; st.pyplot(fig)

    if not ret_tbl.empty:
        st.markdown("#### Retention")
        fig, png = A.make_retention_heatmap_fig(ret_tbl)
        if fig: figs["retention_heatmap"] = png; st.pyplot(fig)

    if not concurrency.empty:
        st.markdown("#### Estimated Peak Concurrency")
        fig, png = A.make_time_series_fig(concurrency, "date", "peak_concurrency", "Estimated Peak Concurrency (Daily)")
        figs["concurrency_daily"] = png; st.pyplot(fig)

    # Business rollups & quick bars
    st.markdown("### Business Rollups (Lookback Window)")
    biz = build_business_insights(df, lookback_days=lookback_days)
    if not biz["by_industry"].empty:
        st.markdown("**Top industries (unique users)**")
        st.bar_chart(biz["by_industry"].set_index("Industry"))
    if not biz["by_org_top10"].empty:
        st.markdown("**Top orgs**")
        st.bar_chart(biz["by_org_top10"].set_index("org_name"))
    if not biz["by_browser"].empty:
        st.markdown("**Browser share**")
        st.bar_chart(biz["by_browser"].set_index("browser"))
    if not biz["by_os"].empty:
        st.markdown("**OS share**")
        st.bar_chart(biz["by_os"].set_index("operating_system"))
    if not biz["by_hour"].empty:
        st.markdown("**Active users by hour**")
        st.bar_chart(biz["by_hour"].set_index("hour"))

    # Safe JSON for Gemini
    daily2, weekly2, monthly2 = daily.copy(), weekly.copy(), monthly.copy()
    if "date" in daily2.columns: daily2["date"] = daily2["date"].astype(str)
    if "week" in weekly2.columns: weekly2["week"] = weekly2["week"].astype(str)
    if "month" in monthly2.columns: monthly2["month"] = monthly2["month"].astype(str)
    conc2 = concurrency.copy()
    if not conc2.empty and "date" in conc2.columns:
        conc2["date"] = conc2["date"].astype(str)
    org2 = org_30d.copy()
    if not org2.empty and "last_seen" in org2.columns:
        org2["last_seen"] = org2["last_seen"].astype(str)

    metrics_json = {
        "window": {"lookback_days": lookback_days, "timezone": tz},
        "daily_tail": daily2.tail(14).to_dict(orient="list") if not daily2.empty else {},
        "weekly_tail": weekly2.tail(8).to_dict(orient="list") if not weekly2.empty else {},
        "monthly": monthly2.to_dict(orient="list") if not monthly2.empty else {},
        "duration_stats": duration_stats.to_dict(orient="records"),
        "top_orgs_30d": org2.head(10).to_dict(orient="records") if not org2.empty else [],
        "concurrency_tail": conc2.tail(14).to_dict(orient="list") if not conc2.empty else {},
        "biz_rollups": {
            "by_industry": biz["by_industry"].to_dict(orient="records"),
            "by_org_top10": biz["by_org_top10"].to_dict(orient="records"),
            "by_browser": biz["by_browser"].to_dict(orient="records"),
            "by_os": biz["by_os"].to_dict(orient="records"),
            "by_hour": biz["by_hour"].to_dict(orient="records"),
            "window_start": biz["window_start"],
            "window_end": biz["window_end"],
        },
    }

    # Optionally include tiny anonymized sample (no names/emails)
    if include_sample_rows and not df.empty:
        cols = [c for c in df.columns if c not in ["email_address", "first_name", "last_name", "Full Name"]]
        sample = df[cols].head(50).copy()
        for c in ["logon_date_parsed", "logout_date_parsed", "date", "month"]:
            if c in sample.columns:
                sample[c] = sample[c].astype(str)
        metrics_json["sample_rows_head"] = sample.to_dict(orient="records")

    dfs = {
        "metrics_daily": daily, "metrics_weekly": weekly, "metrics_monthly": monthly,
        "session_duration_stats": duration_stats, "org_activity_30d": org_30d,
        "browser_os_share": bshare, "retention_table": ret_tbl, "concurrency_daily_peak": concurrency
    }
    return dfs, figs, metrics_json


# -------------------------------
# Run button
# -------------------------------
if run_btn:
    df, err = _load_df(uploaded, sheet)
    if err:
        st.error(err)
    else:
        with st.spinner("Crunching numbers (in memory)..."):
            results = run_analytics(df, tz=tz, session_timeout=session_timeout, lookback_days=lookback_days)
            st.session_state["results"] = {
                "dfs": results[0], "figs": results[1], "metrics_json": results[2]
            }

# -------------------------------
# UI: Tables / Downloads / Chat / Plan
# -------------------------------
res = st.session_state["results"]
tabs = st.tabs(["Tables", "Downloads", "Gemini Chat", "Insights & Action Plan"])

with tabs[0]:
    st.subheader("Data Tables")
    if not res:
        st.info("Upload a file and click **Run Analysis** to begin.")
    else:
        for name, d in res["dfs"].items():
            st.markdown(f"**{name}**")
            st.dataframe(d)

with tabs[1]:
    st.subheader("Download your results (in memory)")
    if not res:
        st.info("Run analysis first.")
    else:
        csv_bytes = _dfs_to_csv_bytes(res["dfs"])
        fig_bytes = _fig_bytes_zip(res["figs"])
        for fname, b in csv_bytes.items():
            st.download_button(f"Download {fname}", b, file_name=fname, mime="text/csv")
        for fname, b in fig_bytes.items():
            st.download_button(f"Download {fname}", b, file_name=fname, mime="image/png")

with tabs[2]:
    st.subheader("Chat with Gemini")
    if user_prompt and res:
        st.info("Querying Gemini...")
        reply = _gemini_generate(user_prompt, res["metrics_json"], gemini_api_key, gemini_model, max_tokens)
        st.markdown("**Gemini says:**")
        st.write(reply)
    elif not res:
        st.info("Run analysis first.")

with tabs[3]:
    st.subheader("Auto Insights & 2-Week Action Plan")
    if not res:
        st.info("Run analysis first.")
    else:
        if st.button("🧠 Generate executive summary & plan", type="primary", use_container_width=True):
            with st.spinner("Asking Gemini for an executive summary and action plan..."):
                text = gemini_summary_and_plan(res["metrics_json"], gemini_api_key, gemini_model, max_tokens)
                st.markdown(text)
