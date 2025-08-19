
import io, json, zipfile
from typing import Dict, Any
import os
from pathlib import Path
import pandas as pd
import streamlit as st

import analytics as A

st.set_page_config(page_title="Login Analytics + Gemini (In-Memory)", layout="wide")
st.title("🔐 Login/Activity Analytics Toolkit (In-Memory)")
st.caption("Upload a CSV/Excel, compute metrics & charts completely in memory, and chat with Gemini.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Options")
    tz = st.text_input("Timezone (e.g., UTC, US/Eastern)", value="UTC")
    session_timeout = st.number_input("Session timeout (minutes, 0=disable)", min_value=0, value=30, step=5)

    st.markdown("---")
    st.subheader("🤖 Gemini")
    # No default value from env for safety; user can paste or use Streamlit Secrets via st.secrets
    default_key = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""
    gemini_api_key = st.text_input("Gemini API Key", type="password", value=default_key)
    gemini_model = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash"], index=0)
    max_tokens = st.slider("Max output tokens", min_value=200, max_value=2000, value=600, step=50)

# ---------- File upload ----------
uploaded = st.file_uploader("Upload CSV or Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
sheet = None
if uploaded and uploaded.name.lower().endswith((".xlsx", ".xls")):
    sheet = st.text_input("Excel sheet name or index", value="0")

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

st.markdown("### 💬 Gemini Copilot")
st.caption("Ask anything about the computed metrics. Example: 'compare DAU last 14 days vs previous 14'.")
user_prompt = st.text_input("Your question", placeholder="e.g., What stands out in the last 30 days?")

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = None

def load_df(uploaded_file, sheet_name):
    if uploaded_file is None:
        return None, "No file uploaded."
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            sn = int(sheet_name) if isinstance(sheet_name, str) and sheet_name.isdigit() else sheet_name
            df = pd.read_excel(uploaded_file, sheet_name=sn)
        return df, None
    except Exception as e:
        return None, f"Failed to read file: {e}"

def normalize_columns(df: pd.DataFrame):
    for col in ["user_id", "org_id", "org_name", "browser", "browser_version", "operating_system", "Industry", "Domain"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def dataframes_to_csv_bytes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, bytes]:
    out = {}
    for name, d in dfs.items():
        buf = io.StringIO()
        if isinstance(d, pd.DataFrame):
            d.to_csv(buf, index=False)
        else:
            # retention table can have index/columns; handle gracefully
            try:
                d.to_csv(buf)
            except Exception:
                continue
        out[f"{name}.csv"] = buf.getvalue().encode("utf-8")
    return out

def figures_to_png_bytes(fig_bytes_map: Dict[str, bytes]) -> Dict[str, bytes]:
    # Already PNG bytes; just rename
    return {f"{k}.png": v for k, v in fig_bytes_map.items() if v}

def build_zip(dfs_bytes: Dict[str, bytes], figs_bytes: Dict[str, bytes]) -> bytes:
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, b in dfs_bytes.items():
            z.writestr(f"csv/{fname}", b)
        for fname, b in figs_bytes.items():
            z.writestr(f"charts/{fname}", b)
    mem_zip.seek(0)
    return mem_zip.getvalue()

def run_analytics(df: pd.DataFrame, tz: str, session_timeout: int):
    df = normalize_columns(df)
    df = A.derive_email_domain(df)
    df = A.add_time_parts(df, tz=tz)

    timeout = None if session_timeout == 0 else session_timeout
    df = A.compute_session_minutes(df, session_timeout_min=timeout)
    df["logout_date_effective"] = A.derive_session_end(df, timeout)

    daily = A.daily_metrics(df)
    weekly = A.weekly_metrics(df)
    monthly = A.monthly_metrics(df)
    duration_stats = A.session_duration_stats(df)
    org_30d = A.org_activity_last_30d(df)
    bshare = A.browser_os_share(df)
    ret_tbl = A.retention_table(df)
    concurrency = A.estimate_daily_peak_concurrency(df)

    figs = {}
    if not daily.empty:
        fig, png = A.make_time_series_fig(daily, "date", "dau", "Daily Active Users (DAU)")
        figs["dau_daily"] = png
        st.pyplot(fig)

        fig, png = A.make_time_series_fig(daily, "date", "sessions", "Daily Sessions")
        figs["sessions_daily"] = png
        st.pyplot(fig)

        daily_plt = daily.copy()
        daily_plt["stickiness_smoothed"] = daily_plt["stickiness_dau_over_mau"].rolling(7, min_periods=1).mean()
        fig, png = A.make_time_series_fig(daily_plt, "date", "stickiness_smoothed", "Stickiness (DAU/MAU, 7d avg)")
        figs["stickiness_daily"] = png
        st.pyplot(fig)

    if df["session_minutes"].notna().any():
        fig, png = A.make_hist_fig(df["session_minutes"], "Session Duration (trimmed at 99th pct)", "minutes", bins=60)
        if fig:
            figs["session_duration_hist"] = png
            st.pyplot(fig)

    if df["hour"].notna().any() and df["dow"].notna().any():
        fig, png = A.make_heatmap_hour_dow_fig(df)
        figs["heatmap_hour_dow"] = png
        st.pyplot(fig)

    if not ret_tbl.empty:
        fig, png = A.make_retention_heatmap_fig(ret_tbl)
        if fig:
            figs["retention_heatmap"] = png
            st.pyplot(fig)

    if not concurrency.empty:
        fig, png = A.make_time_series_fig(concurrency, "date", "peak_concurrency", "Estimated Peak Concurrency (Daily)")
        figs["concurrency_daily"] = png
        st.pyplot(fig)

    dfs = {
        "metrics_daily": daily, "metrics_weekly": weekly, "metrics_monthly": monthly,
        "session_duration_stats": duration_stats, "org_activity_30d": org_30d,
        "browser_os_share": bshare, "retention_table": ret_tbl, "concurrency_daily_peak": concurrency
    }

    metrics_json = {
        "daily_tail": daily.tail(14).to_dict(orient="list") if not daily.empty else {},
        "weekly_tail": weekly.tail(8).to_dict(orient="list") if not weekly.empty else {},
        "monthly": monthly.to_dict(orient="list") if not monthly.empty else {},
        "duration_stats": duration_stats.to_dict(orient="records"),
        "top_orgs_30d": org_30d.head(10).to_dict(orient="records") if not org_30d.empty else [],
        "concurrency_tail": concurrency.tail(14).to_dict(orient="list") if not concurrency.empty else {},
    }

    return dfs, figs, metrics_json

def call_gemini(prompt: str, metrics_json: dict, api_key: str, model_name: str, max_tokens: int) -> str:
    if not api_key:
        return "⚠️ Provide a Gemini API key in the sidebar to use the chat."
    try:
        import google.generativeai as genai
    except Exception as e:
        return f"⚠️ google-generativeai not installed. Install it with: pip install google-generativeai\nDetails: {e}"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        sys_prompt = (
            "You are an analytics copilot. Use the provided metrics to answer the user's question. "
            "Be precise, cite specific values when possible, and keep responses under 250 words.\n\n"
            f"METRICS(JSON): {json.dumps(metrics_json)[:200000]}"
        )
        full_prompt = sys_prompt + "\n\nUSER:\n" + prompt
        resp = model.generate_content(full_prompt, generation_config={"max_output_tokens": max_tokens})
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            text = resp.candidates[0].content.parts[0].text
        return text or "⚠️ Empty response from Gemini."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ---------- Run ----------
if run_btn:
    df, err = load_df(uploaded, sheet)
    if err:
        st.error(err)
    else:
        with st.spinner("Crunching numbers (in memory)..."):
            dfs, figs, metrics_json = run_analytics(df, tz=tz, session_timeout=session_timeout)
            st.session_state["results"] = {"dfs": dfs, "figs": figs, "metrics_json": metrics_json}

# ---------- Display + Downloads ----------
res = st.session_state["results"]
if res:
    st.success("Analysis complete (in memory). No files saved to disk.")

    tabs = st.tabs(["Tables", "Downloads", "Gemini Chat"])
    with tabs[0]:
        st.subheader("Data Tables")
        for name, d in res["dfs"].items():
            st.markdown(f"**{name}**")
            st.dataframe(d)

    with tabs[1]:
        st.subheader("Download your results")
        csv_bytes = dataframes_to_csv_bytes(res["dfs"])
        fig_bytes = figures_to_png_bytes(res["figs"])

        # Individual downloads
        for fname, b in csv_bytes.items():
            st.download_button(f"Download {fname}", b, file_name=fname, mime="text/csv")
        for fname, b in fig_bytes.items():
            st.download_button(f"Download {fname}", b, file_name=fname, mime="image/png")

        # ZIP everything
        zip_all = build_zip(csv_bytes, fig_bytes)
        st.download_button("💾 Download ALL as ZIP", zip_all, file_name="analytics_outputs.zip", mime="application/zip")

    with tabs[2]:
        st.subheader("Chat with Gemini")
        if user_prompt:
            st.info("Querying Gemini...")
            reply = call_gemini(user_prompt, res["metrics_json"], gemini_api_key, gemini_model, max_tokens)
            st.markdown("**Gemini says:**")
            st.write(reply)
else:
    st.info("Upload a file and click **Run Analysis** to begin.")
