import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import analytics as A  # local analytics.py module

def main():
    st.set_page_config(page_title="Login/Activity Analytics", layout="wide")
    st.title("📊 Login / Activity Analytics Dashboard")

    st.sidebar.header("⚙️ Settings")
    tz = st.sidebar.text_input("Timezone", "UTC")
    session_timeout = st.sidebar.number_input("Session timeout (minutes)", 30, 240, 30)
    lookback_days = st.sidebar.number_input("Lookback Days", 7, 90, 30)

    uploaded = st.file_uploader("Upload login/activity CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)

            # normalize
            for col in [
                "user_id","org_id","org_name","browser","browser_version",
                "operating_system","Industry","Domain"
            ]:
                if col not in df.columns:
                    df[col] = pd.NA

            df = A.derive_email_domain(df)
            df = A.add_time_parts(df, tz=tz)
            df = A.compute_session_minutes(df, session_timeout_min=session_timeout)
            df["logout_date_effective"] = A.derive_session_end(df, session_timeout)

            # analytics
            daily = A.daily_metrics(df)
            weekly = A.weekly_metrics(df)
            monthly = A.monthly_metrics(df)
            durations = A.session_duration_stats(df)
            org30 = A.org_activity_last_30d(df)
            browser_share = A.browser_os_share(df)
            retention = A.retention_table(df)

            st.subheader("Daily Metrics")
            st.dataframe(daily)

            st.subheader("Weekly Metrics")
            st.dataframe(weekly)

            st.subheader("Monthly Metrics")
            st.dataframe(monthly)

            st.subheader("Session Duration Stats")
            st.dataframe(durations)

            st.subheader("Org Activity (Last 30d)")
            st.dataframe(org30)

            st.subheader("Browser/OS Share")
            st.dataframe(browser_share)

            st.subheader("Retention Table")
            st.dataframe(retention)

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")


if __name__ == "__main__":
    main()
