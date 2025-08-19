from __future__ import annotations
from typing import Optional, Tuple
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------

def _parse_dt(series: pd.Series, tz: Optional[str]) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if tz:
        try:
            s = s.dt.tz_convert(tz)
        except Exception:
            pass
    return s

def _safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns and df[name].notna().any()

def derive_email_domain(df: pd.DataFrame) -> pd.DataFrame:
    if "email_address" in df.columns:
        df["email_domain"] = (
            df["email_address"].astype(str).str.extract(r"@(.+)$")[0].str.lower()
        )
    else:
        df["email_domain"] = np.nan
    return df

def derive_session_end(df: pd.DataFrame, session_timeout_min: Optional[int]) -> pd.Series:
    if "logout_date" in df.columns:
        logout = df["logout_date_parsed"]
    else:
        logout = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    if session_timeout_min is not None:
        need_fill = logout.isna() & df["logon_date_parsed"].notna()
        logout.loc[need_fill] = df.loc[need_fill, "logon_date_parsed"] + pd.to_timedelta(session_timeout_min, unit="m")
    return logout

def _quantiles(series: pd.Series, qs=(0.5, 0.9, 0.99)) -> dict:
    q = series.dropna().quantile(qs)
    return {f"p{int(p*100)}": q.loc[p] for p in qs}

def _week_floor(d: pd.Series) -> pd.Series:
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()

def add_time_parts(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    df["logon_date_parsed"] = _parse_dt(df.get("logon_date"), tz)
    df["logout_date_parsed"] = _parse_dt(df.get("logout_date"), tz)
    if _safe_col(df, "Date") and df["logon_date_parsed"].isna().all():
        df["logon_date_parsed"] = _parse_dt(df["Date"], tz)

    dt = df["logon_date_parsed"]
    df["date"] = pd.to_datetime(dt.dt.date)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.to_period("M").dt.to_timestamp()

    if "Time of Day" in df.columns and df["Time of Day"].notna().any():
        df["time_of_day_raw"] = df["Time of Day"].astype(str)
    return df

# ---------------------------
# Analytics
# ---------------------------

def compute_session_minutes(df: pd.DataFrame, session_timeout_min: Optional[int]) -> pd.DataFrame:
    df = df.copy()
    end = derive_session_end(df, session_timeout_min)
    start = df["logon_date_parsed"]
    mins = (end - start).dt.total_seconds() / 60.0
    df["session_minutes"] = mins.where(mins > 0, np.nan)
    return df

def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    sessions = df.groupby("date").size().rename("sessions")
    dau = df.groupby("date")["user_id"].nunique().rename("dau")
    mau_monthly = df.groupby("month")["user_id"].nunique().rename("mau")
    mau_for_day = pd.Series(index=dau.index, dtype=float)
    for d in dau.index:
        m = pd.Timestamp(d).to_period("M").to_timestamp()
        mau_for_day.loc[d] = float(mau_monthly.get(m, np.nan))
    stickiness = (dau / mau_for_day).rename("stickiness_dau_over_mau")
    return pd.concat([sessions, dau, stickiness], axis=1).reset_index()

def weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = _week_floor(df["date"])
    wau = df.groupby("week")["user_id"].nunique().rename("wau")
    sessions = df.groupby("week").size().rename("sessions")
    return pd.concat([wau, sessions], axis=1).reset_index()

def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    mau = df.groupby("month")["user_id"].nunique().rename("mau")
    sessions = df.groupby("month").size().rename("sessions")
    return pd.concat([mau, sessions], axis=1).reset_index()

def session_duration_stats(df: pd.DataFrame) -> pd.DataFrame:
    s = df["session_minutes"]
    stats = {
        "count_non_null": int(s.notna().sum()),
        "count_null": int(s.isna().sum()),
        "mean_minutes": float(np.nanmean(s)),
        **_quantiles(s, (0.5, 0.9, 0.99)),
    }
    return pd.DataFrame([stats])

def org_activity_last_30d(df: pd.DataFrame, today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize() if today is None else today.normalize()
    start = today - pd.Timedelta(days=30)
    mask = (df["date"] >= start) & (df["date"] <= today)
    scope = df.loc[mask].copy()

    for col in ["org_id", "org_name", "Industry"]:
        if col not in scope.columns:
            scope[col] = np.nan

    if scope.empty:
        return pd.DataFrame(
            columns=[
                "org_id", "org_name", "Industry",
                "users_30d", "sessions_30d",
                "median_session_min", "p90_session_min",
                "last_seen", "health_score",
            ]
        )

    agg = scope.groupby(["org_id", "org_name", "Industry"]).agg(
        users_30d=("user_id", "nunique"),
        sessions_30d=("user_id", "size"),
        median_session_min=("session_minutes", "median"),
        p90_session_min=("session_minutes",
                         lambda x: np.nanpercentile(x.dropna(), 90) if x.notna().any() else np.nan),
        last_seen=("date", "max"),
    ).reset_index()

    u = (agg["users_30d"] / agg["users_30d"].max()).fillna(0)
    s = (agg["sessions_30d"] / agg["sessions_30d"].max()).fillna(0)
    d = 1 - ((today - agg["last_seen"]).dt.days.clip(lower=0) / 30).fillna(1)
    agg["health_score"] = ((0.45*u + 0.45*s + 0.10*d) * 100).round(1)

    sort_cols = [c for c in ["health_score", "users_30d", "sessions_30d"] if c in agg.columns]
    return agg.sort_values(sort_cols, ascending=False) if sort_cols else agg

def browser_os_share(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["browser", "browser_version", "operating_system"]:
        if col not in df.columns:
            df[col] = np.nan
    out = (
        df.groupby(["browser", "browser_version", "operating_system"])
          .size()
          .rename("sessions")
          .reset_index()
          .sort_values("sessions", ascending=False)
    )
    return out

def retention_table(df: pd.DataFrame) -> pd.DataFrame:
    # Need these columns
    if "user_id" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()

    # Keep only rows with both user_id and date
    tmp = df[["user_id", "date"]].dropna()
    if tmp.empty:
        return pd.DataFrame()

    # First day per user = cohort start
    first_seen = tmp.groupby("user_id")["date"].min()
    cohort_month = first_seen.dt.to_period("M").dt.to_timestamp()
    cohort = pd.DataFrame({
        "user_id": first_seen.index,
        "first_day": first_seen.values,
        "cohort_month": cohort_month.values
    })

    # User events joined with cohort
    events = tmp.merge(cohort, on="user_id", how="inner")
    events["day_index"] = (events["date"] - events["first_day"]).dt.days
    # Keep non-negative day indexes
    events = events[events["day_index"] >= 0]

    if events.empty:
        return pd.DataFrame()

    cohort_sizes = cohort.groupby("cohort_month")["user_id"].nunique().rename("cohort_size")
    active_by_day = (
        events.groupby(["cohort_month", "day_index"])["user_id"].nunique()
              .rename("active_users")
              .reset_index()
              .merge(cohort_sizes, on="cohort_month", how="left")
    )
    active_by_day["retention"] = (
        active_by_day["active_users"] / active_by_day["cohort_size"]
    ).fillna(0).round(4)

    if active_by_day.empty:
        return pd.DataFrame()

    ret = active_by_day.pivot_table(
        index="cohort_month", columns="day_index", values="retention", fill_value=0.0
    )
    ret = ret.sort_index()
    return ret

