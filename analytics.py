
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
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
    result = pd.concat([sessions, dau, stickiness], axis=1).reset_index().rename(columns={"index": "date"})
    return result

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

    agg = scope.groupby(["org_id", "org_name", "Industry"]).agg(
        users_30d=("user_id", "nunique"),
        sessions_30d=("user_id", "size"),
        median_session_min=("session_minutes", "median"),
        p90_session_min=("session_minutes", lambda x: np.nanpercentile(x.dropna(), 90) if x.notna().any() else np.nan),
        last_seen=("date", "max"),
    ).reset_index()

    if not agg.empty:
        u = (agg["users_30d"] / agg["users_30d"].max()).fillna(0)
        s = (agg["sessions_30d"] / agg["sessions_30d"].max()).fillna(0)
        d = 1 - ((today - agg["last_seen"]).dt.days.clip(lower=0) / 30).fillna(1)
        agg["health_score"] = ((0.45*u + 0.45*s + 0.10*d) * 100).round(1)
    return agg.sort_values(["health_score", "users_30d", "sessions_30d"], ascending=False)

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
    first_seen = df.groupby("user_id")["date"].min().rename("first_day")
    cohort_month = first_seen.dt.to_period("M").dt.to_timestamp()
    cohort = pd.DataFrame({"user_id": first_seen.index, "first_day": first_seen.values, "cohort_month": cohort_month.values})

    events = df[["user_id", "date"]].merge(cohort, on="user_id", how="left")
    events["day_index"] = (events["date"] - events["first_day"]).dt.days

    cohort_sizes = cohort.groupby("cohort_month")["user_id"].nunique().rename("cohort_size")
    active_by_day = (
        events.groupby(["cohort_month", "day_index"])["user_id"].nunique().rename("active_users")
        .reset_index()
        .merge(cohort_sizes, on="cohort_month", how="left")
    )
    active_by_day["retention"] = (active_by_day["active_users"] / active_by_day["cohort_size"]).round(4)

    ret = active_by_day.pivot_table(index="cohort_month", columns="day_index", values="retention", fill_value=0.0)
    ret = ret.sort_index()
    return ret

def estimate_daily_peak_concurrency(df: pd.DataFrame) -> pd.DataFrame:
    starts = df["logon_date_parsed"].dropna()
    ends = df["logout_date_effective"].dropna()

    events = pd.concat([
        pd.Series(1, index=starts.values),
        pd.Series(-1, index=ends.values),
    ]).sort_index()

    concur = events.cumsum()
    daily_peak = concur.resample("D").max().fillna(0).astype(int).rename("peak_concurrency")
    return daily_peak.reset_index(names=["date"])

# ---------------------------
# Visualization (return Figures and PNG bytes)
# ---------------------------

def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def make_time_series_fig(df_ts: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_ts[x_col], df_ts[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.autofmt_xdate()
    return fig, _fig_to_png_bytes(fig)

def make_hist_fig(series: pd.Series, title: str, xlabel: str, bins: int = 50, xmax_p: float = 0.99):
    s = series.dropna()
    if s.empty:
        return None, None
    clip = s.quantile(xmax_p)
    s = s.clip(upper=clip)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.hist(s, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    return fig, _fig_to_png_bytes(fig)

def make_heatmap_hour_dow_fig(df: pd.DataFrame):
    tbl = df.pivot_table(index="dow", columns="hour", values="user_id", aggfunc="count", fill_value=0)
    tbl = tbl.reindex(index=[0,1,2,3,4,5,6], columns=list(range(0,24)), fill_value=0)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(tbl.values, aspect="auto")
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_xticks(range(0,24,2))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    fig.colorbar(im, ax=ax, label="logins")
    ax.set_title("Logins Heatmap (Hour x DayOfWeek)")
    return fig, _fig_to_png_bytes(fig)

def make_retention_heatmap_fig(ret: pd.DataFrame):
    if ret.empty:
        return None, None
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(ret.values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(ret.index)))
    ax.set_yticklabels([d.strftime("%Y-%m") for d in ret.index])
    max_day = min(60, ret.shape[1]-1) if ret.shape[1] else 0
    ax.set_xticks(range(0, max_day+1, 5))
    ax.set_xlabel("Day since first login")
    ax.set_ylabel("Cohort month")
    fig.colorbar(im, ax=ax, label="retention (0..1)")
    ax.set_title("Cohort Retention (Rows: Cohort Month, Cols: Day Index)")
    return fig, _fig_to_png_bytes(fig)
