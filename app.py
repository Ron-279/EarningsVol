import io
import base64
from datetime import timedelta

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # non-GUI backend for web apps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib import patheffects as pe
import numpy as np

app = Flask(__name__)


def get_earnings_windows(ticker: str, cycles: int, days: int):
    """
    Fetch last N earnings timestamps from Yahoo Finance and return
    a list of dicts containing:
      - 'date' (Timestamp with time & tz from get_earnings_dates index)
      - 'df'   (window DataFrame with OHLC)
      - 'move_pct' (Net move with BMO/AMC logic)
      - 'eps_surprise' (EPS surprise in % if available)
      - 'img_b64'  (matplotlib chart as base64 PNG)
    """

    t = yf.Ticker(ticker)

    # --- 1) Fetch earnings dates (single call) ---
    try:
        ed_df = t.get_earnings_dates(limit=max(cycles, 4))
    except Exception as e:
        msg = str(e)
        if "Too Many Requests" in msg or "429" in msg:
            raise RuntimeError(
                "Yahoo Finance is rate-limiting this app (HTTP 429). "
                "Try again in a few minutes or reduce the number of earnings cycles."
            )
        raise

    if ed_df is None or ed_df.empty:
        raise ValueError(f"No earnings dates found for {ticker}")

    # Make a copy and normalize index to Timestamp (with tz if provided)
    ed_df = ed_df.copy()
    ed_df.index = pd.to_datetime(ed_df.index)

    # Full earnings timestamps as DatetimeIndex
    earnings_ts = ed_df.index

    # ---- Filter out earnings that are today or in the future ----
    # We only want completed earnings, not scheduled-for-today ones.
    tz = earnings_ts.tz
    now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
    today = now.date()

    mask_past = np.array([ts.date() < today for ts in earnings_ts])
    earnings_ts = earnings_ts[mask_past]

    if len(earnings_ts) == 0:
        raise ValueError(f"No past earnings timestamps found for {ticker}")

    # Take the most recent `cycles` timestamps (newest first)
    earnings_ts = earnings_ts.sort_values(ascending=False)[:cycles]

    # Build EPS surprise series (may be None if column missing)
    surprise_series = None
    if "Surprise(%)" in ed_df.columns:
        # Reindex to selected earnings_ts; this keeps alignment
        surprise_series = ed_df.loc[earnings_ts, "Surprise(%)"]

    # Oldest and newest calendar dates for the global history window
    oldest_date = earnings_ts.min().date()
    newest_date = earnings_ts.max().date()

    global_start = pd.Timestamp(oldest_date) - timedelta(days=days)
    global_end = pd.Timestamp(newest_date) + timedelta(days=days)

    # --- 2) Fetch ONE history window that covers ALL earnings windows ---
    try:
        hist = t.history(start=global_start, end=global_end)
    except Exception as e:
        msg = str(e)
        if "Too Many Requests" in msg or "429" in msg:
            raise RuntimeError(
                "Yahoo Finance is rate-limiting this app (HTTP 429). "
                "Try again in a few minutes or reduce the number of earnings cycles."
            )
        raise

    if hist.empty:
        raise ValueError("No price history returned for this period.")

    # Keep full OHLC so we can do candlesticks
    hist_ohlc = hist[["Open", "High", "Low", "Close"]].copy()

    # ---- strip timezone from history index so comparisons work ----
    idx = hist_ohlc.index
    if getattr(idx, "tz", None) is not None:
        hist_ohlc.index = idx.tz_convert(None)

    trading_idx = hist_ohlc.index
    trading_dates = np.array([d.date() for d in trading_idx])

    results = []

    # --- 3) For each earnings timestamp, build window + BMO/AMC net move ---
    for earn_ts in earnings_ts:  # DESC order, so results are [latest ... oldest]
        earn_date = earn_ts.date()  # calendar date
        hour = earn_ts.hour  # used for BMO/AMC

        # Window bounds in calendar days around the earnings *date*
        start_ts = pd.Timestamp(earn_date) - timedelta(days=days)
        end_ts = pd.Timestamp(earn_date) + timedelta(days=days)

        # Window DataFrame for plotting (hist_ohlc index is tz-naive)
        df = hist_ohlc.loc[(hist_ohlc.index >= start_ts) & (hist_ohlc.index <= end_ts)]
        if df.empty:
            continue

        # --- Compute Net move using BMO/AMC definitions ---
        # After close (AMC): (next_day_close - same_day_close)/same_day_close
        # Before open (BMO): (same_day_close - prev_day_close)/prev_day_close
        move_pct = 0.0
        marker_pos = None  # index in trading_idx to mark with yellow ring

        eq_mask = trading_dates == earn_date

        if hour >= 12:
            # AFTER CLOSE (AMC)
            if eq_mask.any():
                base_pos = np.where(eq_mask)[0][-1]
            else:
                le_mask = trading_dates < earn_date
                if not le_mask.any():
                    base_pos = None
                else:
                    base_pos = np.where(le_mask)[0][-1]

            if base_pos is not None and base_pos + 1 < len(trading_idx):
                next_pos = base_pos + 1
                base_close = hist_ohlc.iloc[base_pos]["Close"]
                next_close = hist_ohlc.iloc[next_pos]["Close"]
                move_pct = (next_close - base_close) / base_close * 100.0
                marker_pos = base_pos

        else:
            # BEFORE OPEN (BMO)
            if eq_mask.any():
                event_pos = np.where(eq_mask)[0][0]
            else:
                ge_mask = trading_dates > earn_date
                if not ge_mask.any():
                    event_pos = None
                else:
                    event_pos = np.where(ge_mask)[0][0]

            if event_pos is not None and event_pos - 1 >= 0:
                prev_pos = event_pos - 1
                prev_close = hist_ohlc.iloc[prev_pos]["Close"]
                event_close = hist_ohlc.iloc[event_pos]["Close"]
                move_pct = (event_close - prev_close) / prev_close * 100.0
                marker_pos = event_pos

        # --- EPS surprise for this earnings ---
        eps_surprise = None
        if surprise_series is not None and earn_ts in surprise_series.index:
            val = surprise_series.loc[earn_ts]
            try:
                eps_surprise = float(val)
            except Exception:
                eps_surprise = None

        # ----- Candlestick-style plotting -----
        bg = "#1E1E1E"
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=bg)
        ax.set_facecolor(bg)

        pdf = df.copy().sort_index()
        pdf_reset = pdf.reset_index().rename(columns={"index": "Date"})

        dates = pdf_reset["Date"].to_numpy()
        x = mdates.date2num(dates)
        O = pdf_reset["Open"].to_numpy()
        H = pdf_reset["High"].to_numpy()
        L = pdf_reset["Low"].to_numpy()
        C = pdf_reset["Close"].to_numpy()

        # --- axes styling ---
        ax.grid(True, color="#555", ls="--", alpha=0.4)
        ax.tick_params(colors="white", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#888")

        # --- wicks ---
        ax.vlines(x, L, H, color="white", lw=1, zorder=1)

        # --- candle bodies ---
        if len(x) > 1:
            dx = float(pd.Series(x).diff().median())
            if pd.isna(dx) or dx <= 0:
                dx = 1.0
        else:
            dx = 1.0
        w = 0.6 * dx

        for xi, o, c in zip(x, O, C):
            color = "#4CAF50" if c >= o else "#E53935"
            y = min(o, c)
            h = abs(c - o)
            if h == 0:
                ax.hlines(o, xi - w / 2, xi + w / 2, color=color, lw=2, zorder=3)
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (xi - w / 2, y),
                        w,
                        h,
                        facecolor=color,
                        edgecolor="black",
                        lw=0.8,
                        zorder=3,
                    )
                )

        # --- per-candle % labels (Open -> Close) ---
        pct = (C - O) / np.where(O == 0, np.nan, O) * 100.0
        ytop = np.maximum(O, C)
        for xi, yt, p in zip(x, ytop, pct):
            if np.isfinite(p):
                ax.annotate(
                    f"{p:+.1f}%",
                    (xi, yt),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=9,
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground="black", alpha=0.35)
                    ],
                    annotation_clip=False,
                    zorder=5,
                )

        # --- earnings marker (yellow ring on the trading day we used) ---
        if marker_pos is not None:
            marker_date = trading_idx[marker_pos].date()
            mask_md = pdf_reset["Date"].dt.date == marker_date
            if mask_md.any():
                j = pdf_reset.index[mask_md][0]
                xm = mdates.date2num(pdf_reset.loc[j, "Date"])
                ym = 0.5 * (O[j] + C[j])
                ax.scatter(
                    [xm],
                    [ym],
                    s=180,
                    facecolors="none",
                    edgecolors="#FFD54F",
                    lw=2.2,
                    zorder=6,
                )

        # --- y-limits with headroom/footroom ---
        lo = float(np.nanmin([L.min(), O.min(), C.min()]))
        hi = float(np.nanmax([H.max(), O.max(), C.max()]))
        span = max(1e-9, hi - lo)
        y_headroom = 0.35
        y_footroom = 0.05
        ax.set_ylim(lo - span * y_footroom, hi + span * y_headroom)

        # --- x formatting: ticks centered under candles ---
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
        fig.autofmt_xdate(rotation=30)
        ax.set_ylabel("Price ($)", color="white")

        # --- title: ±days, Net move %, and EPS surprise if available ---
        gap_str = f" | Net move: {move_pct:+.1f}%"
        ttl_suffix = f" (±{days} calendar days)"

        surprise_str = ""
        if eps_surprise is not None and np.isfinite(eps_surprise):
            surprise_str = f" | EPS surprise: {eps_surprise:+.1f}%"

        ax.set_title(
            f"Earnings {earn_date}{ttl_suffix}{gap_str}{surprise_str}",
            color="white",
            fontsize=12,
            pad=12,
        )

        fig.suptitle(
            f"{ticker.upper()} — Earnings Cycle",
            color="white",
            y=0.99,
            fontsize=14,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")

        results.append(
            {
                "date": earn_ts,  # full timestamp (with time & tz)
                "df": df,
                "move_pct": move_pct,  # BMO/AMC-aware net move
                "eps_surprise": eps_surprise,
                "img_b64": img_b64,
            }
        )

    if not results:
        raise ValueError("No usable data windows found around earnings dates.")

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    default_params = {"ticker": "AAPL", "cycles": 4, "days": 10}

    if request.method == "GET":
        return render_template(
            "index.html",
            params=default_params,
            cycles_data=None,
            summary=None,
            error=None,
        )

    ticker = request.form.get("ticker", "AAPL").upper().strip()
    cycles = int(request.form.get("cycles", 4))
    days = int(request.form.get("days", 10))

    try:
        cycles_data = get_earnings_windows(ticker, cycles, days)

        moves = [c["move_pct"] for c in cycles_data]
        avg_abs_move = sum(abs(m) for m in moves) / len(moves)
        positive = sum(1 for m in moves if m > 0)

        summary = {
            "ticker": ticker,
            "cycles": len(cycles_data),
            "days": days,
            "avg_abs_move": avg_abs_move,
            "positive": positive,
            "total": len(moves),
        }

        return render_template(
            "index.html",
            params={"ticker": ticker, "cycles": cycles, "days": days},
            cycles_data=cycles_data,
            summary=summary,
            error=None,
        )

    except Exception as e:
        msg = str(e)
        return render_template(
            "index.html",
            params={"ticker": ticker, "cycles": cycles, "days": days},
            cycles_data=None,
            summary=None,
            error=msg,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
