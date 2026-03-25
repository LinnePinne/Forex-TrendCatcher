import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "USDJPY",
        "csv": "USDJPY_1H_2003(05)-2026.csv",
        "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.12,
            "fixed_spread_pips": 0.45,
            "comm_pips_per_side": 0.25,
        },
    },
    {
        "name": "EURJPY",
        "csv": "EURJPY_1H_2003(05)-2026.csv",
        "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.35,
            "fixed_spread_pips": 0.90,
            "comm_pips_per_side": 0.25,
        },
    },
    {
        "name": "GBPJPY",
        "csv": "GBPJPY_1H_2003(05)-2026.csv",
        "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.50,
            "fixed_spread_pips": 1.20,
            "comm_pips_per_side": 0.25,
        },
    },
]

HALF = 0.5


# ==========================
# HJÄLPFUNKTIONER
# ==========================

def clamp_time_series_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ unik, sorterad datetime-index."""
    df = df.sort_index()
    if df.index.has_duplicates:
        agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg_map["volume"] = "sum"
        elif "tick_volume" in df.columns:
            agg_map["tick_volume"] = "sum"

        df = df.groupby(df.index).agg(agg_map).sort_index()
    return df


def annualize_factor_from_resample(rule: str) -> float:
    """Ann-faktor för returns baserat på resample-regel."""
    rule = rule.upper()
    if rule in ("D", "1D"):
        return 365.0
    if rule in ("B", "1B"):
        return 252.0
    return 365.0


def compute_stats_from_trades(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    df_ = trades_df.copy()
    df_["equity"] = df_["pnl"].cumsum()
    df_["is_win"] = df_["pnl"] > 0

    gross_profit = df_.loc[df_["pnl"] > 0, "pnl"].sum()
    gross_loss = df_.loc[df_["pnl"] < 0, "pnl"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = df_.loc[df_["pnl"] > 0, "pnl"].mean()
    avg_loss = df_.loc[df_["pnl"] < 0, "pnl"].mean()

    winrate = df_["is_win"].mean()
    expectancy = df_["pnl"].mean()

    roll_max = df_["equity"].cummax()
    dd = df_["equity"] - roll_max
    max_dd_points = float(abs(dd.min())) if len(dd) > 0 else 0.0

    loss_streak = 0
    max_loss_streak = 0
    for is_win in df_["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    pnl_std = df_["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(df_)) if pnl_std and pnl_std > 0 else np.nan

    return {
        "Trades": int(len(df_)),
        "Total PnL (points)": float(df_["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }


def is_usd_quote(symbol: str) -> bool:
    # XXXUSD
    return symbol.endswith("USD") and not symbol.startswith("USD")


def is_usd_base(symbol: str) -> bool:
    # USDXXX
    return symbol.startswith("USD") and not symbol.endswith("USD")


def is_jpy_cross(symbol: str) -> bool:
    # XXXJPY där XXX != USD
    return symbol.endswith("JPY") and not symbol.startswith("USD")


def get_usdjpy_price_at_time(price_lookup: dict, t: pd.Timestamp) -> float:
    """
    Returnerar USDJPY-pris vid tid t för konvertering JPY -> USD.
    Kräver att USDJPY finns i price_lookup.
    """
    s = price_lookup.get("USDJPY")
    if s is None:
        raise ValueError("USDJPY krävs i price lookup för att konvertera JPY-kors till USD.")
    val = s.asof(t)
    if pd.isna(val):
        raise ValueError(f"Kunde inte hitta USDJPY-pris vid {t}")
    return float(val)


def price_in_usd_per_base_unit(symbol: str, pair_price: float, usdjpy_price: float | None = None) -> float:
    """
    Omvandlar pair price till USD per 1 unit base currency.
    Exempel:
    - EURUSD: price redan i USD
    - USDJPY: 1 USD kostar price JPY => USD per base = 1
    - EURJPY: EURJPY / USDJPY = USD per 1 EUR
    """
    if is_usd_quote(symbol):
        return float(pair_price)

    if is_usd_base(symbol):
        return 1.0

    if is_jpy_cross(symbol):
        if usdjpy_price is None:
            raise ValueError(f"USDJPY krävs för att prissätta {symbol} i USD")
        return float(pair_price) / float(usdjpy_price)

    raise NotImplementedError(f"Valutakonvertering ej implementerad för {symbol}")


def pip_value_usd_per_unit(symbol: str, pip_size: float, pair_price: float, usdjpy_price: float | None = None) -> float:
    """
    USD-värde av 1 pip för 1 unit base.
    """
    if is_usd_quote(symbol):
        return float(pip_size)

    if is_usd_base(symbol):
        return float(pip_size) / float(pair_price)

    if is_jpy_cross(symbol):
        if usdjpy_price is None:
            raise ValueError(f"USDJPY krävs för pip value i {symbol}")
        return float(pip_size) / float(usdjpy_price)

    raise NotImplementedError(f"Pip value ej implementerad för {symbol}")


def approx_usd_per_pip(sym: str, pip_size: float, units: float, price: float, usdjpy_price: float | None = None) -> float:
    pip_value = pip_value_usd_per_unit(
        symbol=sym,
        pip_size=pip_size,
        pair_price=price,
        usdjpy_price=usdjpy_price,
    )
    return units * pip_value


# ==========================
# EN-MARKNADS-BACKTEST
# ==========================

def run_backtest_for_market(
    market_name: str,
    csv_path: str,
    pip_size: float,
    spread_points_per_pip: float = 10.0,
    cost_model: dict | None = None,
):
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = clamp_time_series_index_unique(df)

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    if cost_model is None:
        cost_model = {}

    slippage_pips = float(cost_model.get("slippage_pips", 0.0))
    fixed_spread_pips = float(cost_model.get("fixed_spread_pips", 0.0))
    comm_pips_per_side = float(cost_model.get("comm_pips_per_side", 0.0))

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(comm_pips_per_side)

    # volymkolumn
    if "volume" in df.columns:
        vol_col = "volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["volume_dummy"] = 1.0
        vol_col = "volume_dummy"

    def asia_range(df_in: pd.DataFrame) -> pd.DataFrame:
        data = df_in.copy()

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame måste ha DateTimeIndex")

        session = data.between_time("00:00", "07:00")

        daily_range = session.groupby(session.index.date).agg(
            asia_high=("high", "max"),
            asia_low=("low", "min")
        )

        data["date"] = data.index.date
        data = data.merge(
            daily_range,
            left_on="date",
            right_index=True,
            how="left"
        )

        data["asia_range"] = data["asia_high"] - data["asia_low"]
        data["asia_mid"] = (data["asia_high"] + data["asia_low"]) / 2

        data.loc[data.index.hour < 8, ["asia_high", "asia_low", "asia_range", "asia_mid"]] = np.nan
        data.drop(columns="date", inplace=True)

        return data

    df = asia_range(df)

    def pct_rank_last(x):
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1]

    asia_valid = df[df["asia_range"].notna()].copy()

    daily_asia = (
        asia_valid
        .groupby(asia_valid.index.date)
        .first()[["asia_range"]]
        .copy()
    )

    daily_asia.index = pd.to_datetime(daily_asia.index)

    daily_asia["asia_range_pct_rank"] = (
        daily_asia["asia_range"]
        .rolling(window=252, min_periods=50)
        .apply(pct_rank_last, raw=False)
    )

    df["date_only"] = pd.to_datetime(df.index.date)

    df = df.merge(
        daily_asia[["asia_range_pct_rank"]],
        left_on="date_only",
        right_index=True,
        how="left"
    )

    df.drop(columns="date_only", inplace=True)

    def ATR(df_in: pd.DataFrame, period: int = 14, method: str = "wilder") -> pd.Series:
        required_cols_local = {"high", "low", "close"}
        if not required_cols_local.issubset(df_in.columns):
            raise ValueError(f"DataFrame måste innehålla kolumnerna {required_cols_local}")

        high = df_in["high"]
        low = df_in["low"]
        close = df_in["close"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        if method.lower() == "sma":
            atr = tr.rolling(window=period, min_periods=period).mean()
        elif method.lower() == "wilder":
            atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        else:
            raise ValueError("method måste vara 'wilder' eller 'sma'")

        return atr

    df["atr"] = ATR(df, period=14, method="wilder")
    df["atr_ma"] = df["atr"].rolling(75).mean()

    # session (svensk tid)
    session_start = "08:00:00"
    session_end = "09:00:00"

    session_start_t = pd.to_datetime(session_start).time()
    session_end_t = pd.to_datetime(session_end).time()

    def in_session(ts) -> bool:
        t = ts.time()
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)
        return (t >= session_start_t) or (t < session_end_t)

    USE_SPREAD_PIPS_COL = "spread_pips" in df.columns
    USE_SPREAD_POINTS_COL = "spread_points" in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row["spread_pips"])
        if USE_SPREAD_POINTS_COL:
            return float(row["spread_points"]) / float(spread_points_per_pip)
        return float(fixed_spread_pips)

    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None
    entry_mid = None
    entry_asia_range_pct_rank = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # EXIT
        if in_position:
            exit_price = None
            exit_reason = None

            atr_now = row["atr"]
            atr_ma_now = row["atr_ma"]

            if pos_direction == "LONG":
                if np.isfinite(atr_now) and np.isfinite(atr_ma_now) and atr_now < atr_ma_now:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)

                    exit_price = next_row["open"] - HALF * spread_px - slip_px
                    exit_reason = "atr_exit"

            else:  # SHORT
                if np.isfinite(atr_now) and np.isfinite(atr_ma_now) and atr_now < atr_ma_now:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)

                    exit_price = next_row["open"] + HALF * spread_px + slip_px
                    exit_reason = "atr_exit"

            if exit_price is not None:
                exit_time = idx_list[i + 1]
                comm_px = commission_round_turn_price()

                if pos_direction == "LONG":
                    pnl = (exit_price - entry_price) - comm_px
                else:
                    pnl = (entry_price - exit_price) - comm_px

                exit_mid = float(next_row["open"])

                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": pos_direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Entry Mid": entry_mid,
                    "Exit Mid": exit_mid,
                    "pnl": pnl,
                    "asia_range_pct_rank": entry_asia_range_pct_rank,
                    "Exit Reason": exit_reason,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                entry_mid = None
                entry_asia_range_pct_rank = None
                continue

        # sessionfilter
        if not in_session(ts):
            continue

        if in_position:
            continue

        close_price = row["close"]
        next_open = next_row["open"]
        asia_low = row["asia_low"]
        asia_high = row["asia_high"]
        atr_now = row["atr"]
        atr_ma_now = row["atr_ma"]
        asia_range_pct_rank = row["asia_range_pct_rank"]

        if not np.isfinite(asia_high) or not np.isfinite(asia_low):
            continue

        if not np.isfinite(atr_now) or not np.isfinite(atr_ma_now):
            continue

        if not np.isfinite(asia_range_pct_rank):
            continue

        atr_filter = atr_now > atr_ma_now

        bullish_breakout = close_price > asia_high
        bearish_breakout = close_price < asia_low

        long_signal = bullish_breakout and atr_filter
        short_signal = bearish_breakout and atr_filter

        if market_name == "GBPJPY":
            long_entry = long_signal and asia_range_pct_rank > 0.75
            short_entry = short_signal and asia_range_pct_rank < 0.65
        else:  # USDJPY och EURJPY

            long_entry = long_signal and asia_range_pct_rank > 0.7
            short_entry = short_signal and asia_range_pct_rank > 0.7

        if long_entry:
            pos_direction = "LONG"
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            entry_asia_range_pct_rank = row["asia_range_pct_rank"]
            entry_mid = float(next_row["open"])
            entry_price = entry_mid + HALF * spread_px + slip_px
            in_position = True

        elif short_entry:
            pos_direction = "SHORT"
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            entry_asia_range_pct_rank = row["asia_range_pct_rank"]
            entry_mid = float(next_row["open"])
            entry_price = entry_mid - HALF * spread_px - slip_px
            in_position = True

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()
    winrate = trades_df["is_win"].mean()
    expectancy = trades_df["pnl"].mean()

    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()
    max_dd_points = abs(max_dd)

    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL (points)": float(trades_df["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }

    print("\n--- STATS ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["equity"])
    plt.title(f"Equity curve - {market_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats, trades_df


# ==========================
# PORTFÖLJSIMULERING
# ==========================

def build_price_series(market_cfg: dict, start_time=None, end_time=None, price_col="close"):
    """
    Returnerar dict: symbol -> pd.Series (price_col) med datetimeindex.
    """
    out = {}
    for sym, cfg in market_cfg.items():
        df = pd.read_csv(cfg["csv"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            raise ValueError(f"{sym}: saknar timestamp/datetime")

        df = clamp_time_series_index_unique(df)
        s = df[price_col].astype(float)

        if start_time is not None:
            s = s[s.index >= start_time]
        if end_time is not None:
            s = s[s.index <= end_time]

        out[sym] = s
    return out


def usd_pnl_from_price_delta(
    sym: str,
    pip_size: float,
    price_now: float,
    pnl_price_per_unit: float,
    units: float,
    usdjpy_now: float | None = None,
) -> float:
    pip_value = pip_value_usd_per_unit(
        symbol=sym,
        pip_size=pip_size,
        pair_price=price_now,
        usdjpy_price=usdjpy_now,
    )
    pips_move = pnl_price_per_unit / pip_size
    return pips_move * pip_value * units


def simulate_portfolio_equal_risk(
    trades: pd.DataFrame,
    market_cfg: dict,
    start_capital: float = 50_000.0,
    max_one_position_per_market: bool = True,
    exposure_scale: float = 1.0,
    weights: dict | None = None,
):
    """
    Equal risk per market:
    varje marknad får lika stor notional-budget vid entry.
    """

    df = trades.copy()
    df = df.sort_values("Entry Time").reset_index(drop=True)

    markets_list = sorted(df["Market"].unique().tolist())
    n_markets = len(markets_list)
    if n_markets == 0:
        return None, None, None

    if weights is None:
        market_budget = {m: 1.0 / n_markets for m in markets_list}
    else:
        market_budget = {m: float(weights.get(m, 0.0)) for m in markets_list}
        s = sum(market_budget.values())
        if s <= 0:
            market_budget = {m: 1.0 / n_markets for m in markets_list}
        else:
            market_budget = {m: market_budget[m] / s for m in markets_list}

    equity = start_capital
    equity_curve = []
    open_pos = {m: None for m in markets_list}

    exit_events = df[["Exit Time", "Market"]].copy()
    exit_events["idx"] = exit_events.index
    exit_events = exit_events.sort_values("Exit Time").reset_index(drop=True)
    exit_ptr = 0

    price_lookup = build_price_series(
        market_cfg,
        start_time=df["Entry Time"].min(),
        end_time=df["Exit Time"].max(),
        price_col="close",
    )

    def usd_pnl_for_trade(row, units: float) -> float:
        sym = row["Market"]
        pip_size = float(market_cfg[sym]["pip_size"])
        exit_price = float(row["Exit Price"])
        exit_time = pd.Timestamp(row["Exit Time"])

        usdjpy_price = None
        if is_jpy_cross(sym):
            usdjpy_price = get_usdjpy_price_at_time(price_lookup, exit_time)

        pip_value = pip_value_usd_per_unit(
            symbol=sym,
            pip_size=pip_size,
            pair_price=exit_price,
            usdjpy_price=usdjpy_price,
        )

        pnl_price = float(row["pnl"])
        pips_move = pnl_price / pip_size
        return pips_move * pip_value * units

    trade_log = []

    for i, row in df.iterrows():
        t_entry = pd.Timestamp(row["Entry Time"])

        # stäng positioner vars exit inträffar före/vid denna entry
        while exit_ptr < len(exit_events) and pd.Timestamp(exit_events.loc[exit_ptr, "Exit Time"]) <= t_entry:
            idx_to_close = int(exit_events.loc[exit_ptr, "idx"])
            r_close = df.loc[idx_to_close]
            mkt = r_close["Market"]

            pos = open_pos.get(mkt)
            if pos is not None and pos.get("trade_idx") == idx_to_close:
                units = pos["units"]
                pnl_usd = usd_pnl_for_trade(r_close, units)
                equity += pnl_usd

                trade_log.append({
                    "Market": mkt,
                    "Entry Time": r_close["Entry Time"],
                    "Exit Time": r_close["Exit Time"],
                    "Direction": r_close.get("Direction", None),
                    "Entry Price": r_close.get("Entry Price", np.nan),
                    "Exit Price": r_close.get("Exit Price", np.nan),
                    "Entry Mid": r_close.get("Entry Mid", np.nan),
                    "Exit Mid": r_close.get("Exit Mid", np.nan),
                    "Units": units,
                    "PnL_USD": pnl_usd,
                    "Equity": equity,
                })

                open_pos[mkt] = None
                equity_curve.append({"Time": r_close["Exit Time"], "Equity": equity})

            exit_ptr += 1

        mkt = row["Market"]
        if max_one_position_per_market and open_pos.get(mkt) is not None:
            continue

        notional_usd = equity * market_budget[mkt] * exposure_scale
        entry_price = float(row["Entry Price"])

        usdjpy_price = None
        if is_jpy_cross(mkt):
            usdjpy_price = get_usdjpy_price_at_time(price_lookup, t_entry)

        price_usd = price_in_usd_per_base_unit(
            symbol=mkt,
            pair_price=entry_price,
            usdjpy_price=usdjpy_price,
        )

        units = notional_usd / price_usd

        open_pos[mkt] = {"units": units, "trade_idx": i}
        equity_curve.append({"Time": t_entry, "Equity": equity})

    while exit_ptr < len(exit_events):
        idx_to_close = int(exit_events.loc[exit_ptr, "idx"])
        r_close = df.loc[idx_to_close]
        mkt = r_close["Market"]

        pos = open_pos.get(mkt)
        if pos is not None and pos.get("trade_idx") == idx_to_close:
            units = pos["units"]
            pnl_usd = usd_pnl_for_trade(r_close, units)
            equity += pnl_usd

            trade_log.append({
                "Market": mkt,
                "Entry Time": r_close["Entry Time"],
                "Exit Time": r_close["Exit Time"],
                "Direction": r_close.get("Direction", None),
                "Entry Price": r_close.get("Entry Price", np.nan),
                "Exit Price": r_close.get("Exit Price", np.nan),
                "Units": units,
                "PnL_USD": pnl_usd,
                "Equity": equity,
                "Entry Mid": r_close.get("Entry Mid", np.nan),
                "Exit Mid": r_close.get("Exit Mid", np.nan),
            })

            open_pos[mkt] = None
            equity_curve.append({"Time": r_close["Exit Time"], "Equity": equity})

        exit_ptr += 1

    eq_df = pd.DataFrame(equity_curve).drop_duplicates(subset=["Time"]).sort_values("Time")
    log_df = pd.DataFrame(trade_log).sort_values("Exit Time")

    if not eq_df.empty:
        eq_df["RollMax"] = eq_df["Equity"].cummax()
        eq_df["DD_$"] = eq_df["Equity"] - eq_df["RollMax"]
        eq_df["DD_%"] = eq_df["DD_$"] / eq_df["RollMax"]
        max_dd_usd = float(eq_df["DD_$"].min())
        max_dd_pct = float(eq_df["DD_%"].min())
    else:
        max_dd_usd = 0.0
        max_dd_pct = 0.0

    days = 0.0
    years = 0.0
    sharpe = np.nan
    sortino = np.nan
    calmar = np.nan
    cagr = np.nan

    if not eq_df.empty and len(eq_df) >= 2:
        eq_ts = eq_df.copy()
        eq_ts["Time"] = pd.to_datetime(eq_ts["Time"])
        eq_ts = eq_ts.sort_values("Time").set_index("Time")

        daily_eq = eq_ts["Equity"].resample("D").last().ffill()
        daily_ret = daily_eq.pct_change().dropna()

        if len(daily_ret) >= 30:
            ann_factor = 365.0

            ret_mean = daily_ret.mean()
            ret_std = daily_ret.std(ddof=1)

            sharpe = (ret_mean / ret_std) * np.sqrt(ann_factor) if ret_std and ret_std > 0 else np.nan

            mar = 0.0
            downside_sq = np.minimum(0.0, daily_ret - mar) ** 2
            downside_dev = np.sqrt(downside_sq.mean())

            sortino = (ret_mean / downside_dev) * np.sqrt(ann_factor) if downside_dev and downside_dev > 0 else np.nan

            start_val = float(daily_eq.iloc[0])
            end_val = float(daily_eq.iloc[-1])
            days = (daily_eq.index[-1] - daily_eq.index[0]).total_seconds() / 86400.0

            if days > 0 and start_val > 0:
                cagr = (end_val / start_val) ** (ann_factor / days) - 1.0
                years = days / 365.0 if days > 0 else 0.0

            max_dd_frac = abs(max_dd_pct)
            calmar = (cagr / max_dd_frac) if (np.isfinite(cagr) and max_dd_frac and max_dd_frac > 0) else np.nan

    summary = {
        "Start Capital": start_capital,
        "End Equity": float(equity),
        "Net PnL ($)": float(equity - start_capital),
        "Return (%)": float((equity / start_capital - 1.0) * 100.0),
        "Max Drawdown ($)": abs(max_dd_usd),
        "Max Drawdown (%)": abs(max_dd_pct) * 100.0,
        "Markets": n_markets,
        "Trades Closed": int(len(log_df)),
        "Days": float(days),
        "Years": float(years),
        "Sharpe (daily, ann.)": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Sortino (daily, ann.)": float(sortino) if np.isfinite(sortino) else np.nan,
        "CAGR (%)": float(cagr * 100.0) if np.isfinite(cagr) else np.nan,
        "Calmar": float(calmar) if np.isfinite(calmar) else np.nan,
    }

    return summary, eq_df, log_df


def compute_portfolio_mtm_equity_and_intraday_dd(
    port_log: pd.DataFrame,
    market_cfg: dict,
    start_capital: float = 50_000.0,
    freq: str = "5min",
    price_col: str = "close",
):
    """
    Bygger MTM equitykurva inklusive orealiserad PnL och räknar intraday DD.
    port_log måste ha: Market, Entry Time, Exit Time, Units, Direction, Entry Price, Entry Mid
    """

    if port_log.empty:
        return None, None

    log = port_log.copy()
    log["Entry Time"] = pd.to_datetime(log["Entry Time"])
    log["Exit Time"] = pd.to_datetime(log["Exit Time"])

    t0 = log["Entry Time"].min()
    t1 = log["Exit Time"].max()

    prices = build_price_series(market_cfg, start_time=t0, end_time=t1, price_col=price_col)

    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("MTM: t0/t1 är NaT. Kontrollera att port_log har giltiga Entry/Exit Time.")

    start = pd.Timestamp(t0).floor(freq)
    end = pd.Timestamp(t1).ceil(freq)

    if end < start:
        raise ValueError(f"MTM: end < start ({end} < {start}). Kontrollera tiderna i port_log.")

    master_index = pd.date_range(start=start, end=end, freq=freq)
    if len(master_index) == 0:
        raise ValueError("MTM: master_index blev tom. Kontrollera freq och tidsintervall.")

    price_df = pd.DataFrame(index=master_index)
    for sym, s in prices.items():
        aligned = s.reindex(master_index).ffill()
        price_df[sym] = aligned

    opens = log.sort_values("Entry Time").reset_index(drop=True)
    closes = log.sort_values("Exit Time").reset_index(drop=True)

    o_ptr = 0
    c_ptr = 0

    open_pos = {}
    cash_equity = start_capital
    mtm_records = []

    for t in master_index:
        # close events
        while c_ptr < len(closes) and closes.loc[c_ptr, "Exit Time"] <= t:
            r = closes.loc[c_ptr]
            sym = r["Market"]

            if "PnL_USD" in r:
                cash_equity += float(r["PnL_USD"])
            else:
                raise ValueError("port_log saknar PnL_USD. Lägg till det i simulate_portfolio_equal_risk.")

            if sym in open_pos and len(open_pos[sym]) > 0:
                open_pos[sym].pop(0)
                if len(open_pos[sym]) == 0:
                    del open_pos[sym]

            c_ptr += 1

        # open events
        while o_ptr < len(opens) and opens.loc[o_ptr, "Entry Time"] <= t:
            r = opens.loc[o_ptr]
            sym = r["Market"]
            cm = market_cfg[sym].get("cost_model", {})
            pos = {
                "Direction": r.get("Direction", None),
                "Units": float(r["Units"]),
                "EntryPrice": float(r["Entry Price"]) if "Entry Price" in r else np.nan,
                "EntryMid": float(r["Entry Mid"]) if "Entry Mid" in r else np.nan,
                "SpreadPips": float(cm.get("fixed_spread_pips", 0.0)),
                "SlipPips": float(cm.get("slippage_pips", 0.0)),
            }
            open_pos.setdefault(sym, []).append(pos)
            o_ptr += 1

        # MTM orealiserad PnL
        unreal = 0.0
        for sym, plist in open_pos.items():
            mid_now = float(price_df.loc[t, sym])
            pip_size = float(market_cfg[sym]["pip_size"])

            usdjpy_now = None
            if is_jpy_cross(sym):
                usdjpy_now = float(price_df.loc[t, "USDJPY"])

            for pos in plist:
                units = float(pos["Units"])
                direction = pos.get("Direction", None)

                entry_exec = float(pos.get("EntryPrice", np.nan))
                if not np.isfinite(entry_exec) or not np.isfinite(mid_now):
                    continue

                spread_pips = float(pos.get("SpreadPips", 0.0))
                slip_pips = float(pos.get("SlipPips", 0.0))

                spread_px = spread_pips * pip_size
                slip_px = slip_pips * pip_size

                if direction == "LONG":
                    liquidation = mid_now - 0.5 * spread_px - slip_px
                    pnl_price_per_unit = liquidation - entry_exec
                else:
                    liquidation = mid_now + 0.5 * spread_px + slip_px
                    pnl_price_per_unit = entry_exec - liquidation

                unreal += usd_pnl_from_price_delta(
                    sym=sym,
                    pip_size=pip_size,
                    price_now=mid_now,
                    pnl_price_per_unit=pnl_price_per_unit,
                    units=units,
                    usdjpy_now=usdjpy_now,
                )

        mtm_equity = cash_equity + unreal
        mtm_records.append({
            "Time": t,
            "Equity_MTM": mtm_equity,
            "Cash": cash_equity,
            "Unreal": unreal
        })

    mtm_df = pd.DataFrame(mtm_records).set_index("Time")

    mtm_df["RollMax"] = mtm_df["Equity_MTM"].cummax()
    mtm_df["DD_$"] = mtm_df["Equity_MTM"] - mtm_df["RollMax"]
    mtm_df["DD_%"] = mtm_df["DD_$"] / mtm_df["RollMax"]

    max_dd_usd = float(mtm_df["DD_$"].min())
    max_dd_pct = float(mtm_df["DD_%"].min())

    g = mtm_df.groupby(mtm_df.index.date)
    daily_peak = g["Equity_MTM"].cummax()
    intraday_dd = mtm_df["Equity_MTM"] - daily_peak
    mtm_df["Intraday_DD_$"] = intraday_dd
    mtm_df["Intraday_DD_%"] = intraday_dd / daily_peak

    max_intraday_dd_usd = float(mtm_df["Intraday_DD_$"].min())
    max_intraday_dd_pct = float(mtm_df["Intraday_DD_%"].min())

    dd_summary = {
        "Max DD MTM ($)": abs(max_dd_usd),
        "Max DD MTM (%)": abs(max_dd_pct) * 100.0,
        "Max Intraday DD MTM ($)": abs(max_intraday_dd_usd),
        "Max Intraday DD MTM (%)": abs(max_intraday_dd_pct) * 100.0,
    }

    return dd_summary, mtm_df


def compute_risk_metrics_from_equity(eq_series: pd.Series, resample_rule: str = "D") -> dict:
    out = {
        "Sharpe (daily, ann.)": np.nan,
        "Sortino (daily, ann.)": np.nan,
        "CAGR (%)": np.nan,
        "Calmar": np.nan,
        "Max Drawdown ($)": np.nan,
        "Max Drawdown (%)": np.nan,
        "Days": np.nan,
        "Years": np.nan,
    }
    if eq_series is None or eq_series.empty or len(eq_series) < 2:
        return out

    eq = eq_series.copy()
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()

    eq_r = eq.resample(resample_rule).last().ffill()
    ret = eq_r.pct_change().dropna()

    roll_max = eq_r.cummax()
    dd = eq_r - roll_max
    max_dd_usd = float(dd.min())
    max_dd_pct = float((dd / roll_max).min())

    out["Max Drawdown ($)"] = abs(max_dd_usd)
    out["Max Drawdown (%)"] = abs(max_dd_pct) * 100.0

    ann_factor = annualize_factor_from_resample(resample_rule)

    if len(ret) >= 30:
        mu = ret.mean()
        sd = ret.std(ddof=1)
        sharpe = (mu / sd) * np.sqrt(ann_factor) if sd and sd > 0 else np.nan

        mar = 0.0
        downside_sq = np.minimum(0.0, ret - mar) ** 2
        downside_dev = np.sqrt(downside_sq.mean())
        sortino = (mu / downside_dev) * np.sqrt(ann_factor) if downside_dev and downside_dev > 0 else np.nan

        out["Sharpe (daily, ann.)"] = float(sharpe) if np.isfinite(sharpe) else np.nan
        out["Sortino (daily, ann.)"] = float(sortino) if np.isfinite(sortino) else np.nan

    start_val = float(eq_r.iloc[0])
    end_val = float(eq_r.iloc[-1])
    days = (eq_r.index[-1] - eq_r.index[0]).total_seconds() / 86400.0
    years = days / 365.0 if days > 0 else np.nan
    out["Days"] = float(days) if np.isfinite(days) else np.nan
    out["Years"] = float(years) if np.isfinite(years) else np.nan

    if days > 0 and start_val > 0:
        cagr = (end_val / start_val) ** (ann_factor / days) - 1.0
        out["CAGR (%)"] = float(cagr * 100.0) if np.isfinite(cagr) else np.nan

        max_dd_frac = abs(max_dd_pct)
        calmar = (cagr / max_dd_frac) if (np.isfinite(cagr) and max_dd_frac and max_dd_frac > 0) else np.nan
        out["Calmar"] = float(calmar) if np.isfinite(calmar) else np.nan

    return out


def build_daily_returns_matrix_from_port_log(
    port_log: pd.DataFrame,
    markets: list[str],
    start_capital: float = 50_000.0,
    date_col: str = "Exit Time",
    pnl_col: str = "PnL_USD",
) -> pd.DataFrame:
    log = port_log.copy()
    log[date_col] = pd.to_datetime(log[date_col])
    log["Date"] = log[date_col].dt.floor("D")

    daily_pnl = (
        log.groupby(["Date", "Market"])[pnl_col]
        .sum()
        .unstack("Market")
        .reindex(columns=markets)
        .fillna(0.0)
    )

    daily_ret = daily_pnl / float(start_capital)
    return daily_ret


def erc_weights_from_cov_long_only(cov: pd.DataFrame, iters: int = 2000) -> dict:
    assets = cov.columns.tolist()
    n = len(assets)
    C = cov.values

    w = np.ones(n) / n

    for _ in range(iters):
        sigma_p = np.sqrt(w @ C @ w)
        if sigma_p <= 0:
            break

        mrc = (C @ w) / sigma_p
        rc = w * mrc
        target = rc.mean()

        w = w * (target / (rc + 1e-12))
        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum()

    return {assets[i]: float(w[i]) for i in range(n)}


def block_bootstrap_1d_returns(
    ret: pd.Series,
    block_size_days: int = 20,
    n_sims: int = 50000,
    seed: int = 42,
    ann_factor: float = 365.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = ret.values.astype(float)
    T = len(x)
    if T < block_size_days:
        raise ValueError("För få dagar för block_size_days")

    max_start = T - block_size_days
    n_blocks = int(np.ceil(T / block_size_days))

    out = np.empty((n_sims, 4), dtype=float)

    for s in range(n_sims):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([x[i:i + block_size_days] for i in starts])[:T]

        eq = np.cumprod(1.0 + sample)
        total_return = eq[-1] - 1.0

        mu = sample.mean()
        sd = sample.std(ddof=1)
        sharpe = (mu / sd) * np.sqrt(ann_factor) if sd > 0 else np.nan

        years = T / ann_factor
        cagr = (eq[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan

        roll_max = np.maximum.accumulate(eq)
        dd = eq / roll_max - 1.0
        maxdd = dd.min()

        out[s, 0] = total_return * 100.0
        out[s, 1] = cagr * 100.0
        out[s, 2] = sharpe
        out[s, 3] = maxdd * 100.0

    return pd.DataFrame(out, columns=["TotalReturn_%", "CAGR_%", "Sharpe", "MaxDD_%"])


def bootstrap_from_mtm_df(
    mtm_df_in: pd.DataFrame,
    label: str,
    start_capital: float = 50_000.0,
    block_size_days: int = 20,
    n_sims: int = 50000,
    seed: int = 42
) -> pd.DataFrame:
    mtm_daily_eq = mtm_df_in["Equity_MTM"].resample("D").last().ffill()
    mtm_daily_ret = mtm_daily_eq.pct_change().dropna()

    boot_df = block_bootstrap_1d_returns(
        mtm_daily_ret,
        block_size_days=block_size_days,
        n_sims=n_sims,
        seed=seed,
    )

    boot_df["EndEquity"] = start_capital * (1.0 + boot_df["TotalReturn_%"] / 100.0)

    print("\n" + "=" * 70)
    print(f"BOOTSTRAP SUMMARY ({label})")
    print("=" * 70)
    print(
        boot_df[["EndEquity", "TotalReturn_%", "MaxDD_%", "Sharpe", "CAGR_%"]]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    )
    print("Prob(MaxDD worse than -10%):", float((boot_df["MaxDD_%"] < -10.0).mean()))
    print("Prob(MaxDD worse than -15%):", float((boot_df["MaxDD_%"] < -15.0).mean()))
    print("Prob(CAGR < 0):", float((boot_df["CAGR_%"] < 0.0).mean()))

    return boot_df


# ==========================
# KÖR BACKTEST + PORTFÖLJ
# ==========================

all_results = []
all_trades = []


def iter_market_items(markets_obj):
    if isinstance(markets_obj, dict):
        for name, cfg in markets_obj.items():
            if isinstance(cfg, dict):
                cfg = cfg.copy()
                cfg.setdefault("name", name)
                yield cfg
            else:
                yield {"name": str(name), "csv": None, "pip_size": None}
    else:
        for item in markets_obj:
            yield item


for m in iter_market_items(markets):
    try:
        if not isinstance(m, dict):
            raise TypeError(f"Market config är inte dict: {type(m)} value={m}")

        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
            m["pip_size"],
            m.get("spread_points_per_pip", 10.0),
            cost_model=m.get("cost_model", None),
        )

        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)

    except Exception as e:
        name = m.get("name", str(m)) if isinstance(m, dict) else str(m)
        csv_ = m.get("csv", "?") if isinstance(m, dict) else "?"
        print(f"\n*** FEL för {name} ({csv_}): {e}\n")


if all_trades:
    portfolio_trades = pd.concat(all_trades, ignore_index=True)
    portfolio_trades["Entry Time"] = pd.to_datetime(portfolio_trades["Entry Time"])
    portfolio_trades["Exit Time"] = pd.to_datetime(portfolio_trades["Exit Time"])

    market_cfg = {m["name"]: m for m in markets}

    port_summary, port_eq, port_log = simulate_portfolio_equal_risk(
        portfolio_trades,
        market_cfg,
        start_capital=50_000.0,
        max_one_position_per_market=True,
    )

    plt.figure(figsize=(12, 5))
    plt.plot(port_eq["Time"], port_eq["Equity"])
    plt.title("Portfolio Equity Curve ($)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    dd_summary, mtm_df = compute_portfolio_mtm_equity_and_intraday_dd(
        port_log,
        market_cfg,
        start_capital=50_000.0,
        freq="5min",
        price_col="close",
    )

    port_summary.update(dd_summary)

    print("\n--- MTM / INTRADAY DD ---")
    for k, v in dd_summary.items():
        print(f"{k}: {v:.4f}")

    mtm_metrics = compute_risk_metrics_from_equity(mtm_df["Equity_MTM"], resample_rule="D")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    print("Sum PnL_USD:", float(port_log["PnL_USD"].sum()))
    print("Equity change:", float(port_summary["End Equity"] - port_summary["Start Capital"]))
    print("Diff:", float(port_log["PnL_USD"].sum() - (port_summary["End Equity"] - port_summary["Start Capital"])))

    print("Avg Units:", port_log["Units"].mean())
    print("Median Units:", port_log["Units"].median())

    print("\n$ per pip (approx) per market @ avg units (using median Entry Price):")
    for sym in sorted(port_log["Market"].unique()):
        sub = port_log[port_log["Market"] == sym].dropna(subset=["Entry Price"])
        if sub.empty:
            continue

        avg_units_sym = float(sub["Units"].mean())
        pip_size_sym = float(market_cfg[sym]["pip_size"])
        price_sym = float(sub["Entry Price"].median())

        usdjpy_price_sym = None
        if is_jpy_cross(sym):
            usd_series = build_price_series(market_cfg, price_col="close")["USDJPY"]
            usdjpy_price_sym = float(usd_series.asof(pd.Timestamp(sub["Entry Time"].median())))

        usd_per_pip_sym = approx_usd_per_pip(
            sym, pip_size_sym, avg_units_sym, price_sym, usdjpy_price=usdjpy_price_sym
        )
        print(f"{sym}: {usd_per_pip_sym:.4f}")

    print("\n" + "=" * 70)
    print(" PORTFÖLJ-RESULTAT (USD) ")
    print("=" * 70)

    for k in [
        "Sharpe (daily, ann.)", "Sortino (daily, ann.)", "CAGR (%)", "Calmar",
        "Days", "Years", "Max Drawdown ($)", "Max Drawdown (%)"
    ]:
        if k in mtm_metrics:
            port_summary[k] = mtm_metrics[k]

    for k, v in port_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ==========================
    # ERC WEIGHTS
    # ==========================

    erc_markets = sorted(port_log["Market"].unique().tolist())

    daily_ret = build_daily_returns_matrix_from_port_log(
        port_log=port_log,
        markets=erc_markets,
        start_capital=50_000.0,
    )

    daily_ret = daily_ret.sort_index()
    daily_ret = daily_ret.asfreq("D", fill_value=0.0)

    cov = daily_ret.cov()
    erc_w = erc_weights_from_cov_long_only(cov, iters=2000)

    w = np.array([erc_w[m] for m in erc_markets])
    C = cov.values
    sigma_p = np.sqrt(w @ C @ w)
    mrc = (C @ w) / sigma_p
    rc = w * mrc

    print("\nRisk contributions:")
    for i, m in enumerate(erc_markets):
        print(m, rc[i])
    if rc.min() > 0:
        print("RC ratio max/min:", rc.max() / rc.min())

    print("\n" + "=" * 70)
    print("ERC WEIGHTS (from equal-weight daily returns)")
    print("=" * 70)
    for k, v in erc_w.items():
        print(f"{k}: {v:.4f}")
    print("Sum:", sum(erc_w.values()))

    port_summary_erc, port_eq_erc, port_log_erc = simulate_portfolio_equal_risk(
        portfolio_trades,
        market_cfg,
        start_capital=50_000.0,
        max_one_position_per_market=True,
        exposure_scale=1.0,
        weights=erc_w,
    )

    dd_summary_erc, mtm_df_erc = compute_portfolio_mtm_equity_and_intraday_dd(
        port_log_erc,
        market_cfg,
        start_capital=50_000.0,
        freq="5min",
        price_col="close",
    )

    port_summary_erc.update(dd_summary_erc)

    print("\n" + "=" * 70)
    print("PORTFÖLJ-RESULTAT (ERC weights)")
    print("=" * 70)
    for k, v in port_summary_erc.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    plt.figure(figsize=(12, 5))
    plt.plot(mtm_df.index, mtm_df["Equity_MTM"], label="Equal Weight")
    plt.plot(mtm_df_erc.index, mtm_df_erc["Equity_MTM"], label="ERC Weight")
    plt.title("MTM Equity Comparison: Equal vs ERC (5m)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nDaily return std per market:")
    print(daily_ret.std())

    print("\nDaily return correlation:")
    print(daily_ret.corr())


    def pnl_attribution_by_market(port_log: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        df = port_log.copy()
        df["Exit Time"] = pd.to_datetime(df["Exit Time"])

        mask = (df["Exit Time"] >= pd.Timestamp(start)) & (df["Exit Time"] <= pd.Timestamp(end))
        sub = df.loc[mask].copy()

        if sub.empty:
            return pd.DataFrame()

        out = (
            sub.groupby("Market")
            .agg(
                Trades=("PnL_USD", "size"),
                PnL_USD=("PnL_USD", "sum"),
                AvgTrade_USD=("PnL_USD", "mean"),
                GrossProfit_USD=("PnL_USD", lambda x: x[x > 0].sum()),
                GrossLoss_USD=("PnL_USD", lambda x: x[x < 0].sum()),
            )
            .sort_values("PnL_USD", ascending=False)
        )

        total = out["PnL_USD"].sum()
        out["PnL_Share_%"] = np.where(total != 0, 100.0 * out["PnL_USD"] / total, np.nan)

        gp = out["GrossProfit_USD"]
        gl = out["GrossLoss_USD"].abs()
        out["ProfitFactor"] = np.where(gl > 0, gp / gl, np.inf)

        return out

    print("\n=== PnL attribution 2003(05)-2011 ===")
    print(pnl_attribution_by_market(port_log, "2003-05-01", "2011-12-31"))

    print("\n=== PnL attribution 2012-2020 ===")
    print(pnl_attribution_by_market(port_log, "2012-01-01", "2020-12-31"))

    print("\n=== PnL attribution 2021-2025 ===")
    print(pnl_attribution_by_market(port_log, "2021-01-01", "2025-12-31"))

    def drawdown_attribution_by_market_realized(port_log: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        df = port_log.copy()
        df["Exit Time"] = pd.to_datetime(df["Exit Time"])

        mask = (df["Exit Time"] >= pd.Timestamp(start)) & (df["Exit Time"] <= pd.Timestamp(end))
        sub = df.loc[mask].copy()

        if sub.empty:
            return pd.DataFrame()

        rows = []
        for mkt, g in sub.groupby("Market"):
            g = g.sort_values("Exit Time").copy()
            g["equity"] = g["PnL_USD"].cumsum()
            g["roll_max"] = g["equity"].cummax()
            g["dd"] = g["equity"] - g["roll_max"]

            rows.append({
                "Market": mkt,
                "Trades": len(g),
                "PnL_USD": float(g["PnL_USD"].sum()),
                "MaxDD_USD": float(abs(g["dd"].min())),
                "AvgTrade_USD": float(g["PnL_USD"].mean()),
            })

        out = pd.DataFrame(rows).sort_values("PnL_USD", ascending=False)
        return out

    print("\n=== Realized DD attribution 2003(05)-2011 ===")
    print(drawdown_attribution_by_market_realized(port_log, "2003-05-01", "2011-12-31"))

    print("\n=== Realized DD attribution 2012-2020 ===")
    print(drawdown_attribution_by_market_realized(port_log, "2012-01-01", "2020-12-31"))

    print("\n=== Realized DD attribution 2021-2025 ===")
    print(drawdown_attribution_by_market_realized(port_log, "2021-01-01", "2025-12-31"))


    def daily_market_attribution(port_log: pd.DataFrame, markets: list[str], start: str, end: str,
                                 start_capital: float = 50000.0):
        daily_ret = build_daily_returns_matrix_from_port_log(
            port_log=port_log,
            markets=markets,
            start_capital=start_capital,
        )

        daily_ret = daily_ret.sort_index()
        daily_ret = daily_ret.asfreq("D", fill_value=0.0)

        mask = (daily_ret.index >= pd.Timestamp(start)) & (daily_ret.index <= pd.Timestamp(end))
        sub = daily_ret.loc[mask].copy()

        daily_pnl = sub * start_capital

        rows = []
        for mkt in markets:
            s = daily_pnl[mkt].copy()
            eq = s.cumsum()
            roll_max = eq.cummax()
            dd = eq - roll_max

            rows.append({
                "Market": mkt,
                "PnL_USD": float(s.sum()),
                "DailyVol_USD": float(s.std(ddof=1)),
                "MaxDD_USD": float(abs(dd.min())),
                "BestDay_USD": float(s.max()),
                "WorstDay_USD": float(s.min()),
            })

        return pd.DataFrame(rows).sort_values("PnL_USD", ascending=False)

    mkts = sorted(port_log["Market"].unique())

    print("\n=== Daily attribution 2003(05)-2011 ===")
    print(daily_market_attribution(port_log, mkts, "2003-05-01", "2011-12-31"))

    print("\n=== Daily attribution 2012-2020 ===")
    print(daily_market_attribution(port_log, mkts, "2012-01-01", "2020-12-31"))

    print("\n=== Daily attribution 2021-2025 ===")
    print(daily_market_attribution(port_log, mkts, "2021-01-01", "2025-12-31"))



    boot_equal_df = bootstrap_from_mtm_df(
        mtm_df,
        label="Equal weights",
        block_size_days=20,
        n_sims=50000,
        seed=42,
    )

    boot_erc_df = bootstrap_from_mtm_df(
        mtm_df_erc,
        label="ERC weights",
        block_size_days=20,
        n_sims=50000,
        seed=42,
    )

    plt.figure(figsize=(10, 4))
    plt.hist(boot_equal_df["TotalReturn_%"], bins=60, alpha=0.5, label="Equal")
    plt.title("Bootstrap distribution: Total Return (%)")
    plt.xlabel("Total Return (%)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(boot_equal_df["MaxDD_%"], bins=60, alpha=0.5, label="Equal")
    plt.title("Bootstrap distribution: Max Drawdown (%)")
    plt.xlabel("Max DD (%) (negativt)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


else:
    print("Inga trades att simulera i portföljen.")
