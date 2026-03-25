import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from math import sqrt
import os

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "EURJPY",
        "csv": "EURJPY_1H_2003(05)-2026.csv",
        "pip_size": 0.01,
        # USDJPY: 0.01, NZDUSD och AUDUSD och USDCHF och USDCAD och EURUSD och GBPUSD och USDCAD: 0.0001 pip
        "spread_points_per_pip": 10.0,  # om spread_points är pipetter (vanligast). Om er kolumn redan är pips: sätt 1.0
    },
]

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5

SLIPPAGE_PIPS = 0.35  # NZDUSD: 0.00, USDCAD: 0.10, AUDUSD och USDCHF och USDJPY och GBPUSD: 0.08, EURUSD:  0.05 pip (0.5 pipette)
FIXED_SPREAD_PIPS = 0.9  # NZDUSD: 1.2, AUDUSD och USDCHF: 0.18, USDCAD: 0.22, USDJPY och GBPUSD: 0.15, EURUSD: 0.10 pip (1 pipette) - anpassa!
COMM_PIPS_PER_SIDE = 0.25  # AUDUSD och USDCHF och USDCAD och USDJPY och GBPUSD och EURUSD: 0.02 pip per sida - exempel


def compute_stats_from_trades(trades_df: pd.DataFrame) -> dict:
    """
    Samma logik som era totala stats, men på en subset av trades.
    trades_df måste ha kolumner: pnl, equity (valfritt), is_win (valfritt)
    """
    if trades_df.empty:
        return {}

    df_ = trades_df.copy()

    # equity behövs för DD
    df_["equity"] = df_["pnl"].cumsum()

    df_["is_win"] = df_["pnl"] > 0

    gross_profit = df_.loc[df_["pnl"] > 0, "pnl"].sum()
    gross_loss = df_.loc[df_["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = df_.loc[df_["pnl"] > 0, "pnl"].mean()
    avg_loss = df_.loc[df_["pnl"] < 0, "pnl"].mean()

    winrate = df_["is_win"].mean()
    expectancy = df_["pnl"].mean()

    roll_max = df_["equity"].cummax()
    dd = df_["equity"] - roll_max
    max_dd_points = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Losing streak
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


def run_backtest_for_market(market_name: str, csv_path: str, pip_size: float, spread_points_per_pip: float = 10.0):
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    # 1) Ladda data
    df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()
    # ---- DEDUPE INDEX (viktigt för groupby/transform) ----
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()

    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(COMM_PIPS_PER_SIDE)


    def asia_range(df: pd.DataFrame) -> pd.DataFrame:
        """
        Beräknar Asia range för 1H-data.

        Asia range definieras här som high/low från 00:00 till 07:00.
        Nivåerna blir tillgängliga från och med 08:00-candlen.

        Kräver:
        - DateTimeIndex
        - kolumner: 'High', 'Low'
        """
        data = df.copy()

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame måste ha DateTimeIndex")

        # Filtrera ut candles som tillhör Asia range: 00:00–07:00
        session = data.between_time("00:00", "07:00")

        # Beräkna high/low per dag
        daily_range = session.groupby(session.index.date).agg(
            asia_high=("high", "max"),
            asia_low=("low", "min")
        )

        # Mappa tillbaka till hela dagen
        data["date"] = data.index.date
        data = data.merge(
            daily_range,
            left_on="date",
            right_index=True,
            how="left"
        )

        # Hjälpkolumner
        data["asia_range"] = data["asia_high"] - data["asia_low"]
        data["asia_mid"] = (data["asia_high"] + data["asia_low"]) / 2

        # Gör nivåerna otillgängliga innan Asia-sessionen är klar
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

    #print(df[["asia_range", "asia_range_pct_rank"]].dropna().head(20))

    def ATR(df: pd.DataFrame, period: int = 14, method: str = "wilder") -> pd.Series:
        """
        Beräknar ATR (Average True Range).

        Parametrar
        ----------
        df : pd.DataFrame
            Måste innehålla kolumnerna: 'High', 'Low', 'Close'
        period : int
            ATR-period, t.ex. 14
        method : str
            'wilder' för klassisk Wilder ATR
            'sma' för enkelt glidande medelvärde av True Range

        Returnerar
        ----------
        pd.Series
            ATR-serie med samma index som df
        """
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame måste innehålla kolumnerna {required_cols}")

        high = df["high"]
        low = df["low"]
        close = df["close"]
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

        # Normal session (ex 01:00 -> 07:00)
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)

        # Wrap session (ex 23:00 -> 03:00)
        # Då är det "i session" om tiden är >= start ELLER < end
        return (t >= session_start_t) or (t < session_end_t)

    USE_SPREAD_PIPS_COL = 'spread_pips' in df.columns
    USE_SPREAD_POINTS_COL = 'spread_points' in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row['spread_pips'])
        if USE_SPREAD_POINTS_COL:
            return float(row['spread_points']) / float(spread_points_per_pip)
        return float(FIXED_SPREAD_PIPS)


    # 4) Backtest-loop
    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None
    entry_asia_range_pct_rank = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):

        ts = idx_list[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        prev_row = df.iloc[i - 1]

        # ======================
        # EXIT-logik (signal på bar i, fill på bar i+1 open)
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            atr_now = row["atr"]
            atr_ma_now = row["atr_ma"]

            #'''
            if pos_direction == 'LONG':
                if np.isfinite(atr_now) and np.isfinite(atr_ma_now) and atr_now < atr_ma_now:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(SLIPPAGE_PIPS)

                    exit_price = next_row["open"] - HALF * spread_px - slip_px
                    exit_reason = 'atr_exit'
           # '''
            if pos_direction == 'SHORT':  # SHORT
                if np.isfinite(atr_now) and np.isfinite(atr_ma_now) and atr_now < atr_ma_now:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(SLIPPAGE_PIPS)

                    exit_price = next_row["open"] + HALF * spread_px + slip_px
                    exit_reason = 'atr_exit'
            #'''
            if exit_price is not None:
                exit_time = idx_list[i + 1]  # matchar fill på next_row open
                comm_px = commission_round_turn_price()

                if pos_direction == 'LONG':
                    pnl = (exit_price - entry_price) - comm_px
                else:
                    pnl = (entry_price - exit_price) - comm_px

                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Direction': pos_direction,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'pnl': pnl,
                    'asia_range_pct_rank': entry_asia_range_pct_rank,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                entry_asia_range_pct_rank = None

                continue  # viktigt: undvik att gå in i ny trade samma iteration

        # Sessionfilter: både signalbar och fillbar måste vara i session
        if not in_session(ts):
            continue

        # hoppa entry-logik om vi fortfarande är i trade
        if in_position:
            continue
        # ======================
        # ENTRY-logik
        # ======================
        close_price = row["close"]
        next_open = next_row['open']
        prev_close = prev_row["close"]
        asia_low = row["asia_low"]
        asia_high = row["asia_high"]
        prev_asia_low = prev_row["asia_low"]
        prev_asia_high = prev_row["asia_high"]
        atr_now = row["atr"]
        atr_ma_now = row["atr_ma"]
        asia_range_pct_rank = row["asia_range_pct_rank"]

        if not np.isfinite(asia_high) or not np.isfinite(asia_low):
            continue

        if not np.isfinite(atr_now) or not np.isfinite(atr_ma_now):
            continue

        atr_filter = atr_now > atr_ma_now

        bullish_breakout = close_price > asia_high
        bearish_breakout = close_price < asia_low

        long_signal = bullish_breakout and atr_filter
        short_signal = bearish_breakout and atr_filter

        #long_volatility_filter = asia_range_pct_rank > 0.75
        #short_volatility_filter = asia_range_pct_rank < 0.65

        volatility_filter = asia_range_pct_rank > 0.7

        long_entry = long_signal and volatility_filter #long_volatility_filter
        short_entry = short_signal and volatility_filter #short_volatility_filter

        #'''
        if long_entry:
            pos_direction = 'LONG'
            entry_time = idx_list[i + 1]  # fill sker på nästa bar open

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)

            entry_asia_range_pct_rank = row["asia_range_pct_rank"]
            entry_price = next_open + HALF * spread_px + slip_px
            in_position = True

        #'''
        elif short_entry:
            pos_direction = 'SHORT'
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)

            entry_asia_range_pct_rank = row["asia_range_pct_rank"]
            entry_price = next_open - HALF * spread_px - slip_px
            in_position = True
        #'''
    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)

    shorts_high = trades_df[
        (trades_df["Direction"] == "SHORT") &
        (trades_df["asia_range_pct_rank"] > 0.7)
        ].copy()

    shorts_not_high = trades_df[
        (trades_df["Direction"] == "SHORT") &
        (trades_df["asia_range_pct_rank"] <= 0.7)
        ].copy()

    print("SHORTS > 80th percentile Asia range")
    print(compute_stats_from_trades(shorts_high))
    print("\nSHORTS <= 80th percentile Asia range")
    print(compute_stats_from_trades(shorts_not_high))

    longs_high = trades_df[
        (trades_df["Direction"] == "LONG") &
        (trades_df["asia_range_pct_rank"] > 0.7)
        ].copy()

    longs_not_high = trades_df[
        (trades_df["Direction"] == "LONG") &
        (trades_df["asia_range_pct_rank"] <= 0.7)
        ].copy()

    print("LONGS > 80th percentile Asia range")
    print(compute_stats_from_trades(longs_high))
    print("\nLONGS <= 80th percentile Asia range")
    print(compute_stats_from_trades(longs_not_high))

    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    # --- Extra statistik ---
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()  # negativt

    winrate = trades_df["is_win"].mean()

    # Expectancy per trade
    expectancy = trades_df["pnl"].mean()

    # Drawdown
    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()  # negativt
    max_dd_points = abs(max_dd)  # positivt för rapportering

    # Longest losing streak (räknat i trades)
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (inte tidsnormaliserad)
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




        # ==========================
        # PER-ÅR STATS (per market)
        # ==========================
    print("\n--- PER-ÅR STATS ---")

    # vi använder Exit Time som "trade year" (rekommenderat)
    trades_df["Year"] = pd.to_datetime(trades_df["Exit Time"]).dt.year

    years = sorted(trades_df["Year"].unique().tolist())
    for y in years:
        sub = trades_df[trades_df["Year"] == y].copy()
        y_stats = compute_stats_from_trades(sub)

        if not y_stats:
            continue

        print(f"\n{market_name} - {y}:")
        print(f"Trades: {y_stats['Trades']}")
        print(f"Total PnL (points): {y_stats['Total PnL (points)']:.4f}")
        print(f"Gross Profit: {y_stats['Gross Profit']:.4f}")
        print(f"Gross Loss: {y_stats['Gross Loss']:.4f}")
        print(f"Profit Factor: {y_stats['Profit Factor']:.4f}")
        print(f"Winrate: {y_stats['Winrate']:.4f}")
        print(f"Avg Win: {y_stats['Avg Win']:.4f}")
        print(f"Avg Loss: {y_stats['Avg Loss']:.4f}")
        print(f"Expectancy (avg/trade): {y_stats['Expectancy (avg/trade)']:.4f}")
        print(f"Max Drawdown (points): {y_stats['Max Drawdown (points)']:.4f}")
        print(f"Max Losing Streak (trades): {y_stats['Max Losing Streak (trades)']}")
        print(f"Sharpe (trade-level): {y_stats['Sharpe (trade-level)']:.4f}")

    trades_df["Year"] = pd.to_datetime(trades_df["Exit Time"]).dt.year
    filtered = trades_df[~trades_df["Year"].isin([2011])].copy()
    stats_ex_0809 = compute_stats_from_trades(filtered)
    print(stats_ex_0809)

    # PLOT (som du redan får)
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
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []

for m in markets:
    try:
        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
            m["pip_size"],
            m.get("spread_points_per_pip", 10.0),
        )
        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)
    except Exception as e:
        print(f"\n*** FEL för {m['name']} ({m['csv']}): {e}\n")
