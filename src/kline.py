"""周K线生成模块 - 日K线聚合 + 34周MA计算"""

import pandas as pd


def daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """将日K线数据聚合为周K线

    使用自然周（周一至周五）聚合。支持不完整周（节假日或当前周）。

    Args:
        daily_df: 日K线 DataFrame，需包含 date, open, high, low, close, volume, amount

    Returns:
        周K线 DataFrame
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    weekly = df.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
    )

    weekly = weekly.dropna(subset=["open", "close"])
    weekly = weekly.reset_index()
    return weekly


def calculate_ma(weekly_df: pd.DataFrame, period: int = 34) -> pd.DataFrame:
    """计算移动平均线

    Args:
        weekly_df: 周K线 DataFrame，需包含 close 列
        period: MA周期，默认34周

    Returns:
        添加了 ma34 列的 DataFrame
    """
    df = weekly_df.copy()
    df[f"ma{period}"] = df["close"].rolling(window=period).mean()
    return df


def calculate_ma_slope(
    weekly_df: pd.DataFrame, period: int = 34, slope_window: int = 4
) -> pd.DataFrame:
    """计算MA斜率

    斜率 = (当前MA - N周前MA) / N周前MA

    Args:
        weekly_df: 需包含 ma{period} 列
        period: MA周期
        slope_window: 斜率计算窗口，默认4周

    Returns:
        添加了 ma_slope 列的 DataFrame
    """
    df = weekly_df.copy()
    ma_col = f"ma{period}"
    df["ma_slope"] = (df[ma_col] - df[ma_col].shift(slope_window)) / df[
        ma_col
    ].shift(slope_window)
    return df


def calculate_atr(weekly_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 Average True Range

    ATR = 真实波幅（TR）的 N 期简单移动平均。
    TR = max(high-low, |high-prev_close|, |low-prev_close|)

    Args:
        weekly_df: 周K线 DataFrame，需包含 high, low, close
        period: ATR 周期，默认14周

    Returns:
        添加了 atr{period} 列的 DataFrame
    """
    df = weekly_df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df[f"atr{period}"] = tr.rolling(period).mean()
    return df


def generate_weekly_kline(
    daily_df: pd.DataFrame, ma_period: int = 34, slope_window: int = 4
) -> pd.DataFrame:
    """完整的周K线生成流程

    日K → 周K聚合 → MA34计算 → MA斜率计算 → ATR14计算

    Args:
        daily_df: 日K线 DataFrame
        ma_period: MA周期，默认34
        slope_window: 斜率窗口，默认4周

    Returns:
        标准化周K线 DataFrame，包含: date, open, high, low, close, volume, amount, ma34, ma_slope, atr14
    """
    weekly = daily_to_weekly(daily_df)
    weekly = calculate_ma(weekly, period=ma_period)
    weekly = calculate_ma_slope(weekly, period=ma_period, slope_window=slope_window)
    weekly = calculate_atr(weekly)
    return weekly
