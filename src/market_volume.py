"""A股大盘成交量数据获取与百分位分析

获取上证综指（sh000001）日K线成交量数据，聚合为周度成交量，
计算5年滚动百分位，用于回测中的成交量过滤（量能择时）。

当A股成交量处于5年百分位60%以下时，视为市场量能不足，不执行买入操作。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKET_DATA_DIR = PROJECT_ROOT / "data" / "market"
MARKET_CSV = MARKET_DATA_DIR / "sh000001_daily.csv"


def download_market_index() -> pd.DataFrame:
    """下载上证综指日K线数据（含成交量）

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    logger.info("下载上证综指(sh000001)日K线数据...")
    df = ak.stock_zh_index_daily_em(symbol="sh000001")

    if df is None or df.empty:
        logger.warning("上证综指数据为空")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    for col in ["volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["volume"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"  下载完成: {len(df)} 条日K线 ({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df


def save_market_data(df: pd.DataFrame) -> Path:
    """保存大盘数据到 CSV"""
    MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MARKET_CSV, index=False)
    logger.info(f"  大盘数据已保存至 {MARKET_CSV}")
    return MARKET_CSV


def load_market_data() -> pd.DataFrame | None:
    """加载本地大盘数据"""
    if not MARKET_CSV.exists():
        return None
    df = pd.read_csv(MARKET_CSV, parse_dates=["date"])
    return df


def download_and_save_market() -> pd.DataFrame:
    """下载并保存大盘数据（增量更新）"""
    existing = load_market_data()
    new_df = download_market_index()

    if new_df.empty:
        return existing if existing is not None else pd.DataFrame()

    if existing is not None and not existing.empty:
        latest = existing["date"].max()
        new_rows = new_df[new_df["date"] > latest]
        if not new_rows.empty:
            merged = pd.concat([existing, new_rows], ignore_index=True)
            merged = merged.sort_values("date").reset_index(drop=True)
        else:
            merged = existing
    else:
        merged = new_df

    save_market_data(merged)
    return merged


def compute_weekly_market_volume(daily_df: pd.DataFrame) -> pd.DataFrame:
    """将日成交量聚合为周成交量（周五截止，与行业周K线对齐）

    Returns:
        DataFrame with columns: date, volume
    """
    df = daily_df[["date", "volume"]].copy()
    df = df.set_index("date")
    weekly = df["volume"].resample("W-FRI").sum()
    weekly = weekly[weekly > 0].reset_index()
    weekly.columns = ["date", "volume"]
    return weekly


def compute_volume_percentile(
    weekly_df: pd.DataFrame,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """计算成交量的滚动N年百分位

    对每一周，计算当前周成交量在过去N年周成交量中的百分位排名。

    Args:
        weekly_df: 周度成交量 DataFrame (date, volume)
        lookback_years: 回看年数，默认5年

    Returns:
        DataFrame with columns: date, volume, percentile (0.0~1.0)
    """
    window = lookback_years * 52
    min_periods = 52  # 至少1年数据才开始计算

    vol = weekly_df["volume"].values
    percentiles = np.full(len(vol), np.nan)

    for i in range(min_periods, len(vol)):
        start = max(0, i - window)
        historical = vol[start:i]  # 不含当前周
        current = vol[i]
        if len(historical) > 0:
            percentiles[i] = (historical < current).sum() / len(historical)

    result = weekly_df.copy()
    result["percentile"] = percentiles
    return result


def get_volume_percentile_lookup(
    daily_df: pd.DataFrame | None = None,
    lookback_years: int = 5,
) -> dict[pd.Timestamp, float]:
    """获取成交量百分位查找表

    Args:
        daily_df: 日K线数据。如果为 None，从本地加载。
        lookback_years: 回看年数

    Returns:
        {weekly_date: percentile} 字典
    """
    if daily_df is None:
        daily_df = load_market_data()
    if daily_df is None or daily_df.empty:
        return {}

    weekly = compute_weekly_market_volume(daily_df)
    pct_df = compute_volume_percentile(weekly, lookback_years)
    pct_df = pct_df.dropna(subset=["percentile"])

    return dict(zip(pct_df["date"], pct_df["percentile"]))


def get_volume_percentile_df(
    daily_df: pd.DataFrame | None = None,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """获取成交量百分位 DataFrame（用于可视化）

    Returns:
        DataFrame with columns: date, volume, percentile
    """
    if daily_df is None:
        daily_df = load_market_data()
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["date", "volume", "percentile"])

    weekly = compute_weekly_market_volume(daily_df)
    return compute_volume_percentile(weekly, lookback_years)
