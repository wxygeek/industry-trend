"""申万行业指数数据采集模块 - 通过 akshare API 下载

支持申万一级和二级行业指数的日K线数据下载。
数据来源: 申万宏源研究所（通过 akshare 封装访问）
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

import pandas as pd
import akshare as ak

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.industries import SW_LEVEL2_INDUSTRIES, get_industries

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEVEL1_DATA_DIR = PROJECT_ROOT / "data" / "sw_level1" / "daily"
LEVEL2_DATA_DIR = PROJECT_ROOT / "data" / "sw_level2" / "daily"
LEVEL1_ANALYSIS_DIR = PROJECT_ROOT / "data" / "sw_level1" / "analysis"
LEVEL2_ANALYSIS_DIR = PROJECT_ROOT / "data" / "sw_level2" / "analysis"

KEEP_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]

# akshare 返回的中文列名 -> 英文标准列名
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}


def data_dir_for_level(level: int) -> Path:
    """返回指定级别的数据目录"""
    if level == 1:
        return LEVEL1_DATA_DIR
    elif level == 2:
        return LEVEL2_DATA_DIR
    raise ValueError(f"不支持的行业级别: {level}")


def analysis_dir_for_level(level: int) -> Path:
    """返回指定级别的分析结果目录"""
    if level == 1:
        return LEVEL1_ANALYSIS_DIR
    elif level == 2:
        return LEVEL2_ANALYSIS_DIR
    raise ValueError(f"不支持的行业级别: {level}")


def csv_path_for(code: str, name: str, level: int = 2) -> Path:
    """返回行业数据的 CSV 文件路径"""
    return data_dir_for_level(level) / f"{code}_{name}.csv"


def load_existing_csv(code: str, name: str, level: int = 2) -> pd.DataFrame | None:
    """加载已有的本地 CSV 数据"""
    path = csv_path_for(code, name, level)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_csv(df: pd.DataFrame, code: str, name: str, level: int = 2) -> Path:
    """保存 DataFrame 到 CSV"""
    d = data_dir_for_level(level)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{code}_{name}.csv"
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def merge_incremental(existing: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """增量合并：只添加比已有数据更新的行"""
    if existing is None or existing.empty:
        return new_df
    latest_date = existing["date"].max()
    new_rows = new_df[new_df["date"] > latest_date]
    if new_rows.empty:
        return existing
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def download_industry(code: str, name: str) -> pd.DataFrame:
    """通过 akshare 下载单个行业指数的日K线数据

    Args:
        code: 行业指数代码（如 "801081"）
        name: 行业名称（如 "半导体"），仅用于日志

    Returns:
        DataFrame，列为 date,open,high,low,close,volume,amount
    """
    logger.info(f"下载 {name}({code}) 行情数据...")

    df = ak.index_hist_sw(symbol=code, period="day")

    if df is None or df.empty:
        logger.warning(f"  {name}({code}) 无数据返回")
        return pd.DataFrame(columns=KEEP_COLUMNS)

    # 删除 "代码" 列，重命名中文列名
    if "代码" in df.columns:
        df = df.drop(columns=["代码"])
    df = df.rename(columns=COLUMN_MAP)

    df = df[KEEP_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"  下载完成: {len(df)} 条日K线记录")
    return df


def download_and_save(code: str, name: str, level: int = 2) -> Path:
    """下载并保存（支持增量更新）单个行业数据"""
    new_df = download_industry(code, name)
    existing = load_existing_csv(code, name, level)
    merged = merge_incremental(existing, new_df)
    path = save_csv(merged, code, name, level)

    if existing is not None:
        new_count = len(merged) - len(existing)
        logger.info(f"  增量更新: 新增 {new_count} 条, 总计 {len(merged)} 条")
    else:
        logger.info(f"  首次下载: 保存 {len(merged)} 条到 {path}")
    return path


def download_all(
    level: int = 2,
    industries: dict[str, str] | None = None,
    delay: float = 1.0,
    max_retries: int = 3,
) -> dict:
    """批量下载所有行业数据

    Args:
        level: 行业级别（1=一级, 2=二级）
        industries: 行业字典 {code: name}，默认按 level 自动选择
        delay: 每次下载间隔秒数
        max_retries: 单个行业最大重试次数

    Returns:
        {"success": [...], "failed": [...]} 结果汇总
    """
    if industries is None:
        industries = get_industries(level)

    results = {"success": [], "failed": []}

    for i, (code, name) in enumerate(industries.items()):
        logger.info(f"[{i+1}/{len(industries)}] 处理 {name}({code})")
        succeeded = False

        for attempt in range(1, max_retries + 1):
            try:
                download_and_save(code, name, level)
                results["success"].append((code, name))
                succeeded = True
                break
            except Exception as e:
                wait_time = delay * (2 ** (attempt - 1))
                logger.warning(
                    f"  第{attempt}次尝试失败: {e}"
                    + (f", {wait_time:.1f}秒后重试..." if attempt < max_retries else "")
                )
                if attempt < max_retries:
                    time.sleep(wait_time)

        if not succeeded:
            results["failed"].append((code, name))
            logger.error(f"  {name}({code}) 下载失败，已跳过")

        if i < len(industries) - 1:
            time.sleep(delay)

    logger.info(
        f"\n下载完成: 成功 {len(results['success'])}/{len(industries)}, "
        f"失败 {len(results['failed'])}"
    )
    if results["failed"]:
        logger.warning(f"失败行业: {[f'{n}({c})' for c, n in results['failed']]}")

    return results
