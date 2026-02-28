"""申万一级行业指数数据采集模块 - 通过 Playwright 自动化浏览器下载"""

from __future__ import annotations

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.industries import SW_LEVEL1_INDUSTRIES, SWS_DETAIL_URL

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sw_level1" / "daily"

COLUMN_MAP = {
    "指数代码": "code",
    "指数名称": "name",
    "发布日期": "date",
    "开盘指数": "open",
    "最高指数": "high",
    "最低指数": "low",
    "收盘指数": "close",
    "成交量(亿股)": "volume",
    "成交额(亿元)": "amount",
}

KEEP_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def parse_xls(xls_path: str) -> pd.DataFrame:
    """解析下载的 XLS 文件，提取并标准化列名"""
    df = pd.read_excel(xls_path)
    df = df.rename(columns=COLUMN_MAP)
    df = df[KEEP_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def csv_path_for(code: str, name: str) -> Path:
    """返回行业数据的 CSV 文件路径"""
    return DATA_DIR / f"{code}_{name}.csv"


def load_existing_csv(code: str, name: str) -> pd.DataFrame | None:
    """加载已有的本地 CSV 数据"""
    path = csv_path_for(code, name)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_csv(df: pd.DataFrame, code: str, name: str) -> Path:
    """保存 DataFrame 到 CSV"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = csv_path_for(code, name)
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


class SWSScraper:
    """申万研究所行情数据采集器"""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._page = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._page.set_default_timeout(30000)

    def stop(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def download_industry(self, code: str, name: str) -> pd.DataFrame:
        """下载单个行业的历史行情数据

        导航到行业详情页，点击"行情下载"获取 XLS 文件，解析后返回 DataFrame。
        """
        url = f"{SWS_DETAIL_URL}?code={code}&name={name}"
        logger.info(f"下载 {name}({code}) 行情数据...")
        self._page.goto(url, wait_until="networkidle")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self._page.expect_download() as download_info:
                self._page.locator("button:has-text('行情下载')").click()
            download = download_info.value
            xls_path = os.path.join(tmpdir, download.suggested_filename)
            download.save_as(xls_path)
            logger.info(f"  已下载: {download.suggested_filename}")
            df = parse_xls(xls_path)

        logger.info(f"  解析完成: {len(df)} 条日K线记录")
        return df

    def download_and_save(self, code: str, name: str) -> Path:
        """下载并保存（支持增量更新）单个行业数据"""
        new_df = self.download_industry(code, name)
        existing = load_existing_csv(code, name)
        merged = merge_incremental(existing, new_df)
        path = save_csv(merged, code, name)

        if existing is not None:
            new_count = len(merged) - len(existing)
            logger.info(f"  增量更新: 新增 {new_count} 条, 总计 {len(merged)} 条")
        else:
            logger.info(f"  首次下载: 保存 {len(merged)} 条到 {path}")
        return path

    def download_all(
        self,
        industries: dict[str, str] | None = None,
        delay: float = 2.5,
        max_retries: int = 3,
    ) -> dict:
        """批量下载所有行业数据

        Args:
            industries: 行业字典 {code: name}，默认全部31个
            delay: 每次下载间隔秒数
            max_retries: 单个行业最大重试次数

        Returns:
            {"success": [...], "failed": [...]} 结果汇总
        """
        if industries is None:
            industries = SW_LEVEL1_INDUSTRIES

        results = {"success": [], "failed": []}

        for i, (code, name) in enumerate(industries.items()):
            logger.info(f"[{i+1}/{len(industries)}] 处理 {name}({code})")
            succeeded = False

            for attempt in range(1, max_retries + 1):
                try:
                    self.download_and_save(code, name)
                    results["success"].append((code, name))
                    succeeded = True
                    break
                except (PlaywrightTimeout, Exception) as e:
                    wait_time = delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"  第{attempt}次尝试失败: {e}"
                        + (f", {wait_time:.0f}秒后重试..." if attempt < max_retries else "")
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
