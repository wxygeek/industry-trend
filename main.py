"""行业趋势跟踪分析框架 - CLI 入口

用法:
    python main.py download     # 下载所有行业数据
    python main.py analyze      # 分析已有数据
    python main.py all          # 下载 + 分析
"""

import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.industries import SW_LEVEL1_INDUSTRIES
from src.scraper import SWSScraper, load_existing_csv
from src.kline import generate_weekly_kline
from src.stage_analyzer import batch_analyze, StageConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_download(headless: bool = True):
    """下载所有行业数据"""
    logger.info("开始下载31个申万一级行业数据...")
    with SWSScraper(headless=headless) as scraper:
        results = scraper.download_all()
    return results


def cmd_analyze(config: StageConfig | None = None) -> pd.DataFrame:
    """分析已有数据，输出阶段判断结果"""
    if config is None:
        config = StageConfig()

    logger.info("加载本地数据并生成周K线...")
    industry_data = {}

    for code, name in SW_LEVEL1_INDUSTRIES.items():
        daily_df = load_existing_csv(code, name)
        if daily_df is None or daily_df.empty:
            logger.warning(f"  {name}({code}) 无本地数据，跳过")
            continue

        weekly_df = generate_weekly_kline(daily_df)
        industry_data[code] = (name, weekly_df)
        logger.info(f"  {name}({code}): {len(daily_df)} 日K → {len(weekly_df)} 周K")

    if not industry_data:
        logger.error("无可用数据，请先运行 download 命令")
        return pd.DataFrame()

    logger.info(f"\n分析 {len(industry_data)} 个行业的趋势阶段...")
    summary = batch_analyze(industry_data, config)
    print_summary(summary)
    return summary


def print_summary(summary: pd.DataFrame):
    """在终端打印阶段汇总表"""
    if summary.empty:
        print("无分析结果")
        return

    stage_names = {1: "Stage 1 熊市", 2: "Stage 2 熊牛转换", 3: "Stage 3 牛市", 4: "Stage 4 牛熊转换"}

    print("\n" + "=" * 80)
    print("申万一级行业趋势阶段分析（基于34周均线）")
    print("=" * 80)

    # 分布统计
    dist = summary["stage"].value_counts().sort_index()
    print("\n阶段分布:")
    for stage_num, count in dist.items():
        print(f"  {stage_names.get(stage_num, f'Stage {stage_num}')}: {count} 个行业")

    # 详细表格
    print(f"\n{'行业名称':　<6} {'代码':>8} {'阶段':>12} {'置信度':>6} {'收盘价':>10} "
          f"{'MA34':>10} {'偏离度':>8} {'MA斜率':>8}")
    print("-" * 80)

    for _, row in summary.iterrows():
        deviation = f"{row['price_position']*100:+.1f}%"
        slope_dir = "↑" if row["ma_slope"] > 0.005 else ("↓" if row["ma_slope"] < -0.005 else "→")
        print(
            f"{row['name']:　<6} {row['code']:>8} "
            f"{'S'+str(row['stage'])+' '+row['stage_label']:>12} "
            f"{row['confidence']:>6.2f} {row['close']:>10.2f} "
            f"{row['ma34']:>10.2f} {deviation:>8} {slope_dir:>8}"
        )
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "download":
        headless = "--no-headless" not in sys.argv
        cmd_download(headless=headless)
    elif command == "analyze":
        cmd_analyze()
    elif command == "all":
        headless = "--no-headless" not in sys.argv
        cmd_download(headless=headless)
        cmd_analyze()
    else:
        print(f"未知命令: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
