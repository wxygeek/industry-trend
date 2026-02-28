"""行业趋势跟踪分析框架 - CLI 入口

用法:
    python main.py download              # 下载一级行业数据（SWS 网站）
    python main.py download --level 2    # 下载二级行业数据（akshare）
    python main.py download --level all  # 下载全部

    python main.py analyze               # 分析一级行业
    python main.py analyze --level 2     # 分析二级行业

    python main.py all                   # 下载 + 分析一级行业
    python main.py all --level all       # 下载 + 分析全部
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.industries import SW_LEVEL1_INDUSTRIES, get_industries
from src.scraper import SWSScraper, load_existing_csv as load_level1_csv
from src.akshare_downloader import (
    download_all as akshare_download_all,
    load_existing_csv as load_level2_csv,
    analysis_dir_for_level,
)
from src.kline import generate_weekly_kline
from src.stage_analyzer import batch_analyze, StageConfig
from src.market_volume import download_and_save_market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_download(level: str = "1", headless: bool = True):
    """下载行业数据"""
    levels = [1, 2] if level == "all" else [int(level)]

    for lv in levels:
        if lv == 1:
            logger.info("开始下载31个申万一级行业数据（SWS 网站）...")
            with SWSScraper(headless=headless) as scraper:
                scraper.download_all()
        elif lv == 2:
            industries = get_industries(2)
            logger.info(f"开始下载{len(industries)}个申万二级行业数据（akshare）...")
            akshare_download_all(level=2)

    # 同时下载大盘成交量数据
    logger.info("下载上证综指大盘数据（用于成交量过滤）...")
    download_and_save_market()


def cmd_analyze(level: str = "1", config: StageConfig | None = None) -> pd.DataFrame:
    """分析已有数据，输出阶段判断结果"""
    if config is None:
        config = StageConfig()

    lv = int(level)
    industries = get_industries(lv)
    level_name = "一" if lv == 1 else "二"
    load_csv = load_level1_csv if lv == 1 else (lambda c, n: load_level2_csv(c, n, lv))

    logger.info(f"加载{level_name}级行业本地数据并生成周K线...")
    industry_data = {}

    for code, name in industries.items():
        daily_df = load_csv(code, name)
        if daily_df is None or daily_df.empty:
            logger.warning(f"  {name}({code}) 无本地数据，跳过")
            continue

        weekly_df = generate_weekly_kline(daily_df)
        industry_data[code] = (name, weekly_df)
        logger.info(f"  {name}({code}): {len(daily_df)} 日K → {len(weekly_df)} 周K")

    if not industry_data:
        logger.error("无可用数据，请先运行 download 命令")
        return pd.DataFrame()

    logger.info(f"\n分析 {len(industry_data)} 个{level_name}级行业的趋势阶段...")
    summary = batch_analyze(industry_data, config)
    print_summary(summary, lv)

    # 保存分析结果
    if not summary.empty:
        out_dir = analysis_dir_for_level(lv)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "stage_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"分析结果已保存至 {summary_path}")

    return summary


def print_summary(summary: pd.DataFrame, level: int = 1):
    """在终端打印阶段汇总表"""
    if summary.empty:
        print("无分析结果")
        return

    stage_names = {1: "Stage 1 熊市", 2: "Stage 2 熊牛转换", 3: "Stage 3 牛市", 4: "Stage 4 牛熊转换"}
    level_name = "一" if level == 1 else "二"

    print("\n" + "=" * 80)
    print(f"申万{level_name}级行业趋势阶段分析（基于34周均线）")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="申万行业趋势阶段分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # download
    dl = subparsers.add_parser("download", help="下载行业数据")
    dl.add_argument("--level", default="1", choices=["1", "2", "all"],
                    help="行业级别: 1=一级, 2=二级, all=全部 (默认: 1)")
    dl.add_argument("--no-headless", action="store_true",
                    help="显示浏览器窗口（仅一级行业下载有效）")

    # analyze
    az = subparsers.add_parser("analyze", help="分析行业阶段")
    az.add_argument("--level", default="1", choices=["1", "2"],
                    help="行业级别: 1=一级, 2=二级 (默认: 1)")

    # all
    al = subparsers.add_parser("all", help="下载 + 分析")
    al.add_argument("--level", default="1", choices=["1", "2", "all"],
                    help="行业级别: 1=一级, 2=二级, all=全部 (默认: 1)")
    al.add_argument("--no-headless", action="store_true",
                    help="显示浏览器窗口")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "download":
        cmd_download(level=args.level, headless=not args.no_headless)
    elif args.command == "analyze":
        cmd_analyze(level=args.level)
    elif args.command == "all":
        cmd_download(level=args.level, headless=not args.no_headless)
        analyze_levels = ["1", "2"] if args.level == "all" else [args.level]
        for lv in analyze_levels:
            cmd_analyze(level=lv)


if __name__ == "__main__":
    main()
