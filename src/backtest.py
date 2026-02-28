"""行业轮动回测引擎 — 基于Weinstein阶段信号的行业轮动策略

策略逻辑:
- 分批建仓: approaching_breakout 试探买入30%, breakout_confirmed 加仓70%
- 假突破止损: breakout_failed 卖出试探仓
- 卖出信号: breakdown_confirmed（确认崩盘）
- 强制卖出: ATR波动率止损
- 仓位管理: 等权仓位，ATR弹性替换（仅确认突破触发）
- 行业分散: 同一级行业下最多持仓3个二级行业
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config.industries import get_level2_parent, is_blacklisted
from src.stage_analyzer import compute_signal_series, compute_consecutive_below_ma, StageConfig


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class Trade:
    """单笔交易记录"""

    date: pd.Timestamp
    code: str
    name: str
    action: str  # "buy" or "sell"
    price: float
    shares: float
    value: float
    reason: str  # "breakout_confirmed", "breakdown_confirmed", "replaced_weakest"


@dataclass
class Position:
    """当前持仓"""

    code: str
    name: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    is_partial: bool = False  # True = 仅30%试探仓（approaching_breakout）

    def market_value(self, current_price: float) -> float:
        return self.shares * current_price

    def return_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class BacktestConfig:
    """回测参数"""

    initial_capital: float = 10_000.0
    max_positions: int = 5
    max_same_parent: int = 3  # 同一级行业下最多持仓的二级行业数
    start_date: str | None = None
    end_date: str | None = None
    volume_filter: bool = False  # 是否启用大盘成交量过滤
    volume_percentile_threshold: float = 0.6  # 成交量百分位阈值（0.6 = 60%）

    @property
    def portion_pct(self) -> float:
        """每次买入比例 = 1 / max_positions"""
        return 1.0 / self.max_positions


@dataclass
class BacktestResult:
    """回测输出"""

    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: list[Trade] = field(default_factory=list)
    final_positions: list[Position] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)

    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_weeks: int = 0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    avg_holding_weeks: float = 0.0
    best_trade_name: str = ""
    best_trade_return: float = 0.0


# ── 信号预计算 ──────────────────────────────────────────────


def prepare_all_signals(
    industry_data: dict[str, tuple[str, pd.DataFrame]],
    config: StageConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """预计算所有行业的信号和价格数据

    Returns:
        (signals_df, prices_df):
        - signals_df: 仅包含有信号的行 (date, code, name, signal)
        - prices_df: 所有行业每周收盘价 (date, code, name, close)
    """
    if config is None:
        config = StageConfig()

    all_signals = []
    all_prices = []

    for code, (name, weekly_df) in industry_data.items():
        signal_df = compute_signal_series(weekly_df, config)
        if signal_df.empty:
            continue

        # 合并收盘价
        price_cols = weekly_df[["date", "close"]].copy()
        price_cols["code"] = code
        price_cols["name"] = name
        # 强制卖出指标
        price_cols["consecutive_below_ma"] = compute_consecutive_below_ma(weekly_df)
        price_cols["atr_stop"] = (
            weekly_df["close"].rolling(8, min_periods=8).min().shift(1)
            - 3.0 * weekly_df["atr14"]
        )
        all_prices.append(price_cols)

        # 仅保留有信号的行
        sig_rows = signal_df[signal_df["signal"].notna()].copy()
        if not sig_rows.empty:
            sig_rows["code"] = code
            sig_rows["name"] = name
            all_signals.append(sig_rows[["date", "code", "name", "signal"]])

    signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame(
        columns=["date", "code", "name", "signal"]
    )
    prices_df = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame(
        columns=["date", "code", "name", "close"]
    )

    return signals_df, prices_df


def compute_historical_atr_pct(
    industry_data: dict[str, tuple[str, pd.DataFrame]],
) -> dict[str, float]:
    """计算每个行业的历史ATR%中位数（股性/弹性指标）

    ATR% = atr14 / close，取全部历史的中位数作为行业特征弹性。
    值越大说明该行业历史波动越大、弹性越强、潜在上涨空间越大。
    """
    result = {}
    for code, (name, weekly_df) in industry_data.items():
        if "atr14" in weekly_df.columns:
            atr_pct = (weekly_df["atr14"] / weekly_df["close"]).dropna()
            result[code] = float(atr_pct.median()) if not atr_pct.empty else 0.0
        else:
            result[code] = 0.0
    return result


def save_atr_ranking(
    industry_data: dict[str, tuple[str, pd.DataFrame]],
    out_dir: Path,
) -> dict[str, float]:
    """计算并保存行业ATR%排名到 CSV 文件

    Returns:
        {code: atr_pct} 字典（同时保存到 out_dir/atr_ranking.csv）
    """
    atr_ranking = compute_historical_atr_pct(industry_data)
    rows = []
    for code, atr_pct in sorted(atr_ranking.items(), key=lambda x: x[1], reverse=True):
        name = industry_data[code][0] if code in industry_data else ""
        rows.append({"code": code, "name": name, "atr_pct": round(atr_pct, 6)})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "atr_ranking.csv", index=False)
    return atr_ranking


def load_atr_ranking(analysis_dir: Path) -> dict[str, float] | None:
    """从 CSV 文件加载预计算的ATR%排名

    Returns:
        {code: atr_pct} 字典，文件不存在时返回 None
    """
    csv_path = analysis_dir / "atr_ranking.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, dtype={"code": str})
    return dict(zip(df["code"], df["atr_pct"]))


# ── 辅助函数 ──────────────────────────────────────────────


def _count_same_parent(
    code: str,
    positions: dict[str, Position],
    parent_lookup: dict[str, str],
) -> int:
    """统计持仓中与 code 同属一个一级行业的二级行业数量"""
    parent = parent_lookup.get(code)
    if parent is None:
        return 0
    return sum(1 for c in positions if parent_lookup.get(c) == parent)


def _get_portfolio_value(
    cash: float,
    positions: dict[str, Position],
    prices: dict[str, float],
) -> float:
    """计算当前总资产 = 现金 + 所有持仓市值"""
    total = cash
    for code, pos in positions.items():
        price = prices.get(code, pos.entry_price)
        total += pos.market_value(price)
    return total


def _execute_sell(
    pos: Position,
    date: pd.Timestamp,
    price: float,
    reason: str,
    trades: list[Trade],
) -> float:
    """执行卖出，返回卖出所得资金"""
    sell_value = pos.shares * price
    trades.append(Trade(
        date=date,
        code=pos.code,
        name=pos.name,
        action="sell",
        price=price,
        shares=pos.shares,
        value=sell_value,
        reason=reason,
    ))
    return sell_value


def _execute_buy(
    code: str,
    name: str,
    date: pd.Timestamp,
    price: float,
    buy_value: float,
    reason: str,
    trades: list[Trade],
) -> Position:
    """执行买入，返回新持仓"""
    shares = buy_value / price
    trades.append(Trade(
        date=date,
        code=code,
        name=name,
        action="buy",
        price=price,
        shares=shares,
        value=buy_value,
        reason=reason,
    ))
    return Position(
        code=code,
        name=name,
        entry_date=date,
        entry_price=price,
        shares=shares,
    )


def _execute_topup(
    pos: Position,
    date: pd.Timestamp,
    price: float,
    buy_value: float,
    trades: list[Trade],
) -> float:
    """对已有试探仓加仓，更新加权平均成本价，返回买入金额"""
    new_shares = buy_value / price
    total_shares = pos.shares + new_shares
    pos.entry_price = (pos.shares * pos.entry_price + new_shares * price) / total_shares
    pos.shares = total_shares
    pos.is_partial = False
    trades.append(Trade(
        date=date,
        code=pos.code,
        name=pos.name,
        action="buy",
        price=price,
        shares=new_shares,
        value=buy_value,
        reason="breakout_confirmed",
    ))
    return buy_value


# ── 核心回测循环 ──────────────────────────────────────────


def _buy_with_atr_replacement(
    candidates: list[tuple[str, str, float]],
    positions: dict[str, Position],
    bt_config: BacktestConfig,
    cash: float,
    latest_prices: dict[str, float],
    atr_ranking: dict[str, float],
    parent_lookup: dict[str, str],
    date: pd.Timestamp,
    trades: list[Trade],
    reason: str,
    buy_pct: float = 1.0,
    partial: bool = False,
) -> float:
    """通用的买入+ATR替换逻辑

    有空余仓位时直接买入；仓位已满时将已持仓和新候选一起按ATR%排序，
    保留ATR%最高的行业组合。

    Args:
        candidates: [(code, name, price), ...] 买入候选
        buy_pct: 买入比例（approaching_breakout=0.30, breakout_confirmed=1.0）
        partial: 是否标记为试探仓

    Returns:
        更新后的现金余额
    """
    available_slots = bt_config.max_positions - len(positions)

    if len(candidates) <= available_slots:
        # 有空余仓位，直接买入
        for code, name, price in candidates:
            if (parent_lookup
                    and _count_same_parent(code, positions, parent_lookup) >= bt_config.max_same_parent):
                continue
            portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
            buy_value = portfolio_value * bt_config.portion_pct * buy_pct
            buy_value = min(buy_value, cash)
            if buy_value < 1.0:
                continue
            pos = _execute_buy(code, name, date, price, buy_value, reason, trades)
            pos.is_partial = partial
            positions[code] = pos
            cash -= buy_value
    else:
        # 仓位不足（含满仓）：按ATR%排序，贪心选择最优组合
        all_candidates: dict[str, dict] = {}

        for code, pos in positions.items():
            all_candidates[code] = {"type": "existing", "position": pos}

        for code, name, price in candidates:
            all_candidates[code] = {
                "type": "new", "code": code, "name": name, "price": price,
            }

        # 按历史ATR%降序排列（高弹性 = 大上涨空间优先）
        sorted_items = sorted(
            all_candidates.items(),
            key=lambda x: atr_ranking.get(x[0], 0.0),
            reverse=True,
        )

        # 贪心选择：遵守 max_positions 和 max_same_parent 约束
        keep_codes: set[str] = set()
        parent_count: dict[str, int] = {}
        for code, info in sorted_items:
            if len(keep_codes) >= bt_config.max_positions:
                break
            parent = parent_lookup.get(code)
            if parent and parent_count.get(parent, 0) >= bt_config.max_same_parent:
                continue
            keep_codes.add(code)
            if parent:
                parent_count[parent] = parent_count.get(parent, 0) + 1

        # 卖出被替换的持仓
        codes_to_sell = [c for c in list(positions.keys()) if c not in keep_codes]
        for code in codes_to_sell:
            price = latest_prices.get(code, positions[code].entry_price)
            cash += _execute_sell(
                positions[code], date, price, "replaced_weakest", trades
            )
            del positions[code]

        # 买入新选中的行业
        for code, info in sorted_items:
            if info["type"] == "new" and code in keep_codes:
                portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
                buy_value = portfolio_value * bt_config.portion_pct * buy_pct
                buy_value = min(buy_value, cash)
                if buy_value < 1.0:
                    continue
                pos = _execute_buy(
                    info["code"], info["name"], date, info["price"],
                    buy_value, reason, trades,
                )
                pos.is_partial = partial
                positions[pos.code] = pos
                cash -= buy_value

    return cash


def run_backtest(
    industry_data: dict[str, tuple[str, pd.DataFrame]],
    bt_config: BacktestConfig | None = None,
    stage_config: StageConfig | None = None,
    atr_ranking: dict[str, float] | None = None,
    volume_percentile_lookup: dict[pd.Timestamp, float] | None = None,
) -> BacktestResult:
    """执行完整回测

    Args:
        industry_data: {code: (name, weekly_df)} 与 app.py load_all_data() 格式一致
        bt_config: 回测参数
        stage_config: 阶段分析参数
        atr_ranking: 预计算的行业ATR%排名（None时自动计算）
        volume_percentile_lookup: 大盘成交量百分位查找表 {date: percentile}
    """
    if bt_config is None:
        bt_config = BacktestConfig()
    if stage_config is None:
        stage_config = StageConfig()

    signals_df, prices_df = prepare_all_signals(industry_data, stage_config)

    # 行业历史弹性（ATR%）：优先使用传入的预计算数据
    if atr_ranking is None:
        atr_ranking = compute_historical_atr_pct(industry_data)

    # 一级行业归属映射（仅二级行业有效）
    atr_ranking = compute_historical_atr_pct(industry_data)
    parent_lookup: dict[str, str] = {}
    for code in industry_data:
        parent = get_level2_parent(code)
        if parent is not None:
            parent_lookup[code] = parent

    # 构建价格查找表: {date: {code: close}}
    price_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    # 强制卖出指标查找表
    below_ma_lookup: dict[pd.Timestamp, dict[str, int]] = {}
    atr_stop_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for _, row in prices_df.iterrows():
        d = row["date"]
        if d not in price_lookup:
            price_lookup[d] = {}
            below_ma_lookup[d] = {}
            atr_stop_lookup[d] = {}
        price_lookup[d][row["code"]] = row["close"]
        below_ma_lookup[d][row["code"]] = int(row["consecutive_below_ma"])
        if pd.notna(row["atr_stop"]):
            atr_stop_lookup[d][row["code"]] = row["atr_stop"]

    # 构建信号查找表: {date: [(code, name, signal), ...]}
    signal_lookup: dict[pd.Timestamp, list[tuple[str, str, str]]] = {}
    for _, row in signals_df.iterrows():
        d = row["date"]
        if d not in signal_lookup:
            signal_lookup[d] = []
        signal_lookup[d].append((row["code"], row["name"], row["signal"]))

    # 获取所有日期并排序
    all_dates = sorted(price_lookup.keys())

    # 应用日期过滤
    if bt_config.start_date:
        start = pd.Timestamp(bt_config.start_date)
        all_dates = [d for d in all_dates if d >= start]
    if bt_config.end_date:
        end = pd.Timestamp(bt_config.end_date)
        all_dates = [d for d in all_dates if d <= end]

    if not all_dates:
        return BacktestResult(config=bt_config)

    # 构建成交量百分位对齐查找表（用 merge_asof 对齐到回测日期）
    vol_pct_aligned: dict[pd.Timestamp, float] = {}
    if bt_config.volume_filter and volume_percentile_lookup:
        vol_df = pd.DataFrame(
            {"date": list(volume_percentile_lookup.keys()),
             "vol_pct": list(volume_percentile_lookup.values())}
        ).sort_values("date")
        bt_dates_df = pd.DataFrame({"date": all_dates})
        aligned = pd.merge_asof(bt_dates_df, vol_df, on="date")
        vol_pct_aligned = dict(zip(aligned["date"], aligned["vol_pct"]))

    # 初始化
    cash = bt_config.initial_capital
    positions: dict[str, Position] = {}  # {code: Position}
    trades: list[Trade] = []
    equity_records = []

    # 用于追踪最新价格（处理某些周个别行业无数据的情况）
    latest_prices: dict[str, float] = {}

    for date in all_dates:
        # 更新价格
        week_prices = price_lookup.get(date, {})
        latest_prices.update(week_prices)

        # 获取本周信号
        week_signals = signal_lookup.get(date, [])

        # ── 第一步：处理卖出信号 ──
        sell_codes = set()
        for code, name, signal in week_signals:
            if signal == "breakdown_confirmed" and code in positions:
                price = latest_prices.get(code)
                if price is not None:
                    cash += _execute_sell(
                        positions[code], date, price, "breakdown_confirmed", trades
                    )
                    sell_codes.add(code)

        for code in sell_codes:
            del positions[code]

        # ── 第二步：强制卖出检查 ──
        force_sell_codes = set()
        for code in list(positions.keys()):
            if code in sell_codes:
                continue

            price = latest_prices.get(code)
            if price is None:
                continue

            # ATR波动率止损（收盘价 < 前8周最低收盘价 - 3×ATR14）
            atr_stop = atr_stop_lookup.get(date, {}).get(code)
            if atr_stop is not None and price < atr_stop:
                cash += _execute_sell(
                    positions[code], date, price, "force_sell_atr_stop", trades
                )
                force_sell_codes.add(code)

        for code in force_sell_codes:
            del positions[code]

        # ── 第三步：试探仓假突破检查（连续2周收盘低于MA34） ──
        failed_sell_codes = set()
        for code in list(positions.keys()):
            if not positions[code].is_partial:
                continue
            below_weeks = below_ma_lookup.get(date, {}).get(code, 0)
            if below_weeks >= 2:
                price = latest_prices.get(code)
                if price is not None:
                    cash += _execute_sell(
                        positions[code], date, price, "breakout_failed", trades
                    )
                    failed_sell_codes.add(code)

        for code in failed_sell_codes:
            del positions[code]

        # ── 成交量过滤：大盘量能不足时跳过所有买入 ──
        vol_pct = vol_pct_aligned.get(date)
        volume_ok = True
        if bt_config.volume_filter and vol_pct is not None:
            if vol_pct < bt_config.volume_percentile_threshold:
                volume_ok = False

        # ── 第四步：approaching_breakout 买入30%试探仓 ──
        if volume_ok:
            ab_candidates = []
            for code, name, signal in week_signals:
                if signal == "approaching_breakout" and code not in positions:
                    if is_blacklisted(code):
                        continue
                    price = latest_prices.get(code)
                    if price is not None and price > 0:
                        ab_candidates.append((code, name, price))

            if ab_candidates:
                cash = _buy_with_atr_replacement(
                    candidates=ab_candidates,
                    positions=positions,
                    bt_config=bt_config,
                    cash=cash,
                    latest_prices=latest_prices,
                    atr_ranking=atr_ranking,
                    parent_lookup=parent_lookup,
                    date=date,
                    trades=trades,
                    reason="approaching_breakout",
                    buy_pct=0.30,
                    partial=True,
                )

        # ── 第五步：breakout_confirmed 加仓/全仓买入 ──
        if volume_ok:
            # 5a. 加仓已有试探仓（不需要新仓位）
            for code, name, signal in week_signals:
                if signal == "breakout_confirmed" and code in positions and positions[code].is_partial:
                    price = latest_prices.get(code)
                    if price is None or price <= 0:
                        continue
                    portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
                    topup_value = portfolio_value * bt_config.portion_pct * 0.70
                    topup_value = min(topup_value, cash)
                    if topup_value < 1.0:
                        continue
                    _execute_topup(positions[code], date, price, topup_value, trades)
                    cash -= topup_value

            # 5b. 全新买入（无试探仓的行业，100%建仓，满仓时ATR替换）
            buy_candidates = []
            for code, name, signal in week_signals:
                if signal == "breakout_confirmed" and code not in positions:
                    if is_blacklisted(code):
                        continue
                    price = latest_prices.get(code)
                    if price is not None and price > 0:
                        buy_candidates.append((code, name, price))

            if buy_candidates:
                cash = _buy_with_atr_replacement(
                    candidates=buy_candidates,
                    positions=positions,
                    bt_config=bt_config,
                    cash=cash,
                    latest_prices=latest_prices,
                    atr_ranking=atr_ranking,
                    parent_lookup=parent_lookup,
                    date=date,
                    trades=trades,
                    reason="breakout_confirmed",
                    buy_pct=1.0,
                    partial=False,
                )

        # ── 记录本周净值 ──
        portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
        equity_records.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "n_positions": len(positions),
            "volume_percentile": vol_pct_aligned.get(date),
        })

    equity_curve = pd.DataFrame(equity_records)

    result = BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        final_positions=list(positions.values()),
        config=bt_config,
    )

    return compute_metrics(result)


# ── 绩效指标计算 ──────────────────────────────────────────


def compute_metrics(result: BacktestResult) -> BacktestResult:
    """计算回测绩效指标"""
    ec = result.equity_curve
    if ec.empty or len(ec) < 2:
        return result

    start_value = ec["portfolio_value"].iloc[0]
    end_value = ec["portfolio_value"].iloc[-1]

    # 总收益率
    result.total_return = (end_value / start_value) - 1

    # 年化收益率
    n_weeks = len(ec)
    n_years = n_weeks / 52.0
    if n_years > 0 and (1 + result.total_return) > 0:
        result.annualized_return = (1 + result.total_return) ** (1 / n_years) - 1
    else:
        result.annualized_return = 0.0

    # 最大回撤
    running_max = ec["portfolio_value"].cummax()
    drawdown = (ec["portfolio_value"] - running_max) / running_max
    result.max_drawdown = float(drawdown.min())

    # 最大回撤持续周数
    in_drawdown = ec["portfolio_value"] < running_max
    groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(groups).sum()
    result.max_drawdown_duration_weeks = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    # 夏普比率（周收益年化）
    weekly_returns = ec["portfolio_value"].pct_change().dropna()
    if len(weekly_returns) > 1 and weekly_returns.std() > 0:
        result.sharpe_ratio = float(
            (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52)
        )
    else:
        result.sharpe_ratio = 0.0

    # 胜率（配对买卖）
    buy_map: dict[str, list[Trade]] = {}
    for t in result.trades:
        if t.action == "buy":
            buy_map.setdefault(t.code, []).append(t)

    wins = 0
    total_roundtrips = 0
    holding_weeks_list = []
    win_returns = []
    loss_returns = []
    best_return = -float("inf")
    best_name = ""
    for t in result.trades:
        if t.action == "sell":
            buys = [b for b in buy_map.get(t.code, []) if b.date <= t.date]
            if buys:
                entry = buys[-1]
                ret = (t.price - entry.price) / entry.price
                if t.price > entry.price:
                    wins += 1
                    win_returns.append(ret)
                else:
                    loss_returns.append(ret)
                total_roundtrips += 1
                weeks_held = (t.date - entry.date).days / 7
                holding_weeks_list.append(weeks_held)
                if ret > best_return:
                    best_return = ret
                    best_name = t.name

    result.win_rate = wins / total_roundtrips if total_roundtrips > 0 else 0.0
    avg_win = float(np.mean(win_returns)) if win_returns else 0.0
    avg_loss = abs(float(np.mean(loss_returns))) if loss_returns else 0.0
    result.profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    result.best_trade_name = best_name
    result.best_trade_return = best_return if best_return > -float("inf") else 0.0
    result.total_trades = len(result.trades)
    result.avg_holding_weeks = float(np.mean(holding_weeks_list)) if holding_weeks_list else 0.0

    return result


# ── 基准指数计算 ──────────────────────────────────────────


def compute_benchmark(
    industry_data: dict[str, tuple[str, pd.DataFrame]],
    initial_capital: float = 10_000.0,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """计算31行业等权基准指数

    每个行业的收盘价标准化为1.0（起始日），取所有行业均值，
    再缩放到初始资金规模。

    Returns:
        DataFrame with columns: date, benchmark_value
    """
    all_series = []

    for code, (name, weekly_df) in industry_data.items():
        df = weekly_df[["date", "close"]].dropna().copy()
        if df.empty:
            continue

        if start_date is not None:
            df = df[df["date"] >= start_date]
        if end_date is not None:
            df = df[df["date"] <= end_date]

        if df.empty:
            continue

        # 标准化到1.0
        base_price = df["close"].iloc[0]
        if base_price > 0:
            df["norm_close"] = df["close"] / base_price
            df = df[["date", "norm_close"]].rename(columns={"norm_close": code})
            all_series.append(df.set_index("date"))

    if not all_series:
        return pd.DataFrame(columns=["date", "benchmark_value"])

    merged = pd.concat(all_series, axis=1)
    # 前向填充缺失值（某些行业在某些周无数据）
    merged = merged.ffill()

    benchmark = merged.mean(axis=1) * initial_capital
    result = benchmark.reset_index()
    result.columns = ["date", "benchmark_value"]

    return result
