"""行业轮动回测引擎 — 基于Weinstein阶段信号的行业轮动策略

策略逻辑:
- 买入信号: compute_signal_series() 产生 breakout_confirmed（确认突破）
- 卖出信号: compute_signal_series() 产生 breakdown_confirmed（确认崩盘）
- 仓位管理: 等权5仓位，每次买入当前总资产的20%
- 弱势替换: 持仓满5个时新信号出现，用收益率最低的换出
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.stage_analyzer import compute_signal_series, StageConfig


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
    start_date: str | None = None
    end_date: str | None = None

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


# ── 辅助函数 ──────────────────────────────────────────────


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


# ── 核心回测循环 ──────────────────────────────────────────


def run_backtest(
    industry_data: dict[str, tuple[str, pd.DataFrame]],
    bt_config: BacktestConfig | None = None,
    stage_config: StageConfig | None = None,
) -> BacktestResult:
    """执行完整回测

    Args:
        industry_data: {code: (name, weekly_df)} 与 app.py load_all_data() 格式一致
        bt_config: 回测参数
        stage_config: 阶段分析参数
    """
    if bt_config is None:
        bt_config = BacktestConfig()
    if stage_config is None:
        stage_config = StageConfig()

    signals_df, prices_df = prepare_all_signals(industry_data, stage_config)

    # 构建价格查找表: {date: {code: close}}
    price_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for _, row in prices_df.iterrows():
        d = row["date"]
        if d not in price_lookup:
            price_lookup[d] = {}
        price_lookup[d][row["code"]] = row["close"]

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

        # ── 第二步：处理买入信号 ──
        buy_candidates = []
        for code, name, signal in week_signals:
            if signal == "breakout_confirmed" and code not in positions:
                price = latest_prices.get(code)
                if price is not None and price > 0:
                    buy_candidates.append((code, name, price))

        if buy_candidates:
            available_slots = bt_config.max_positions - len(positions)

            if len(buy_candidates) <= available_slots:
                # 简单情况：直接买入所有候选
                for code, name, price in buy_candidates:
                    portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
                    buy_value = portfolio_value * bt_config.portion_pct
                    buy_value = min(buy_value, cash)
                    if buy_value < 1.0:
                        continue
                    pos = _execute_buy(code, name, date, price, buy_value, "breakout_confirmed", trades)
                    positions[code] = pos
                    cash -= buy_value
            else:
                # 满仓替换逻辑：需要在所有候选中挑选最强的
                # 收集所有候选（现有持仓 + 新买入候选）
                all_candidates: dict[str, dict] = {}

                # 现有持仓的收益率
                for code, pos in positions.items():
                    current_price = latest_prices.get(code, pos.entry_price)
                    all_candidates[code] = {
                        "type": "existing",
                        "return_pct": pos.return_pct(current_price),
                        "position": pos,
                    }

                # 新候选收益率为 0（刚进入）
                for code, name, price in buy_candidates:
                    all_candidates[code] = {
                        "type": "new",
                        "return_pct": 0.0,
                        "code": code,
                        "name": name,
                        "price": price,
                    }

                # 按收益率降序排列，保留前 max_positions 个
                sorted_items = sorted(
                    all_candidates.items(),
                    key=lambda x: x[1]["return_pct"],
                    reverse=True,
                )
                keep_codes = set(code for code, _ in sorted_items[: bt_config.max_positions])

                # 卖出不在保留集中的现有持仓
                codes_to_sell = [c for c in list(positions.keys()) if c not in keep_codes]
                for code in codes_to_sell:
                    price = latest_prices.get(code, positions[code].entry_price)
                    cash += _execute_sell(
                        positions[code], date, price, "replaced_weakest", trades
                    )
                    del positions[code]

                # 买入在保留集中的新候选
                for code, info in sorted_items[: bt_config.max_positions]:
                    if info["type"] == "new" and code in keep_codes:
                        portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
                        buy_value = portfolio_value * bt_config.portion_pct
                        buy_value = min(buy_value, cash)
                        if buy_value < 1.0:
                            continue
                        pos = _execute_buy(
                            info["code"], info["name"], date, info["price"],
                            buy_value, "breakout_confirmed", trades,
                        )
                        positions[pos.code] = pos
                        cash -= buy_value

        # ── 记录本周净值 ──
        portfolio_value = _get_portfolio_value(cash, positions, latest_prices)
        equity_records.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "n_positions": len(positions),
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
