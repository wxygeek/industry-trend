"""温斯坦（Weinstein）四阶段判断算法 — 状态机版本

基于34周均线，采用状态机模型进行阶段判断。
核心逻辑：趋势方向（均线斜率）+ 相对位置（价格与均线距离）+ 动能确认（成交量）

状态转换路径（只允许合法跳转）：
    1(熊市) → 2(熊牛转换) → 3(牛市) → 4(牛熊转换) → 1(熊市)
    2 可退回 1（底部失败），4 可退回 3（假跌破修复）

量化规则体系参考：
- 状态3(牛市): Price > MA34*(1+alpha), Slope > 0, 近4周低点未破MA
- 状态1(熊市): Price < MA34*(1-alpha), Slope < 0, 近4周高点未破MA
- 状态2/4(转换): |Slope| < flat_threshold, 穿越频率>=2, |Dev| < alpha
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StageConfig:
    """阶段判断的可配置阈值参数

    参数基于 improvement.md 的量化建议设定，适配 A 股行业指数的波动特征。
    """

    # 价格偏离MA的显著性阈值（improvement.md 建议 3%~5%，A股行业指数取5%）
    price_position_threshold: float = 0.05

    # MA斜率"走平"的阈值（4周内变化<1.5%视为走平）
    slope_flat_threshold: float = 0.015

    # 穿越频率统计窗口（周）
    crossover_window: int = 8

    # 高频穿越次数阈值（>=2次即视为"反复争夺"）
    crossover_threshold: int = 2

    # 短期成交量窗口（周）
    short_volume_window: int = 4

    # 长期成交量窗口（周）— 用于计算成交量基准
    long_volume_window: int = 10

    # 近期价格位置的回看周数
    recent_weeks: int = 8

    # 支撑/压力确认窗口（周）
    support_window: int = 4

    # 支撑/压力容忍度（允许瞬间刺破2%）
    support_tolerance: float = 0.02

    # 状态突破时的放量确认阈值（VolRatio >= 1.5 确认有效突破）
    breakout_volume_threshold: float = 1.5

    # 牛市中连续N周跌破MA34则不视为牛市
    bull_break_weeks: int = 2

    # 连续N周在MA34下方 + 斜率下行 → 强制熊市
    bear_confirm_weeks: int = 4

    # ATR 自适应系数（固定阈值 / 全行业 ATR% 中位数 ≈ 0.047）
    atr_alpha_multiplier: float = 1.0       # price_position_threshold ≈ 1.0 * atr_pct
    atr_slope_multiplier: float = 0.3       # slope_flat_threshold ≈ 0.3 * atr_pct
    atr_support_multiplier: float = 0.4     # support_tolerance ≈ 0.4 * atr_pct

    # 持续性恢复周数（用于宽松的 Stage 4→3 / Stage 2→3 路径）
    recovery_sustain_weeks: int = 8


@dataclass
class StageResult:
    """单个行业的阶段判断结果"""

    code: str
    name: str
    stage: int  # 1, 2, 3, 4
    confidence: float  # 0.0 - 1.0
    price_position: float
    ma_slope: float
    crossover_count: int
    volume_ratio: float
    close: float
    ma34: float
    stage_label: str = ""

    def __post_init__(self):
        labels = {1: "熊市", 2: "熊牛转换", 3: "牛市", 4: "牛熊转换"}
        self.stage_label = labels.get(self.stage, "未知")


# ── 特征计算函数 ──────────────────────────────────────────────


def compute_price_position(weekly_df: pd.DataFrame) -> pd.Series:
    """计算价格偏离MA34的比例: (close - ma34) / ma34"""
    return (weekly_df["close"] - weekly_df["ma34"]) / weekly_df["ma34"]


def detect_crossovers(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """检测价格穿越MA34的事件"""
    df = weekly_df.copy()
    above = df["close"] > df["ma34"]
    prev_above = pd.Series(False, index=df.index)
    prev_above.iloc[1:] = above.iloc[:-1].values

    df["cross_above"] = (~prev_above) & above
    df["cross_below"] = prev_above & (~above)
    df["crossover"] = df["cross_above"] | df["cross_below"]
    return df


def count_crossovers(weekly_df: pd.DataFrame, window: int = 8) -> pd.Series:
    """统计滚动窗口内的穿越次数"""
    df = detect_crossovers(weekly_df)
    return df["crossover"].rolling(window=window, min_periods=1).sum()


def compute_volume_ratio(
    weekly_df: pd.DataFrame, short_window: int = 4, long_window: int = 10
) -> pd.Series:
    """计算成交量比: 短期均量 / 长期均量"""
    short_avg = weekly_df["volume"].rolling(window=short_window, min_periods=1).mean()
    long_avg = weekly_df["volume"].rolling(window=long_window, min_periods=1).mean()
    return short_avg / long_avg


def compute_consecutive_below_ma(weekly_df: pd.DataFrame) -> pd.Series:
    """计算连续收盘价低于MA34的周数"""
    below = (weekly_df["close"] < weekly_df["ma34"]).astype(int)
    groups = (~below.astype(bool)).cumsum()
    return below.groupby(groups).cumsum()


def compute_consecutive_above_ma(weekly_df: pd.DataFrame) -> pd.Series:
    """计算连续收盘价高于MA34的周数"""
    above = (weekly_df["close"] > weekly_df["ma34"]).astype(int)
    groups = (~above.astype(bool)).cumsum()
    return above.groupby(groups).cumsum()


def compute_support_hold(weekly_df: pd.DataFrame, window: int = 4, tolerance=0.02) -> pd.Series:
    """检查近N周最低点是否守住MA34支撑

    条件: Min(Low_{t-window..t}) > MA34 * (1 - tolerance)

    Args:
        tolerance: 标量或 Series（逐行不同阈值）
    """
    rolling_low = weekly_df["low"].rolling(window=window, min_periods=1).min()
    threshold = weekly_df["ma34"] * (1 - tolerance)
    return rolling_low > threshold


def compute_resistance_hold(weekly_df: pd.DataFrame, window: int = 4, tolerance=0.02) -> pd.Series:
    """检查近N周最高点是否被MA34压力压制

    条件: Max(High_{t-window..t}) < MA34 * (1 + tolerance)

    Args:
        tolerance: 标量或 Series（逐行不同阈值）
    """
    rolling_high = weekly_df["high"].rolling(window=window, min_periods=1).max()
    threshold = weekly_df["ma34"] * (1 + tolerance)
    return rolling_high < threshold


# ── 状态评分函数 ──────────────────────────────────────────────


def _score_bull(price_pos, ma_slope, cross_count, volume_ratio, support_held, config,
                alpha=None, slope_flat=None):
    """评估牛市(Stage 3)的匹配度"""
    alpha = alpha if alpha is not None else config.price_position_threshold
    slope_flat = slope_flat if slope_flat is not None else config.slope_flat_threshold
    score = 0.0
    if price_pos > alpha:
        score += 0.30
    if ma_slope > slope_flat:
        score += 0.25
    if support_held:
        score += 0.20
    if cross_count < config.crossover_threshold:
        score += 0.15
    if volume_ratio >= 1.0:
        score += 0.10
    return score


def _score_bear(price_pos, ma_slope, cross_count, volume_ratio, resistance_held, config,
                alpha=None, slope_flat=None):
    """评估熊市(Stage 1)的匹配度"""
    alpha = alpha if alpha is not None else config.price_position_threshold
    slope_flat = slope_flat if slope_flat is not None else config.slope_flat_threshold
    score = 0.0
    if price_pos < -alpha:
        score += 0.30
    if ma_slope < -slope_flat:
        score += 0.25
    if resistance_held:
        score += 0.20
    if cross_count < config.crossover_threshold:
        score += 0.15
    if volume_ratio < 1.0:
        score += 0.10
    return score


def _score_bottom_transition(price_pos, ma_slope, cross_count, volume_ratio, config,
                             alpha=None, slope_flat=None):
    """评估熊牛转换(Stage 2)的匹配度"""
    alpha = alpha if alpha is not None else config.price_position_threshold
    slope_flat = slope_flat if slope_flat is not None else config.slope_flat_threshold
    score = 0.0
    if abs(ma_slope) < slope_flat:
        score += 0.25
    if cross_count >= config.crossover_threshold:
        score += 0.30
    if abs(price_pos) < alpha:
        score += 0.20
    if volume_ratio > 0.8:
        score += 0.15
    if ma_slope > -slope_flat:
        score += 0.10
    return score


def _score_top_transition(price_pos, ma_slope, cross_count, volume_ratio, config,
                          alpha=None, slope_flat=None):
    """评估牛熊转换(Stage 4)的匹配度"""
    alpha = alpha if alpha is not None else config.price_position_threshold
    slope_flat = slope_flat if slope_flat is not None else config.slope_flat_threshold
    score = 0.0
    if abs(ma_slope) < slope_flat:
        score += 0.25
    if cross_count >= config.crossover_threshold:
        score += 0.30
    if abs(price_pos) < alpha:
        score += 0.20
    if volume_ratio < 1.2:
        score += 0.15
    if ma_slope < slope_flat:
        score += 0.10
    return score


# ── 核心：状态机分类 ──────────────────────────────────────────


def classify_stage_stateful(
    price_pos: float,
    ma_slope: float,
    cross_count: int,
    volume_ratio: float,
    support_held: bool,
    resistance_held: bool,
    config: StageConfig,
    previous_state: int = 0,
    consecutive_below_ma: int = 0,
    consecutive_above_ma: int = 0,
    adaptive_alpha: float | None = None,
    adaptive_slope_flat: float | None = None,
) -> tuple[int, float]:
    """基于状态机的阶段判断

    与原版纯评分不同，本方法引入 previous_state，只允许合法的状态转换路径。
    转换路径: 1→2→3→4→1（正常循环），2→1（底部失败），4→3（假跌破修复）

    Args:
        price_pos: 价格偏离度 (close - ma34) / ma34
        ma_slope: MA34斜率 (ma34_t - ma34_{t-4}) / ma34_{t-4}
        cross_count: 近N周穿越次数
        volume_ratio: 短期/长期成交量比
        support_held: 近4周低点是否守住MA34支撑
        resistance_held: 近4周高点是否被MA34压制
        config: 阈值配置
        previous_state: 上一期的状态（0=未知/初始化）
        consecutive_below_ma: 连续收盘价低于MA34的周数
        consecutive_above_ma: 连续收盘价高于MA34的周数
        adaptive_alpha: 当前行的自适应 alpha 阈值（None 时用 config 固定值）
        adaptive_slope_flat: 当前行的自适应 slope_flat 阈值（None 时用 config 固定值）

    Returns:
        (stage, confidence) 元组
    """
    alpha = adaptive_alpha if adaptive_alpha is not None else config.price_position_threshold
    slope_flat = adaptive_slope_flat if adaptive_slope_flat is not None else config.slope_flat_threshold

    # ── 硬性规则 ──
    is_slope_neg = ma_slope < -slope_flat

    # 连续4周跌破MA + 斜率下行 → 强制熊市（无论前一状态）
    if consecutive_below_ma >= config.bear_confirm_weeks and is_slope_neg:
        return 1, 0.90

    # ── 计算各状态评分 ──
    s_bull = _score_bull(price_pos, ma_slope, cross_count, volume_ratio, support_held, config,
                         alpha=alpha, slope_flat=slope_flat)
    s_bear = _score_bear(price_pos, ma_slope, cross_count, volume_ratio, resistance_held, config,
                         alpha=alpha, slope_flat=slope_flat)
    s_bottom = _score_bottom_transition(price_pos, ma_slope, cross_count, volume_ratio, config,
                                        alpha=alpha, slope_flat=slope_flat)
    s_top = _score_top_transition(price_pos, ma_slope, cross_count, volume_ratio, config,
                                  alpha=alpha, slope_flat=slope_flat)

    scores = {1: s_bear, 2: s_bottom, 3: s_bull, 4: s_top}

    # 连续2周跌破MA → 不可能在牛市
    if consecutive_below_ma >= config.bull_break_weeks:
        scores[3] = 0.0

    # ── 状态机转换逻辑 ──
    if previous_state == 0:
        # 初始状态：选评分最高
        state = max(scores, key=scores.get)
    elif previous_state == 1:
        # 熊市 → 只能转到熊牛转换(2)，或维持(1)
        if scores[2] > scores[1] and scores[2] > 0.3:
            state = 2
        else:
            state = 1
    elif previous_state == 2:
        # 熊牛转换 → 可突破到牛市(3)，或回落到熊市(1)，或维持(2)
        # 突破条件：价格站上均线 + 均线拐头向上 + 放量确认
        bull_breakout = (
            price_pos > alpha
            and ma_slope > slope_flat
            and volume_ratio >= config.breakout_volume_threshold
        )
        # 失败回落条件：价格跌破均线 + 均线恢复下行
        bear_fallback = (
            price_pos < -alpha
            and ma_slope < -slope_flat
        )
        # 宽松持续性突破：连续站稳 N 周 + 价格在均线上方 + 支撑守住
        sustained_breakout = (
            consecutive_above_ma >= config.recovery_sustain_weeks
            and price_pos > alpha
            and support_held
        )
        if bull_breakout:
            state = 3
        elif bear_fallback:
            state = 1
        elif sustained_breakout:
            state = 3
        else:
            # 宽松突破：评分显著优于底部转换，但仍需最低放量门槛
            if (scores[3] > scores[2] * 1.5
                    and price_pos > alpha
                    and ma_slope > slope_flat
                    and volume_ratio >= 1.2):
                state = 3
            else:
                state = 2
    elif previous_state == 3:
        # 牛市 → 可转到牛熊转换(4)，或维持(3)
        if scores[4] > scores[3] and scores[4] > 0.3:
            state = 4
        else:
            state = 3
    elif previous_state == 4:
        # 牛熊转换 → 可崩盘到熊市(1)，或修复回牛市(3)，或维持(4)
        # 崩盘条件：价格跌破均线 + 均线拐头向下
        bear_breakdown = (
            price_pos < -alpha
            and ma_slope < -slope_flat
        )
        # 修复条件：价格重新站上均线 + 均线恢复上行 + 放量
        bull_recovery = (
            price_pos > alpha
            and ma_slope > slope_flat
            and volume_ratio >= config.breakout_volume_threshold
        )
        # 宽松持续性恢复：连续站稳 N 周 + 价格显著在均线上方 + 支撑守住
        sustained_recovery = (
            consecutive_above_ma >= config.recovery_sustain_weeks
            and price_pos > alpha
            and support_held
        )
        if bear_breakdown:
            state = 1
        elif bull_recovery:
            state = 3
        elif sustained_recovery:
            state = 3
        else:
            # 宽松崩盘：评分显著偏熊
            if scores[1] > scores[4] * 1.3 and price_pos < -alpha:
                state = 1
            else:
                state = 4
    else:
        state = max(scores, key=scores.get)

    # ── 置信度计算 ──
    total = sum(scores.values())
    if total > 0:
        confidence = scores[state] / total
    else:
        confidence = 0.25
    confidence = min(max(confidence, 0.0), 1.0)

    return state, round(confidence, 3)


# ── 兼容接口：无状态机的单次分类（用于初始化或单帧场景） ──


def classify_stage(
    price_pos: float,
    ma_slope: float,
    cross_count: int,
    volume_ratio: float,
    recent_positions: pd.Series,
    config: StageConfig,
    consecutive_below_ma: int = 0,
) -> tuple[int, float]:
    """无状态机的单次分类（兼容旧接口）

    内部委托给 classify_stage_stateful，previous_state=0（自由选择最佳匹配）。
    """
    return classify_stage_stateful(
        price_pos=price_pos,
        ma_slope=ma_slope,
        cross_count=cross_count,
        volume_ratio=volume_ratio,
        support_held=True,   # 无具体数据时默认假设
        resistance_held=True,
        config=config,
        previous_state=0,
        consecutive_below_ma=consecutive_below_ma,
        consecutive_above_ma=0,
    )


# ── 特征预计算 ──────────────────────────────────────────────


def _prepare_features(df: pd.DataFrame, config: StageConfig) -> pd.DataFrame:
    """为 DataFrame 计算所有需要的特征列（含 ATR 自适应阈值）"""
    df = df.copy()
    df["price_position"] = compute_price_position(df)
    df["crossover_count"] = count_crossovers(df, window=config.crossover_window)
    df["volume_ratio"] = compute_volume_ratio(
        df, short_window=config.short_volume_window, long_window=config.long_volume_window
    )
    df["consecutive_below_ma"] = compute_consecutive_below_ma(df)
    df["consecutive_above_ma"] = compute_consecutive_above_ma(df)

    # ATR 自适应阈值（atr14 不存在时 fallback 到固定值）
    if "atr14" in df.columns:
        df["atr_pct"] = df["atr14"] / df["ma34"]
        df["adaptive_alpha"] = (config.atr_alpha_multiplier * df["atr_pct"]).fillna(config.price_position_threshold)
        df["adaptive_slope_flat"] = (config.atr_slope_multiplier * df["atr_pct"]).fillna(config.slope_flat_threshold)
        df["adaptive_support_tol"] = (config.atr_support_multiplier * df["atr_pct"]).fillna(config.support_tolerance)
    else:
        df["atr_pct"] = np.nan
        df["adaptive_alpha"] = config.price_position_threshold
        df["adaptive_slope_flat"] = config.slope_flat_threshold
        df["adaptive_support_tol"] = config.support_tolerance

    df["support_held"] = compute_support_hold(
        df, window=config.support_window, tolerance=df["adaptive_support_tol"]
    )
    df["resistance_held"] = compute_resistance_hold(
        df, window=config.support_window, tolerance=df["adaptive_support_tol"]
    )
    return df


# ── 单个行业分析 ──────────────────────────────────────────────


def analyze_industry(
    weekly_df: pd.DataFrame,
    code: str,
    name: str,
    config: StageConfig | None = None,
) -> StageResult:
    """分析单个行业的当前趋势阶段

    使用状态机逐周推演，取最后一周的状态作为当前结果。
    """
    if config is None:
        config = StageConfig()

    df = weekly_df.dropna(subset=["ma34", "ma_slope"]).copy()
    if df.empty:
        return StageResult(
            code=code, name=name, stage=0, confidence=0.0,
            price_position=0.0, ma_slope=0.0, crossover_count=0,
            volume_ratio=0.0, close=0.0, ma34=0.0,
        )

    df = _prepare_features(df, config)

    # 状态机逐周推演
    prev_state = 0
    final_stage = 0
    final_confidence = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        stage, confidence = classify_stage_stateful(
            price_pos=row["price_position"],
            ma_slope=row["ma_slope"],
            cross_count=int(row["crossover_count"]),
            volume_ratio=row["volume_ratio"],
            support_held=bool(row["support_held"]),
            resistance_held=bool(row["resistance_held"]),
            config=config,
            previous_state=prev_state,
            consecutive_below_ma=int(row["consecutive_below_ma"]),
            consecutive_above_ma=int(row["consecutive_above_ma"]),
            adaptive_alpha=float(row["adaptive_alpha"]),
            adaptive_slope_flat=float(row["adaptive_slope_flat"]),
        )
        prev_state = stage
        final_stage = stage
        final_confidence = confidence

    latest = df.iloc[-1]
    return StageResult(
        code=code,
        name=name,
        stage=final_stage,
        confidence=final_confidence,
        price_position=round(latest["price_position"], 4),
        ma_slope=round(latest["ma_slope"], 4),
        crossover_count=int(latest["crossover_count"]),
        volume_ratio=round(latest["volume_ratio"], 2),
        close=round(latest["close"], 2),
        ma34=round(latest["ma34"], 2),
    )


# ── 阶段转换检测 ──────────────────────────────────────────────


def detect_stage_transitions(
    weekly_df: pd.DataFrame, config: StageConfig | None = None
) -> pd.DataFrame:
    """检测阶段转换事件（基于状态机）

    逐周推演状态机，记录状态变化的时间点。
    """
    if config is None:
        config = StageConfig()

    df = weekly_df.dropna(subset=["ma34", "ma_slope"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "prev_stage", "new_stage"])

    df = _prepare_features(df, config)

    prev_state = 0
    transitions = []

    for i in range(len(df)):
        row = df.iloc[i]
        stage, _ = classify_stage_stateful(
            price_pos=row["price_position"],
            ma_slope=row["ma_slope"],
            cross_count=int(row["crossover_count"]),
            volume_ratio=row["volume_ratio"],
            support_held=bool(row["support_held"]),
            resistance_held=bool(row["resistance_held"]),
            config=config,
            previous_state=prev_state,
            consecutive_below_ma=int(row["consecutive_below_ma"]),
            consecutive_above_ma=int(row["consecutive_above_ma"]),
            adaptive_alpha=float(row["adaptive_alpha"]),
            adaptive_slope_flat=float(row["adaptive_slope_flat"]),
        )
        if prev_state != 0 and stage != prev_state:
            transitions.append({
                "date": row["date"],
                "prev_stage": prev_state,
                "new_stage": stage,
            })
        prev_state = stage

    return pd.DataFrame(transitions)


# ── 计算逐周阶段序列（供K线图着色使用） ──────────────────────


def compute_stage_series(
    weekly_df: pd.DataFrame, config: StageConfig | None = None
) -> pd.Series:
    """计算每周的阶段（基于状态机），返回 stage Series

    适用于K线图的阶段背景色标注。
    """
    if config is None:
        config = StageConfig()

    df = weekly_df.dropna(subset=["ma34", "ma_slope"]).copy()
    if df.empty:
        return pd.Series(dtype=int)

    df = _prepare_features(df, config)

    prev_state = 0
    stages = []
    for i in range(len(df)):
        row = df.iloc[i]
        stage, _ = classify_stage_stateful(
            price_pos=row["price_position"],
            ma_slope=row["ma_slope"],
            cross_count=int(row["crossover_count"]),
            volume_ratio=row["volume_ratio"],
            support_held=bool(row["support_held"]),
            resistance_held=bool(row["resistance_held"]),
            config=config,
            previous_state=prev_state,
            consecutive_below_ma=int(row["consecutive_below_ma"]),
            consecutive_above_ma=int(row["consecutive_above_ma"]),
            adaptive_alpha=float(row["adaptive_alpha"]),
            adaptive_slope_flat=float(row["adaptive_slope_flat"]),
        )
        stages.append(stage)
        prev_state = stage

    return pd.Series(stages, index=df.index)


# ── 细粒度交易信号检测 ──────────────────────────────────────


# 信号分类：利好（标注在K线下方）/ 利空（标注在K线上方）
BULLISH_SIGNALS = {"approaching_breakout", "breakout_confirmed", "breakdown_failed", "bull_recovery"}
BEARISH_SIGNALS = {"approaching_breakdown", "breakdown_confirmed", "breakout_failed", "bear_fallback"}

SIGNAL_LABELS = {
    "approaching_breakout": "接近突破",
    "breakout_confirmed": "确认突破",
    "breakout_failed": "假突破",
    "bear_fallback": "回落熊市",
    "approaching_breakdown": "接近崩盘",
    "breakdown_confirmed": "确认崩盘",
    "breakdown_failed": "假崩盘",
    "bull_recovery": "修复回牛",
}


def compute_signal_series(
    weekly_df: pd.DataFrame, config: StageConfig | None = None
) -> pd.DataFrame:
    """计算每周的阶段和交易信号

    在 Stage 2（熊牛转换）和 Stage 4（牛熊转换）中检测细粒度信号，
    帮助判断突破/崩盘的进展。

    Returns:
        DataFrame 包含 columns: date, stage, signal
        signal 为 str 或 None
    """
    if config is None:
        config = StageConfig()

    df = weekly_df.dropna(subset=["ma34", "ma_slope"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "stage", "signal"])

    df = _prepare_features(df, config)

    vol_threshold = config.breakout_volume_threshold

    prev_state = 0
    had_approaching_breakout = False
    had_approaching_breakdown = False
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        price_pos = row["price_position"]
        ma_slope = row["ma_slope"]
        volume_ratio = row["volume_ratio"]
        alpha = float(row["adaptive_alpha"])
        flat_threshold = float(row["adaptive_slope_flat"])

        stage, _ = classify_stage_stateful(
            price_pos=price_pos,
            ma_slope=ma_slope,
            cross_count=int(row["crossover_count"]),
            volume_ratio=volume_ratio,
            support_held=bool(row["support_held"]),
            resistance_held=bool(row["resistance_held"]),
            config=config,
            previous_state=prev_state,
            consecutive_below_ma=int(row["consecutive_below_ma"]),
            consecutive_above_ma=int(row["consecutive_above_ma"]),
            adaptive_alpha=alpha,
            adaptive_slope_flat=flat_threshold,
        )

        signal = None

        # ── Stage 2 信号检测 ──
        if prev_state == 2 and stage == 3:
            # 从 Stage 2 转入 Stage 3 → 确认突破
            signal = "breakout_confirmed"
            had_approaching_breakout = False
        elif prev_state == 2 and stage == 1:
            # 从 Stage 2 退回 Stage 1 → 回落熊市
            signal = "bear_fallback"
            had_approaching_breakout = False
        elif stage == 2:
            # 仍在 Stage 2 中，检测接近突破 / 假突破
            if had_approaching_breakout and price_pos < 0:
                # 之前接近突破，但价格重新跌破 MA34
                signal = "breakout_failed"
                had_approaching_breakout = False
            elif (not had_approaching_breakout
                  and price_pos > 0 and ma_slope >= 0
                  and not (price_pos > alpha and ma_slope > flat_threshold
                           and volume_ratio >= vol_threshold)):
                # 首次：价格站上 MA34 + 斜率不再下行，但未满足完整突破条件
                signal = "approaching_breakout"
                had_approaching_breakout = True

        # ── Stage 4 信号检测 ──
        if prev_state == 4 and stage == 1:
            # 从 Stage 4 转入 Stage 1 → 确认崩盘
            signal = "breakdown_confirmed"
            had_approaching_breakdown = False
        elif prev_state == 4 and stage == 3:
            # 从 Stage 4 退回 Stage 3 → 修复回牛
            signal = "bull_recovery"
            had_approaching_breakdown = False
        elif stage == 4:
            # 仍在 Stage 4 中，检测接近崩盘 / 假崩盘
            if had_approaching_breakdown and price_pos > 0:
                # 之前接近崩盘，但价格重新站上 MA34
                signal = "breakdown_failed"
                had_approaching_breakdown = False
            elif (not had_approaching_breakdown
                  and price_pos < 0 and ma_slope <= 0
                  and not (price_pos < -alpha and ma_slope < -flat_threshold)):
                # 首次：价格跌破 MA34 + 斜率不再上行，但未满足完整崩盘条件
                signal = "approaching_breakdown"
                had_approaching_breakdown = True

        # 阶段切换时重置追踪标记
        if stage != 2:
            had_approaching_breakout = False
        if stage != 4:
            had_approaching_breakdown = False

        results.append({
            "date": row["date"],
            "stage": stage,
            "signal": signal,
        })
        prev_state = stage

    return pd.DataFrame(results)


# ── 批量分析 ──────────────────────────────────────────────


def batch_analyze(
    industry_weekly_data: dict[str, tuple[str, pd.DataFrame]],
    config: StageConfig | None = None,
) -> pd.DataFrame:
    """批量分析所有行业"""
    results = []
    for code, (name, weekly_df) in industry_weekly_data.items():
        result = analyze_industry(weekly_df, code, name, config)
        results.append({
            "code": result.code,
            "name": result.name,
            "stage": result.stage,
            "stage_label": result.stage_label,
            "confidence": result.confidence,
            "close": result.close,
            "ma34": result.ma34,
            "price_position": result.price_position,
            "ma_slope": result.ma_slope,
            "crossover_count": result.crossover_count,
            "volume_ratio": result.volume_ratio,
        })

    summary = pd.DataFrame(results)
    summary = summary.sort_values("stage").reset_index(drop=True)
    return summary
