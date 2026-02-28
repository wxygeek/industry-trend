"""行业趋势跟踪分析 - Streamlit Web仪表板

启动: streamlit run app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.industries import (
    SW_LEVEL1_INDUSTRIES, SW_LEVEL2_INDUSTRIES,
    get_industries, get_level2_by_parent,
)
from src.scraper import load_existing_csv as load_level1_csv
from src.scraper import DATA_DIR as LEVEL1_DATA_DIR
from src.akshare_downloader import (
    load_existing_csv as load_level2_csv,
    LEVEL2_DATA_DIR,
    analysis_dir_for_level,
)
from src.kline import generate_weekly_kline
from src.stage_analyzer import (
    batch_analyze, analyze_industry, detect_stage_transitions,
    compute_stage_series, compute_signal_series,
    StageConfig, BULLISH_SIGNALS, BEARISH_SIGNALS, SIGNAL_LABELS,
)
from src.backtest import (
    run_backtest, compute_benchmark, BacktestConfig, BacktestResult,
)

STAGE_COLORS = {
    1: "rgba(76, 175, 80, 0.15)",      # 绿 - 熊市
    2: "rgba(255, 99, 71, 0.10)",      # 浅红 - 熊牛转换
    3: "rgba(255, 99, 71, 0.20)",      # 红 - 牛市
    4: "rgba(76, 175, 80, 0.10)",      # 浅绿 - 牛熊转换
}

STAGE_LABELS = {
    1: "Stage 1 熊市",
    2: "Stage 2 熊牛转换",
    3: "Stage 3 牛市",
    4: "Stage 4 牛熊转换",
}


@st.cache_data(ttl=300)
def load_all_data(level: int):
    """加载指定级别的所有行业数据并计算周K线和阶段"""
    industries = get_industries(level)
    load_csv = load_level1_csv if level == 1 else (lambda c, n: load_level2_csv(c, n, 2))

    industry_data = {}
    for code, name in industries.items():
        daily_df = load_csv(code, name)
        if daily_df is None or daily_df.empty:
            continue
        weekly_df = generate_weekly_kline(daily_df)
        industry_data[code] = (name, weekly_df)

    if not industry_data:
        return None, pd.DataFrame()

    config = StageConfig()
    summary = batch_analyze(industry_data, config)

    # 保存分析结果
    if not summary.empty:
        out_dir = analysis_dir_for_level(level)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "stage_summary.csv", index=False)

    return industry_data, summary


def get_last_update_time(level: int) -> str:
    """获取数据最后更新时间"""
    data_dir = LEVEL1_DATA_DIR if level == 1 else LEVEL2_DATA_DIR
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        return "无数据"
    latest = max(csv_files, key=lambda f: f.stat().st_mtime)
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)
    return mtime.strftime("%Y-%m-%d %H:%M")


def render_stage_distribution(summary: pd.DataFrame):
    """渲染阶段分布图"""
    dist = summary["stage"].value_counts().sort_index()

    fig = go.Figure()
    colors = ["#4CAF50", "#FF6347", "#FF6347", "#4CAF50"]
    labels = [STAGE_LABELS.get(s, f"Stage {s}") for s in dist.index]

    fig.add_trace(go.Bar(
        x=labels,
        y=dist.values,
        marker_color=[colors[s - 1] for s in dist.index],
        text=dist.values,
        textposition="auto",
    ))
    fig.update_layout(
        title="行业阶段分布",
        xaxis_title="阶段",
        yaxis_title="行业数量",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_summary_table(summary: pd.DataFrame, stage_filter: str, level: int = 1):
    """渲染行业汇总表格"""
    df = summary.copy()

    if stage_filter != "全部阶段":
        stage_num = int(stage_filter.split(" ")[1])
        df = df[df["stage"] == stage_num]

    cols = ["name", "code", "stage_label", "confidence", "close", "ma34",
            "price_position", "ma_slope"]
    display_cols = ["行业名称", "行业代码", "当前阶段", "置信度", "收盘价", "MA34", "偏离度", "MA斜率"]

    # 二级行业增加所属一级行业列
    if level == 2:
        df["parent_name"] = df["code"].apply(
            lambda c: SW_LEVEL1_INDUSTRIES.get(SW_LEVEL2_INDUSTRIES.get(c, ("", ""))[1], "")
        )
        cols = ["parent_name"] + cols
        display_cols = ["所属一级"] + display_cols

    display_df = df[cols].copy()
    display_df.columns = display_cols
    display_df["偏离度"] = display_df["偏离度"].apply(lambda x: f"{x*100:+.2f}%")
    display_df["MA斜率方向"] = display_df["MA斜率"].apply(
        lambda x: "↑ 上升" if x > 0.005 else ("↓ 下降" if x < -0.005 else "→ 走平")
    )
    display_df["置信度"] = display_df["置信度"].apply(lambda x: f"{x:.2f}")
    display_df = display_df.drop(columns=["MA斜率"])

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    return df


def render_kline_chart(weekly_df: pd.DataFrame, code: str, name: str, weeks: int = 104):
    """渲染交互式K线图 + MA34叠加 + 成交量"""
    df = weekly_df.tail(weeks).copy()

    # 双子图：上方K线（占70%高度），下方成交量（占30%高度），共享x轴
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # K线图
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="周K线",
        increasing_line_color="#FF4136",
        decreasing_line_color="#2ECC40",
    ), row=1, col=1)

    # MA34线
    ma_data = df.dropna(subset=["ma34"])
    fig.add_trace(go.Scatter(
        x=ma_data["date"],
        y=ma_data["ma34"],
        mode="lines",
        name="MA34",
        line=dict(color="#FF6600", width=2),
    ), row=1, col=1)

    # 成交量柱状图（涨红跌绿）
    vol_colors = [
        "#FF4136" if row["close"] >= row["open"] else "#2ECC40"
        for _, row in df.iterrows()
    ]
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["volume"],
        name="成交量",
        marker_color=vol_colors,
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    # 阶段背景色标注（基于状态机）— 同时覆盖上下两个子图
    config = StageConfig()
    stage_df = df.dropna(subset=["ma34", "ma_slope"]).copy()
    if not stage_df.empty:
        stage_series = compute_stage_series(stage_df, config)
        stage_df["stage"] = stage_series.values

        # 收集阶段区间
        bands = []
        prev_stage = None
        band_start = None
        for idx, row in stage_df.iterrows():
            if row["stage"] != prev_stage:
                if prev_stage is not None and band_start is not None:
                    bands.append((band_start, row["date"], prev_stage))
                band_start = row["date"]
                prev_stage = row["stage"]
        if prev_stage is not None and band_start is not None:
            bands.append((band_start, stage_df.iloc[-1]["date"], prev_stage))

        # 对两个子图都添加背景色
        for x0, x1, stage in bands:
            color = STAGE_COLORS.get(stage, "rgba(128,128,128,0.1)")
            for row_num in [1, 2]:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=color,
                    layer="below", line_width=0,
                    row=row_num, col=1,
                )

    # 交易信号标注
    signal_df = compute_signal_series(df, config)
    signal_df = signal_df[signal_df["signal"].notna()].copy()
    if not signal_df.empty:
        # 合并 K 线数据以获取 high/low 用于定位
        signal_df = signal_df.merge(
            df[["date", "high", "low"]], on="date", how="left"
        )
        # 信号样式配置：(颜色, 符号)
        signal_styles = {
            # 利好信号：标注在 low 下方
            "approaching_breakout": ("#90CAF9", "triangle-up"),     # 浅蓝
            "breakout_confirmed":   ("#4CAF50", "triangle-up"),     # 深绿
            "breakdown_failed":     ("#FFA726", "triangle-up"),     # 橙色
            "bull_recovery":        ("#4CAF50", "triangle-up"),     # 深绿
            # 利空信号：标注在 high 上方
            "approaching_breakdown": ("#CE93D8", "triangle-down"),  # 浅紫
            "breakdown_confirmed":   ("#F44336", "triangle-down"),  # 深红
            "breakout_failed":       ("#FFA726", "triangle-down"),  # 橙色
            "bear_fallback":         ("#F44336", "triangle-down"),  # 深红
        }
        # 计算价格范围用于偏移
        price_range = df["high"].max() - df["low"].min()
        offset = price_range * 0.03

        for sig_name, (color, symbol) in signal_styles.items():
            subset = signal_df[signal_df["signal"] == sig_name]
            if subset.empty:
                continue
            is_bullish = sig_name in BULLISH_SIGNALS
            y_vals = (subset["low"] - offset) if is_bullish else (subset["high"] + offset)
            label = SIGNAL_LABELS.get(sig_name, sig_name)

            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=y_vals,
                mode="markers+text",
                marker=dict(symbol=symbol, size=12, color=color),
                text=[label] * len(subset),
                textposition="bottom center" if is_bullish else "top center",
                textfont=dict(size=9, color=color),
                name=label,
                showlegend=True,
                legendgroup="signals",
            ), row=1, col=1)

    fig.update_layout(
        title=f"{name}({code}) 周K线 + 34周均线",
        yaxis_title="指数",
        yaxis2_title="成交量",
        xaxis2_title="日期",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 趋势分析 Tab ──────────────────────────────────────────


def render_trend_tab(industry_data: dict, summary: pd.DataFrame, level: int = 1):
    """渲染趋势分析 Tab 的全部内容"""
    level_name = "一" if level == 1 else "二"

    # 阶段分布概览
    st.subheader("阶段分布概览")
    render_stage_distribution(summary)

    # 阶段筛选
    st.subheader("行业阶段汇总")

    filter_cols = st.columns([1, 1] if level == 2 else [1])

    with filter_cols[0]:
        filter_options = ["全部阶段"] + [STAGE_LABELS[i] for i in range(1, 5)]
        stage_filter = st.selectbox("筛选阶段", filter_options, key=f"stage_filter_l{level}")

    # 二级行业增加一级行业筛选
    parent_filter = None
    if level == 2 and len(filter_cols) > 1:
        with filter_cols[1]:
            parent_options = ["全部一级行业"] + [
                f"{name}({code})" for code, name in SW_LEVEL1_INDUSTRIES.items()
            ]
            parent_filter = st.selectbox("筛选所属一级行业", parent_options, key="parent_filter_l2")

    # 对二级行业应用一级行业筛选
    filtered_summary = summary
    if level == 2 and parent_filter and parent_filter != "全部一级行业":
        parent_code = parent_filter.split("(")[1].rstrip(")")
        child_codes = set(get_level2_by_parent(parent_code).keys())
        filtered_summary = summary[summary["code"].isin(child_codes)]

    render_summary_table(filtered_summary, stage_filter, level)

    # K线图表
    st.subheader("行业周K线详情")

    # 二级行业按一级行业分组选择
    if level == 2:
        group_col, select_col = st.columns([1, 2])
        with group_col:
            group_options = ["全部二级行业"] + [
                f"{name}({code})" for code, name in SW_LEVEL1_INDUSTRIES.items()
                if any(SW_LEVEL2_INDUSTRIES.get(c, ("", ""))[1] == code for c in industry_data)
            ]
            group_choice = st.selectbox("按一级行业分组", group_options, key="kline_group_l2")

        if group_choice == "全部二级行业":
            available = {f"{name}({code})": code for code, (name, _) in industry_data.items()}
        else:
            grp_code = group_choice.split("(")[1].rstrip(")")
            available = {
                f"{name}({code})": code
                for code, (name, _) in industry_data.items()
                if SW_LEVEL2_INDUSTRIES.get(code, ("", ""))[1] == grp_code
            }
        with select_col:
            selected_label = st.selectbox("选择行业", sorted(available.keys()), key=f"kline_select_l{level}")
    else:
        available = {f"{name}({code})": code for code, (name, _) in industry_data.items()}
        selected_label = st.selectbox("选择行业", list(available.keys()), key=f"kline_select_l{level}")

    if selected_label and selected_label in available:
        selected_code = available[selected_label]
        name, weekly_df = industry_data[selected_code]

        col1, col2 = st.columns([3, 1])
        with col2:
            weeks_display = st.slider("显示周数", min_value=26, max_value=260, value=104, step=26,
                                      key=f"weeks_slider_l{level}")

        render_kline_chart(weekly_df, selected_code, name, weeks=weeks_display)

        # 显示该行业的阶段转换历史
        transitions = detect_stage_transitions(weekly_df)
        if not transitions.empty:
            st.caption("阶段转换历史（近期）")
            recent_transitions = transitions.tail(10).copy()
            recent_transitions["prev_stage"] = recent_transitions["prev_stage"].map(STAGE_LABELS)
            recent_transitions["new_stage"] = recent_transitions["new_stage"].map(STAGE_LABELS)
            recent_transitions.columns = ["日期", "前阶段", "新阶段"]
            st.dataframe(recent_transitions, use_container_width=True, hide_index=True)


# ── 回测 Tab ──────────────────────────────────────────────


def _render_metrics_cards(result: BacktestResult):
    """渲染绩效指标卡片"""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("总收益率", f"{result.total_return * 100:+.1f}%")
    with c2:
        st.metric("年化收益率", f"{result.annualized_return * 100:+.1f}%")
    with c3:
        st.metric("最大回撤", f"{result.max_drawdown * 100:.1f}%")
    with c4:
        st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    with c5:
        st.metric("胜率", f"{result.win_rate * 100:.1f}%")
    with c6:
        st.metric("总交易次数", f"{result.total_trades}")


def _render_equity_curve(result: BacktestResult, benchmark: pd.DataFrame, level: int = 1):
    """渲染收益曲线图"""
    ec = result.equity_curve
    level_name = "一" if level == 1 else "二"
    n_industries = len(get_industries(level))

    fig = go.Figure()

    # 策略收益曲线
    fig.add_trace(go.Scatter(
        x=ec["date"],
        y=ec["portfolio_value"],
        mode="lines",
        name="策略净值",
        line=dict(color="#1E88E5", width=2),
    ))

    # 基准曲线
    if not benchmark.empty:
        fig.add_trace(go.Scatter(
            x=benchmark["date"],
            y=benchmark["benchmark_value"],
            mode="lines",
            name="等权基准",
            line=dict(color="#9E9E9E", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="策略收益曲线 vs 等权基准",
        xaxis_title="日期",
        yaxis_title="组合价值",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"等权基准：回测起始日将等量资金平均分配到{n_industries}个申万{level_name}级行业并持有不动，"
        f"反映全行业被动持有的收益水平。"
    )


def _render_drawdown_chart(result: BacktestResult):
    """渲染回撤曲线图"""
    ec = result.equity_curve
    running_max = ec["portfolio_value"].cummax()
    drawdown = (ec["portfolio_value"] - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec["date"],
        y=drawdown,
        fill="tozeroy",
        mode="lines",
        name="回撤",
        line=dict(color="#EF5350", width=1),
        fillcolor="rgba(239, 83, 80, 0.3)",
    ))
    fig.update_layout(
        title="回撤曲线",
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        height=250,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_positions_chart(result: BacktestResult):
    """渲染持仓数量变化图"""
    ec = result.equity_curve

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec["date"],
        y=ec["n_positions"],
        mode="lines",
        name="持仓数量",
        line=dict(color="#AB47BC", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(171, 71, 188, 0.15)",
    ))
    fig.update_layout(
        title="持仓数量变化",
        xaxis_title="日期",
        yaxis_title="持仓行业数",
        height=200,
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_current_positions(result: BacktestResult, industry_data: dict):
    """渲染当前持仓表"""
    st.markdown("**当前持仓**")
    if not result.final_positions:
        st.info("回测结束时无持仓")
        return

    rows = []
    for pos in result.final_positions:
        # 获取最新价格
        if pos.code in industry_data:
            _, weekly_df = industry_data[pos.code]
            current_price = weekly_df["close"].iloc[-1]
        else:
            current_price = pos.entry_price

        ret = pos.return_pct(current_price)
        holding_weeks = 0
        if not result.equity_curve.empty:
            last_date = result.equity_curve["date"].iloc[-1]
            holding_weeks = (last_date - pos.entry_date).days // 7

        rows.append({
            "行业": pos.name,
            "代码": pos.code,
            "买入日期": pos.entry_date.strftime("%Y-%m-%d"),
            "买入价": f"{pos.entry_price:.2f}",
            "现价": f"{current_price:.2f}",
            "收益率": f"{ret * 100:+.1f}%",
            "持仓周数": holding_weeks,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_trade_history(result: BacktestResult):
    """渲染交易历史表"""
    st.markdown("**交易历史**")
    if not result.trades:
        st.info("无交易记录")
        return

    reason_labels = {
        "breakout_confirmed": "确认突破",
        "breakdown_confirmed": "确认崩盘",
        "replaced_weakest": "弱势替换",
    }
    action_labels = {"buy": "买入", "sell": "卖出"}

    # 建立买入记录映射，用于计算卖出时的持仓时间
    buy_history: dict[str, list] = {}  # {code: [Trade, ...]}
    for t in result.trades:
        if t.action == "buy":
            buy_history.setdefault(t.code, []).append(t)

    rows = []
    for t in reversed(result.trades):
        holding_info = ""
        pnl_info = ""
        if t.action == "sell":
            buys = [b for b in buy_history.get(t.code, []) if b.date <= t.date]
            if buys:
                entry = buys[-1]
                weeks = (t.date - entry.date).days // 7
                holding_info = f"{weeks} 周"
                ret = (t.price - entry.price) / entry.price
                pnl_info = f"{ret * 100:+.1f}%"

        rows.append({
            "日期": t.date.strftime("%Y-%m-%d"),
            "行业": t.name,
            "操作": action_labels.get(t.action, t.action),
            "价格": f"{t.price:.2f}",
            "金额": f"{t.value:,.0f}",
            "盈亏": pnl_info,
            "持仓时间": holding_info,
            "原因": reason_labels.get(t.reason, t.reason),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=400)


def render_backtest_tab(industry_data: dict, level: int = 1):
    """渲染策略回测 Tab"""
    level_name = "一" if level == 1 else "二"
    st.subheader(f"Weinstein {level_name}级行业轮动策略回测")

    # 获取数据日期范围
    all_dates = set()
    for code, (name, weekly_df) in industry_data.items():
        valid = weekly_df.dropna(subset=["ma34"])
        if not valid.empty:
            all_dates.update(valid["date"].tolist())

    if not all_dates:
        st.warning("数据不足，无法运行回测")
        return

    sorted_dates = sorted(all_dates)
    min_date = sorted_dates[0].to_pydatetime().date()
    max_date = sorted_dates[-1].to_pydatetime().date()

    # 参数设置
    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "回测区间",
            value=(datetime(2005, 1, 1).date(), max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"bt_date_l{level}",
        )
    with col2:
        initial_capital = st.number_input(
            "初始资金", value=10000, min_value=1000, step=1000,
            key=f"bt_capital_l{level}",
        )
    with col3:
        default_max_pos = 5 if level == 1 else 10
        max_pos_limit = 15 if level == 2 else 10
        max_positions = st.number_input(
            "最大持仓数", value=default_max_pos, min_value=1, max_value=max_pos_limit, step=1,
            key=f"bt_maxpos_l{level}",
        )

    # 处理日期输入（可能是元组或单个日期）
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt, end_dt = date_range
    else:
        st.warning("请选择完整的起止日期")
        return

    session_key_result = f"backtest_result_l{level}"
    session_key_benchmark = f"backtest_benchmark_l{level}"

    run_clicked = st.button("运行回测", type="primary", use_container_width=True,
                            key=f"bt_run_l{level}")

    if run_clicked:
        n = len(industry_data)
        with st.spinner(f"正在运行{level_name}级行业回测（{n}个行业）..."):
            bt_config = BacktestConfig(
                initial_capital=float(initial_capital),
                max_positions=int(max_positions),
                start_date=str(start_dt),
                end_date=str(end_dt),
            )
            result = run_backtest(industry_data, bt_config)
            benchmark = compute_benchmark(
                industry_data,
                initial_capital=float(initial_capital),
                start_date=pd.Timestamp(start_dt),
                end_date=pd.Timestamp(end_dt),
            )
            st.session_state[session_key_result] = result
            st.session_state[session_key_benchmark] = benchmark

    # 展示结果
    if session_key_result not in st.session_state:
        st.info("设置参数后点击「运行回测」查看结果")
        return

    result: BacktestResult = st.session_state[session_key_result]
    benchmark: pd.DataFrame = st.session_state.get(session_key_benchmark, pd.DataFrame())

    if result.equity_curve.empty:
        st.warning("回测期间无交易信号，请调整回测区间")
        return

    # 绩效指标卡片
    _render_metrics_cards(result)

    # 附加指标
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("平均持仓周数", f"{result.avg_holding_weeks:.1f}")
    with c2:
        st.metric("最大回撤持续", f"{result.max_drawdown_duration_weeks} 周")
    with c3:
        final_value = result.equity_curve["portfolio_value"].iloc[-1]
        st.metric("期末资产", f"{final_value:,.0f}")
    with c4:
        st.metric("盈亏比", f"{result.profit_loss_ratio:.2f}")
    with c5:
        best_label = f"{result.best_trade_name} ({result.best_trade_return * 100:+.1f}%)" if result.best_trade_name else "—"
        st.metric("最佳单笔交易", best_label)

    # 收益曲线
    _render_equity_curve(result, benchmark, level)

    # 回撤曲线 + 持仓数量
    col_dd, col_pos = st.columns(2)
    with col_dd:
        _render_drawdown_chart(result)
    with col_pos:
        _render_positions_chart(result)

    # 当前持仓 + 交易历史
    col_left, col_right = st.columns([1, 2])
    with col_left:
        _render_current_positions(result, industry_data)
    with col_right:
        _render_trade_history(result)


# ── 主入口 ──────────────────────────────────────────────


def main():
    st.set_page_config(page_title="行业趋势跟踪", page_icon="📊", layout="wide")
    st.title("📊 申万行业趋势阶段分析")
    st.caption("基于温斯坦（Weinstein）34周均线阶段分析法 · 支持一级/二级行业")

    # 侧边栏
    st.sidebar.markdown(f"**一级行业数据:** {get_last_update_time(1)}")
    st.sidebar.markdown(f"**二级行业数据:** {get_last_update_time(2)}")

    if st.sidebar.button("🔄 刷新一级数据", use_container_width=True):
        with st.spinner("正在下载一级行业最新数据..."):
            from src.scraper import SWSScraper
            with SWSScraper(headless=True) as scraper:
                scraper.download_all()
            st.cache_data.clear()
            st.rerun()

    if st.sidebar.button("🔄 刷新二级数据", use_container_width=True):
        with st.spinner("正在下载二级行业最新数据..."):
            from src.akshare_downloader import download_all as ak_download
            ak_download(level=2)
            st.cache_data.clear()
            st.rerun()

    # 加载数据
    l1_data, l1_summary = load_all_data(1)
    l2_data, l2_summary = load_all_data(2)

    has_l1 = l1_summary is not None and not l1_summary.empty
    has_l2 = l2_summary is not None and not l2_summary.empty

    if not has_l1 and not has_l2:
        st.warning(
            "⚠️ 未找到本地数据。请先运行以下命令下载数据：\n"
            "- 一级行业: `python main.py download`\n"
            "- 二级行业: `python main.py download --level 2`"
        )
        return

    # 4 Tab 页切换
    tab_names = []
    if has_l1:
        tab_names += ["一级趋势分析", "一级策略回测"]
    if has_l2:
        tab_names += ["二级趋势分析", "二级策略回测"]

    tabs = st.tabs(tab_names)

    tab_idx = 0
    if has_l1:
        with tabs[tab_idx]:
            render_trend_tab(l1_data, l1_summary, level=1)
        tab_idx += 1
        with tabs[tab_idx]:
            render_backtest_tab(l1_data, level=1)
        tab_idx += 1

    if has_l2:
        with tabs[tab_idx]:
            render_trend_tab(l2_data, l2_summary, level=2)
        tab_idx += 1
        with tabs[tab_idx]:
            render_backtest_tab(l2_data, level=2)


if __name__ == "__main__":
    main()
