"""行业趋势跟踪分析 - Streamlit Web仪表板

启动: streamlit run app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.industries import SW_LEVEL1_INDUSTRIES
from src.scraper import load_existing_csv, DATA_DIR
from src.kline import generate_weekly_kline
from src.stage_analyzer import (
    batch_analyze, analyze_industry, detect_stage_transitions,
    compute_stage_series, compute_signal_series,
    StageConfig, BULLISH_SIGNALS, BEARISH_SIGNALS, SIGNAL_LABELS,
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
def load_all_data():
    """加载所有行业数据并计算周K线和阶段"""
    industry_data = {}
    for code, name in SW_LEVEL1_INDUSTRIES.items():
        daily_df = load_existing_csv(code, name)
        if daily_df is None or daily_df.empty:
            continue
        weekly_df = generate_weekly_kline(daily_df)
        industry_data[code] = (name, weekly_df)

    if not industry_data:
        return None, pd.DataFrame()

    config = StageConfig()
    summary = batch_analyze(industry_data, config)
    return industry_data, summary


def get_last_update_time() -> str:
    """获取数据最后更新时间"""
    csv_files = list(DATA_DIR.glob("*.csv"))
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


def render_summary_table(summary: pd.DataFrame, stage_filter: str):
    """渲染行业汇总表格"""
    df = summary.copy()

    if stage_filter != "全部阶段":
        stage_num = int(stage_filter.split(" ")[1])
        df = df[df["stage"] == stage_num]

    display_df = df[["name", "code", "stage_label", "confidence", "close", "ma34",
                     "price_position", "ma_slope"]].copy()
    display_df.columns = ["行业名称", "行业代码", "当前阶段", "置信度", "收盘价", "MA34", "偏离度", "MA斜率"]
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


def main():
    st.set_page_config(page_title="行业趋势跟踪", page_icon="📊", layout="wide")
    st.title("📊 申万一级行业趋势阶段分析")
    st.caption("基于温斯坦（Weinstein）34周均线阶段分析法")

    # 数据更新时间
    update_time = get_last_update_time()
    st.sidebar.markdown(f"**数据更新时间:** {update_time}")

    # 刷新按钮
    if st.sidebar.button("🔄 刷新数据", use_container_width=True):
        with st.spinner("正在下载最新数据..."):
            from src.scraper import SWSScraper
            with SWSScraper(headless=True) as scraper:
                scraper.download_all()
            st.cache_data.clear()
            st.rerun()

    # 加载数据
    industry_data, summary = load_all_data()

    if summary.empty:
        st.warning("⚠️ 未找到本地数据。请先运行 `python main.py download` 下载数据，或点击侧边栏的「刷新数据」按钮。")
        return

    # 阶段分布概览
    st.subheader("阶段分布概览")
    render_stage_distribution(summary)

    # 阶段筛选
    st.subheader("行业阶段汇总")
    filter_options = ["全部阶段"] + [STAGE_LABELS[i] for i in range(1, 5)]
    stage_filter = st.selectbox("筛选阶段", filter_options)
    filtered = render_summary_table(summary, stage_filter)

    # K线图表
    st.subheader("行业周K线详情")
    available = {f"{name}({code})": code for code, (name, _) in industry_data.items()}
    selected_label = st.selectbox("选择行业", list(available.keys()))

    if selected_label:
        selected_code = available[selected_label]
        name, weekly_df = industry_data[selected_code]

        col1, col2 = st.columns([3, 1])
        with col2:
            weeks_display = st.slider("显示周数", min_value=26, max_value=260, value=104, step=26)

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


if __name__ == "__main__":
    main()
