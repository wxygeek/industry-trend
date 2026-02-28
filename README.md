# 申万一级行业趋势阶段分析

基于温斯坦（Stan Weinstein）34 周均线阶段分析法，对 A 股 31 个申万一级行业（2021 版）进行趋势阶段判断，帮助识别各行业当前处于熊市、牛市还是转换期。

## 文档

| 文档 | 说明 |
|------|------|
| [阶段分析方法](docs/stage_analysis.md) | 四阶段模型、状态机转换规则、ATR 自适应阈值、参数配置 |
| [交易信号](docs/trading_signals.md) | 8 种交易信号定义、触发条件、信号生命周期、量化指标 |
| [回测策略](docs/backtest_strategy.md) | 行业轮动回测引擎设计与策略说明 |

## 项目结构

```
industry-trend/
├── main.py              # CLI 入口
├── app.py               # Streamlit Web 仪表板
├── requirements.txt     # Python 依赖
├── config/
│   └── industries.py    # 31 个申万一级行业代码和名称
├── src/
│   ├── scraper.py       # 数据采集（Playwright 自动化浏览器）
│   ├── kline.py         # 日K → 周K 转换 + MA34 + ATR14 计算
│   └── stage_analyzer.py # 四阶段状态机判断算法 + ATR 自适应阈值
├── docs/                # 详细文档
└── data/
    ├── daily/           # 日K线 CSV（按行业存储）
    └── xls/             # 原始下载的 XLS 文件
```

## 安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器（数据下载需要）
playwright install chromium
```

## 使用方法

### 1. 下载数据

从申万研究所网站下载全部 31 个行业的历史行情数据：

```bash
python main.py download
```

添加 `--no-headless` 可显示浏览器窗口：

```bash
python main.py download --no-headless
```

### 2. 分析行业阶段

对已下载的数据运行阶段分析，在终端输出结果：

```bash
python main.py analyze
```

输出示例：

```
================================================================================
申万一级行业趋势阶段分析（基于34周均线）
================================================================================

阶段分布:
  Stage 1 熊市: 1 个行业
  Stage 2 熊牛转换: 3 个行业
  Stage 3 牛市: 22 个行业
  Stage 4 牛熊转换: 5 个行业

行业名称         代码         阶段    置信度      收盘价      MA34     偏离度   MA斜率
--------------------------------------------------------------------------------
银行         801780      S1 熊市   0.90   3883.56   4189.75   -7.3%      ↓
基础化工     801030      S3 牛市   0.65   5163.81   4200.32  +22.9%      ↑
...
```

### 3. 下载 + 分析一步完成

```bash
python main.py all
```

### 4. 启动 Web 仪表板

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，仪表板包含：

- **阶段分布概览**：柱状图展示各阶段行业数量
- **行业汇总表**：支持按阶段筛选、按列排序
- **周K线详情图**：交互式蜡烛图 + MA34 均线，背景色标注阶段区间，交易信号箭头标注
- **阶段转换历史**：记录各行业阶段切换的时间节点
- **刷新数据按钮**：一键重新下载并分析

## 数据来源

[申万宏源研究所](https://www.swsresearch.com/institute_sw/allIndex/analysisIndex) — 申万一级行业指数历史行情数据（1999 年至今）。
