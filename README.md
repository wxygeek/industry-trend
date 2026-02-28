# 申万行业趋势阶段分析

基于温斯坦（Stan Weinstein）34 周均线阶段分析法，对 A 股申万行业指数（2021 版）进行趋势阶段判断，帮助识别各行业当前处于熊市、牛市还是转换期。

支持 **一级行业**（31 个）和 **二级行业**（124 个）。

## 文档

| 文档 | 说明 |
|------|------|
| [阶段分析方法](docs/stage_analysis.md) | 四阶段模型、状态机转换规则、ATR 自适应阈值、参数配置 |
| [交易信号](docs/trading_signals.md) | 8 种交易信号定义、触发条件、信号生命周期、量化指标 |
| [回测策略](docs/backtest_strategy.md) | 行业轮动回测引擎设计与策略说明 |

## 项目结构

```
industry-trend/
├── main.py              # CLI 入口（argparse，支持 --level 参数）
├── app.py               # Streamlit Web 仪表板（4 Tab：一/二级趋势 + 回测）
├── requirements.txt     # Python 依赖
├── config/
│   └── industries.py    # 一级（31个）+ 二级（124个）行业代码、名称、归属关系
├── src/
│   ├── scraper.py       # 一级行业数据采集（Playwright 自动化浏览器，SWS 网站）
│   ├── akshare_downloader.py  # 二级行业数据采集（akshare API）
│   ├── kline.py         # 日K → 周K 转换 + MA34 + ATR14 计算
│   ├── stage_analyzer.py # 四阶段状态机判断算法 + ATR 自适应阈值
│   └── backtest.py      # 行业轮动回测引擎
├── docs/                # 详细文档
└── data/
    ├── sw_level1/
    │   ├── daily/           # 一级行业日K线 CSV
    │   └── analysis/        # 一级行业阶段分析结果
    └── sw_level2/
        ├── daily/           # 二级行业日K线 CSV
        └── analysis/        # 二级行业阶段分析结果
```

## 安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Playwright 浏览器（一级行业数据下载需要）
playwright install chromium
```

## 使用方法

### 1. 下载数据

```bash
# 下载一级行业数据（SWS 网站，Playwright）
python main.py download

# 下载二级行业数据（akshare API）
python main.py download --level 2

# 下载全部（一级 + 二级）
python main.py download --level all
```

添加 `--no-headless` 可显示浏览器窗口（仅一级行业下载有效）：

```bash
python main.py download --no-headless
```

### 2. 分析行业阶段

```bash
# 分析一级行业
python main.py analyze

# 分析二级行业
python main.py analyze --level 2
```

### 3. 下载 + 分析一步完成

```bash
python main.py all                # 一级行业
python main.py all --level 2      # 二级行业
python main.py all --level all    # 全部
```

### 4. 启动 Web 仪表板

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，仪表板包含 4 个 Tab：

- **一级趋势分析**：阶段分布、行业汇总表、周K线详情图、阶段转换历史
- **一级策略回测**：Weinstein 行业轮动策略回测，收益曲线、回撤、持仓、交易记录
- **二级趋势分析**：支持按一级行业分组筛选，124 个二级行业趋势分析
- **二级策略回测**：二级行业轮动策略回测

## 数据来源

- **一级行业**：[申万宏源研究所](https://www.swsresearch.com/institute_sw/allIndex/analysisIndex)（Playwright 自动化下载，1999 年至今）
- **二级行业**：[akshare](https://github.com/akfamily/akshare) 封装的申万行业指数 API（1999 年至今）
