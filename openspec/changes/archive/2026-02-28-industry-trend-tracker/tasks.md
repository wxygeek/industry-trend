## 1. Project Setup

- [x] 1.1 Create project directory structure: `config/`, `data/daily/`, `src/`
- [x] 1.2 Create `requirements.txt` with dependencies: playwright, pandas, plotly, streamlit, openpyxl
- [x] 1.3 Create `config/industries.py` with 31 SW Level-1 industry codes and names as a static dictionary
- [x] 1.4 Install dependencies and verify Playwright browser is available

## 2. Data Scraping Module

- [x] 2.1 Create `src/scraper.py` with Playwright-based scraper class that navigates to SWS release detail page per industry
- [x] 2.2 Implement single-industry download: navigate to page, click "行情下载", capture the downloaded XLS file
- [x] 2.3 Implement XLS parsing: read downloaded XLS with pandas, extract and rename columns to standardized names (date, open, high, low, close, volume, amount)
- [x] 2.4 Implement CSV persistence: save parsed DataFrame to `data/daily/{code}_{name}.csv` sorted by date ascending
- [x] 2.5 Implement incremental update logic: detect latest date in existing CSV, only append newer rows after download
- [x] 2.6 Implement batch download: iterate all 31 industries with 2-3 second delay between each
- [x] 2.7 Implement error handling with retry (up to 3 times, exponential backoff) and failure summary reporting

## 3. Weekly K-line Generation Module

- [x] 3.1 Create `src/kline.py` with weekly K-line generation function
- [x] 3.2 Implement daily-to-weekly aggregation using pandas resample: open=first, high=max, low=min, close=last, volume=sum, amount=sum
- [x] 3.3 Handle partial weeks (holidays, current incomplete week) correctly
- [x] 3.4 Implement 34-week SMA calculation on weekly close prices
- [x] 3.5 Implement MA34 slope calculation: percentage change over configurable N-week window (default 4 weeks)
- [x] 3.6 Output standardized DataFrame with columns: date, open, high, low, close, volume, amount, ma34, ma_slope

## 4. Stage Analysis Algorithm

- [x] 4.1 Create `src/stage_analyzer.py` with configurable threshold parameters (price_position threshold, slope flat threshold, crossover window, etc.)
- [x] 4.2 Implement price_position calculation: (close - MA34) / MA34
- [x] 4.3 Implement MA34 crossover detection: identify cross-above and cross-below events, track frequency over rolling 12-week window
- [x] 4.4 Implement volume_ratio calculation: 4-week average volume / 34-week average volume
- [x] 4.5 Implement four-stage classification logic combining price_position, ma_slope, crossover_frequency, and volume_ratio
- [x] 4.6 Implement confidence score calculation (0.0-1.0) based on indicator alignment
- [x] 4.7 Implement stage transition detection: compare current vs previous week's stage, record transition events
- [x] 4.8 Create batch analysis function: run stage analysis for all 31 industries and return summary DataFrame

## 5. CLI Entry Point

- [x] 5.1 Create `main.py` as CLI entry point with commands: `download` (scrape data), `analyze` (run analysis), `all` (download + analyze)
- [x] 5.2 Wire up scraper → kline → stage_analyzer pipeline in `main.py`
- [x] 5.3 Add console output: print stage summary table to terminal after analysis

## 6. Web Dashboard

- [x] 6.1 Create `app.py` as Streamlit entry point
- [x] 6.2 Implement stage distribution overview chart (pie/bar chart showing count per stage)
- [x] 6.3 Implement industry summary table with columns: 行业名称, 行业代码, 当前阶段, 置信度, 收盘价, MA34, 偏离度(%), MA斜率方向
- [x] 6.4 Implement stage filter (dropdown to select Stage 1/2/3/4 or all)
- [x] 6.5 Implement table sorting by any column
- [x] 6.6 Implement industry selector for detailed K-line chart view
- [x] 6.7 Implement interactive Plotly candlestick chart with MA34 overlay line
- [x] 6.8 Implement stage background color annotation on K-line chart (Stage 1=red, 2=yellow, 3=green, 4=orange)
- [x] 6.9 Implement "刷新数据" button to trigger re-download and re-analysis with progress indicator
- [x] 6.10 Display last data update timestamp on dashboard

## 7. Testing & Validation

- [x] 7.1 Test scraper with a single industry (801010 农林牧渔) — verify XLS download and CSV output
- [x] 7.2 Test weekly K-line aggregation — verify OHLCV aggregation logic with known data
- [x] 7.3 Test MA34 calculation — verify against manual calculation for a sample industry
- [x] 7.4 Test stage classification — verify each stage with manually identified examples
- [ ] 7.5 End-to-end test: download all 31 industries, run analysis, verify dashboard displays correctly
