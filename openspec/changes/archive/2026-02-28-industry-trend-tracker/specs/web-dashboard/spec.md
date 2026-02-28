## ADDED Requirements

### Requirement: Industry stage summary table
The system SHALL display a summary table showing all 31 industries with their current Weinstein stage, confidence score, price position relative to MA34, MA34 slope direction, and latest closing price. The table SHALL be sortable by any column.

#### Scenario: View all industries sorted by stage
- **WHEN** the user opens the dashboard
- **THEN** the system SHALL display a table with columns: 行业名称, 行业代码, 当前阶段, 置信度, 收盘价, MA34, 偏离度(%), MA斜率方向
- **THEN** the table SHALL default to sorting by 当前阶段 ascending

#### Scenario: Sort table by confidence
- **WHEN** the user clicks the "置信度" column header
- **THEN** the table SHALL re-sort by confidence score

### Requirement: Stage filter
The system SHALL provide a filter mechanism allowing users to view only industries in a specific stage (1, 2, 3, or 4), or all stages.

#### Scenario: Filter by Stage 2
- **WHEN** the user selects "Stage 2 (熊牛转换)" from the filter
- **THEN** only industries currently classified as Stage 2 SHALL be displayed in the table

#### Scenario: Show all stages
- **WHEN** the user selects "全部阶段" from the filter
- **THEN** all 31 industries SHALL be displayed

### Requirement: Interactive weekly K-line chart with MA34 overlay
The system SHALL display an interactive candlestick chart for any selected industry showing weekly K-lines overlaid with the 34-week moving average line. The chart SHALL support zoom, pan, and hover tooltips showing OHLCV values and MA34 value.

#### Scenario: View industry K-line chart
- **WHEN** the user selects "农林牧渔 (801010)" from the industry selector
- **THEN** the system SHALL render a Plotly candlestick chart with weekly OHLC bars and a MA34 line overlay
- **THEN** the chart SHALL show at least the most recent 52 weeks of data by default

#### Scenario: Chart interaction
- **WHEN** the user hovers over a weekly bar
- **THEN** a tooltip SHALL display: date, open, high, low, close, volume, MA34 value

#### Scenario: Zoom into specific period
- **WHEN** the user uses the range slider or mouse scroll to zoom
- **THEN** the chart SHALL adjust to show the selected time period with smooth rendering

### Requirement: Stage annotation on chart
The system SHALL visually annotate the current stage on the K-line chart using background color bands or markers to indicate which time periods correspond to which stages.

#### Scenario: Stage background coloring
- **WHEN** a K-line chart is displayed for an industry
- **THEN** the background SHALL be color-coded: Stage 1 (red/pink), Stage 2 (yellow/amber), Stage 3 (green), Stage 4 (orange)
- **THEN** stage transition boundaries SHALL be marked

### Requirement: Stage distribution overview
The system SHALL display an overview chart showing the distribution of all 31 industries across the four stages (e.g., a pie chart or bar chart showing count per stage).

#### Scenario: View stage distribution
- **WHEN** the user views the dashboard overview
- **THEN** the system SHALL display a chart showing how many industries are in each stage (e.g., Stage 1: 5, Stage 2: 8, Stage 3: 12, Stage 4: 6)

### Requirement: Data refresh trigger
The system SHALL provide a button or mechanism to trigger data refresh (re-download latest data and re-run analysis) from the dashboard interface.

#### Scenario: Manual refresh
- **WHEN** the user clicks "刷新数据" button
- **THEN** the system SHALL run the data scraper for all industries, regenerate weekly K-lines, re-run stage analysis, and update all dashboard displays
- **THEN** the system SHALL show a progress indicator during refresh

### Requirement: Last update timestamp
The system SHALL display the date and time of the last data update on the dashboard.

#### Scenario: Display last update time
- **WHEN** the dashboard loads
- **THEN** it SHALL display "数据更新时间: YYYY-MM-DD HH:MM" showing when the data was last refreshed
