## ADDED Requirements

### Requirement: Industry metadata local configuration
The system SHALL maintain a local static configuration containing all 31 SW Level-1 industry codes and names (2021 version). The configuration SHALL be stored as a Python dictionary or JSON file, and SHALL NOT require network requests to load.

#### Scenario: Load industry list
- **WHEN** the scraper module initializes
- **THEN** it SHALL load 31 industry entries from the local configuration, each with `code` (e.g., "801010") and `name` (e.g., "农林牧渔")

### Requirement: Download historical OHLCV data via Playwright
The system SHALL use Playwright to control the browser and download historical daily OHLCV data for each of the 31 industries from the SWS Research website. The download SHALL be triggered by making a POST request to `/insWechatSw/swIndex/quotationexportExc` with `{swindexcode: "<code>"}` as the request body. The system SHALL parse the returned XLS file and extract the following fields: 指数代码, 指数名称, 发布日期, 开盘指数, 最高指数, 最低指数, 收盘指数, 成交量(亿股), 成交额(亿元).

#### Scenario: Full download for a single industry
- **WHEN** the scraper is invoked for industry code "801010"
- **THEN** it SHALL navigate to the SWS release detail page for that industry, click "行情下载", and save the XLS file
- **THEN** it SHALL parse the XLS into a pandas DataFrame with columns: date, open, high, low, close, volume, amount

#### Scenario: Download all 31 industries
- **WHEN** the scraper is invoked in batch mode
- **THEN** it SHALL iterate through all 31 industries from the local configuration
- **THEN** it SHALL pause 2-3 seconds between each download to avoid rate limiting

### Requirement: Incremental data update
The system SHALL support incremental updates by detecting the latest date in the existing local CSV and only downloading data newer than that date. If no local data exists, it SHALL perform a full download.

#### Scenario: Incremental update with existing data
- **WHEN** local CSV for "801010" already contains data up to 2026-02-20
- **THEN** the scraper SHALL download the full XLS but only append rows with dates after 2026-02-20 to the local CSV

#### Scenario: First-time download
- **WHEN** no local CSV exists for an industry
- **THEN** the scraper SHALL download the full XLS and save all rows as a new CSV file

### Requirement: Data persistence in CSV format
The system SHALL store daily OHLCV data as CSV files, one per industry, in the `data/daily/` directory. File naming convention SHALL be `{code}_{name}.csv` (e.g., `801010_农林牧渔.csv`).

#### Scenario: CSV file structure
- **WHEN** data is saved for an industry
- **THEN** the CSV SHALL contain columns: date, open, high, low, close, volume, amount
- **THEN** rows SHALL be sorted by date in ascending order
- **THEN** the date column SHALL be in `YYYY-MM-DD` format

### Requirement: Error handling and retry
The system SHALL handle download failures gracefully. If a download fails for a single industry, it SHALL log the error and continue with the next industry. It SHALL retry failed downloads up to 3 times with exponential backoff.

#### Scenario: Single industry download failure
- **WHEN** the download for industry "801030" fails due to network error
- **THEN** the system SHALL retry up to 3 times with increasing delay
- **THEN** if all retries fail, it SHALL log the error and proceed to the next industry
- **THEN** it SHALL report a summary of failed industries at the end
