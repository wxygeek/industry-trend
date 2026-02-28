## ADDED Requirements

### Requirement: Aggregate daily K-lines into weekly K-lines
The system SHALL aggregate daily OHLCV data into weekly K-lines using natural week boundaries (Monday through Friday). Each weekly bar SHALL be computed as:
- Open: opening price of the first trading day of the week
- High: maximum of all daily highs within the week
- Low: minimum of all daily lows within the week
- Close: closing price of the last trading day of the week
- Volume: sum of all daily volumes within the week
- Amount: sum of all daily amounts within the week

#### Scenario: Normal full trading week
- **WHEN** daily data contains 5 trading days for week 2026-02-24 to 2026-02-28
- **THEN** the weekly bar SHALL use Monday's open, max of all highs, min of all lows, Friday's close, and sum of volumes and amounts

#### Scenario: Partial trading week due to holidays
- **WHEN** a week has only 3 trading days (e.g., Spring Festival week)
- **THEN** the weekly bar SHALL still be generated using the available trading days
- **THEN** open SHALL be the first available day's open, close SHALL be the last available day's close

#### Scenario: Current incomplete week
- **WHEN** today is Wednesday and the current week has only 3 trading days so far
- **THEN** the system SHALL generate a weekly bar for the current incomplete week using available data

### Requirement: Calculate 34-week simple moving average (MA34)
The system SHALL calculate a 34-week Simple Moving Average (SMA) based on weekly closing prices. The MA34 value for week N SHALL be the arithmetic mean of closing prices from week (N-33) through week N.

#### Scenario: Sufficient data for MA34
- **WHEN** at least 34 weeks of weekly K-line data are available
- **THEN** the system SHALL compute MA34 for each week starting from the 34th week
- **THEN** weeks before the 34th SHALL have MA34 value as NaN

#### Scenario: MA34 calculation accuracy
- **WHEN** weekly closes for the last 34 weeks are [C1, C2, ..., C34]
- **THEN** MA34 SHALL equal (C1 + C2 + ... + C34) / 34

### Requirement: Calculate MA34 slope
The system SHALL calculate the slope of the MA34 line to determine its trend direction. The slope SHALL be computed as the percentage change of MA34 over the past N weeks (configurable, default N=4): `slope = (MA34_current - MA34_N_weeks_ago) / MA34_N_weeks_ago`.

#### Scenario: Rising MA34
- **WHEN** MA34 has been consistently increasing over the past 4 weeks
- **THEN** ma_slope SHALL be a positive value

#### Scenario: Flat MA34
- **WHEN** MA34 has barely changed over the past 4 weeks (absolute slope < 0.5%)
- **THEN** the slope SHALL be classified as "flat"

### Requirement: Output standardized weekly dataset
The system SHALL output a standardized weekly K-line dataset for each industry containing: date, open, high, low, close, volume, amount, ma34, ma_slope. The dataset SHALL be available as a pandas DataFrame.

#### Scenario: Complete weekly dataset
- **WHEN** weekly K-line generation is run for industry "801010"
- **THEN** the output DataFrame SHALL contain columns: date, open, high, low, close, volume, amount, ma34, ma_slope
- **THEN** all numeric columns SHALL be float type
- **THEN** date column SHALL be datetime type
