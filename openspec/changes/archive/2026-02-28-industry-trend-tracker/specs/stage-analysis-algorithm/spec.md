## ADDED Requirements

### Requirement: Price position relative to MA34
The system SHALL calculate the price position indicator as: `price_position = (close - MA34) / MA34`. This represents the percentage deviation of the closing price from the 34-week moving average.

#### Scenario: Price above MA34
- **WHEN** weekly close is 3000 and MA34 is 2800
- **THEN** price_position SHALL be approximately 0.0714 (7.14%)

#### Scenario: Price below MA34
- **WHEN** weekly close is 2600 and MA34 is 2800
- **THEN** price_position SHALL be approximately -0.0714 (-7.14%)

### Requirement: MA34 crossover detection
The system SHALL detect crossover events where the weekly close price crosses above or below the MA34 line. A cross-above occurs when the previous week's close was below MA34 and the current week's close is above MA34 (and vice versa for cross-below). The system SHALL track crossover frequency over a rolling window (configurable, default 12 weeks).

#### Scenario: Bullish crossover
- **WHEN** last week's close was below MA34 and this week's close is above MA34
- **THEN** the system SHALL record a "cross_above" event

#### Scenario: Bearish crossover
- **WHEN** last week's close was above MA34 and this week's close is below MA34
- **THEN** the system SHALL record a "cross_below" event

#### Scenario: High crossover frequency
- **WHEN** there have been 3 or more crossover events in the past 12 weeks
- **THEN** the crossover frequency SHALL be classified as "high" (indicating a transitional stage)

#### Scenario: Low crossover frequency
- **WHEN** there have been 0-1 crossover events in the past 12 weeks
- **THEN** the crossover frequency SHALL be classified as "low" (indicating a trending stage)

### Requirement: Volume ratio calculation
The system SHALL calculate a volume ratio as: `volume_ratio = short_term_avg_volume / long_term_avg_volume`, where short_term is the average volume of the past 4 weeks and long_term is the average volume of the past 34 weeks.

#### Scenario: Volume expansion
- **WHEN** recent 4-week average volume is 150% of the 34-week average
- **THEN** volume_ratio SHALL be 1.5, indicating volume expansion

#### Scenario: Volume contraction
- **WHEN** recent 4-week average volume is 60% of the 34-week average
- **THEN** volume_ratio SHALL be 0.6, indicating volume contraction

### Requirement: Weinstein four-stage classification
The system SHALL classify each industry into one of four Weinstein stages based on the combination of price_position, ma_slope, crossover_frequency, and volume_ratio. The classification logic SHALL be:

**Stage 1 (Bear Market / 熊市)**:
- price_position < -threshold (default -2%) for the majority of recent weeks
- ma_slope <= 0 (MA34 flat or declining)
- crossover_frequency is "low" (price stays below MA34)
- volume_ratio < 1.0 (volume contracting)

**Stage 2 (Bear-Bull Transition / 熊牛转换)**:
- price_position oscillates around zero (between -threshold and +threshold)
- ma_slope transitions from negative toward zero or slightly positive
- crossover_frequency is "high" (price repeatedly crosses MA34)
- volume_ratio gradually increasing (> 0.8 and trending up)

**Stage 3 (Bull Market / 牛市)**:
- price_position > +threshold (default +2%) for the majority of recent weeks
- ma_slope > 0 (MA34 rising)
- crossover_frequency is "low" (price stays above MA34)
- volume_ratio >= 1.0 (volume active)

**Stage 4 (Bull-Bear Transition / 牛熊转换)**:
- price_position oscillates around zero (between -threshold and +threshold)
- ma_slope transitions from positive toward zero or slightly negative
- crossover_frequency is "high" (price repeatedly crosses MA34)
- volume_ratio unstable or declining

#### Scenario: Clear bear market (Stage 1)
- **WHEN** price has been below MA34 for 8+ consecutive weeks, MA34 is declining, no crossovers in 12 weeks, volume below average
- **THEN** the system SHALL classify the industry as Stage 1 with high confidence (>= 0.8)

#### Scenario: Bear-bull transition (Stage 2)
- **WHEN** price has crossed MA34 3 times in the past 12 weeks, MA34 slope is near zero, volume gradually increasing
- **THEN** the system SHALL classify the industry as Stage 2

#### Scenario: Clear bull market (Stage 3)
- **WHEN** price has been above MA34 for 8+ consecutive weeks, MA34 is rising, no crossovers in 12 weeks, volume above average
- **THEN** the system SHALL classify the industry as Stage 3 with high confidence (>= 0.8)

#### Scenario: Bull-bear transition (Stage 4)
- **WHEN** price has crossed MA34 3 times in the past 12 weeks, MA34 slope turning from positive to flat/negative, volume unstable
- **THEN** the system SHALL classify the industry as Stage 4

### Requirement: Confidence score output
The system SHALL output a confidence score between 0.0 and 1.0 for each stage classification. Higher confidence indicates stronger alignment of all indicators with the stage definition. When indicators conflict (e.g., price above MA34 but MA slope is negative), confidence SHALL be lower.

#### Scenario: High confidence classification
- **WHEN** all four indicators (price_position, ma_slope, cross_frequency, volume_ratio) consistently point to the same stage
- **THEN** confidence SHALL be >= 0.8

#### Scenario: Low confidence classification
- **WHEN** indicators are mixed (e.g., 2 indicators suggest Stage 3, 2 suggest Stage 4)
- **THEN** confidence SHALL be < 0.6

### Requirement: Configurable thresholds
The system SHALL allow key algorithm parameters to be configured: price_position threshold (default 2%), MA slope flat threshold (default 0.5%), crossover frequency window (default 12 weeks), crossover frequency high threshold (default 3), short-term volume window (default 4 weeks), consecutive weeks threshold for trending stages (default 8 weeks).

#### Scenario: Custom threshold configuration
- **WHEN** the user sets price_position threshold to 3% instead of the default 2%
- **THEN** the stage classification SHALL use 3% as the boundary for distinguishing trending vs transitional stages

### Requirement: Stage transition detection
The system SHALL detect when an industry transitions from one stage to another and record the transition date. This enables tracking of stage change history.

#### Scenario: Stage transition from 2 to 3
- **WHEN** an industry was classified as Stage 2 last week and is now classified as Stage 3
- **THEN** the system SHALL record this as a stage transition event with the current week's date
