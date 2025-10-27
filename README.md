# Thesis Analysis Pipeline: Cross-Regional Cryptocurrency Arbitrage Analysis

This project implements a comprehensive analysis pipeline for investigating cross-regional cryptocurrency arbitrage opportunities.

## Overview

The analysis covers:
1. **Data Loading & Spread Construction**: Import and prepare cryptocurrency price data from multiple exchanges
2. **Descriptive Statistics**: Summary statistics and visualizations of arbitrage spreads
3. **Persistence & Co-movement Analysis**: Autocorrelation and cross-correlation analysis
4. **Econometric Models**: Panel regressions examining spread determinants
5. **Robustness Checks**: Alternative specifications and leave-one-out analysis

## Requirements

### Python Version
Python 3.8 or higher is required.

### Dependencies
Install the required packages using:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd thesis-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the required data file: `descriptive_stats.py` (or adjust the import in the script)

## Usage

### Running the Analysis

Run the complete analysis pipeline:
```bash
python thesis_analysis_pipeline.py
```

### Output

All outputs are saved to `./thesis_outputs/` with the following structure:

```
thesis_outputs/
├── figures/
│   ├── hist_spreads_zero_lag.png          # Histograms of spreads
│   ├── timeseries_spreads_zero_lag.png     # Time series plots
│   ├── corr_heatmap_spreads_zero_lag.png   # Correlation heatmap
│   ├── cluster_corr_spreads_zero_lag.png   # Clustered correlation
│   └── acf_spreads_zero_lag.png            # Autocorrelation plots
└── tables/
    ├── spread_zero_lag_summary.csv         # Summary statistics
    └── spread_zero_lag_corr.csv            # Cross-correlation table
```

## Configuration

The script can be configured by modifying the constants at the top of `thesis_analysis_pipeline.py`:

- `OUTPUT_DIR`: Directory for saving outputs (default: `./thesis_outputs`)
- `CHINN_ITO_INDEX`: KAOPEN index values for capital controls analysis
- Various other parameters for volatility windows, confirmation times, etc.

## Project Structure

```
.
├── thesis_analysis_pipeline.py  # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── descriptive_stats.py         # Data preparation module (required)
└── thesis_outputs/              # Generated outputs
    ├── figures/
    └── tables/
```

## Key Functions

### Data Loading
- `load_data()`: Loads cryptocurrency price data

### Descriptive Analysis
- `create_descriptive_statistics()`: Creates summary statistics and visualizations
- `analyze_persistence()`: Analyzes autocorrelation and co-movement

### Panel Analysis
- `create_panel_dataset()`: Creates panel dataset for regression analysis
- `add_premiums_to_panel()`: Computes and adds all premium components

### Premium Computation
- `compute_latency_premium()`: Computes latency premium from BTC volatility
- `compute_liquidity_premium()`: Computes liquidity premium from bid-ask spreads
- `compute_fees_premium()`: Returns exchange-specific fee structures
- `compute_fx_friction_premium()`: Computes FX friction premium from FX volatility
- `compute_capital_controls_premium()`: Computes capital controls premium using KAOPEN

### Robustness
- `run_robustness_checks()`: Performs robustness analysis

## Customization

To adapt this script for your own data:

1. **Data Format**: Ensure your data is in a similar format with columns like:
   - `timestamp`: Datetime column
   - `{exchange}_ask`, `{exchange}_bid`: Price columns for each exchange
   - `kraken_{exchange}_arbitrage`: Arbitrage spread columns

2. **Exchange Mapping**: Update the exchange mappings in the premium computation functions if needed

3. **Output Paths**: Modify the `OUTPUT_DIR` constant to save outputs to your preferred location

## Notes

- The script assumes the presence of a `descriptive_stats.py` module that provides a `final_df` variable. You'll need to create this module or modify the import statement.
- All hardcoded paths from the original notebook have been removed. Make sure to adjust paths as needed for your environment.
- The script suppresses warnings for cleaner output. Comment out the `warnings.filterwarnings('ignore')` line to see them.


## Author

Janek Masojada Edwards
