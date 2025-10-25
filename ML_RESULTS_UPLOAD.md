# Uploading ML Results to Database

This guide explains how to upload your ML model test results to the Supabase database for visualization in the Pairs Trading frontend.

## Prerequisites

1. Make sure you have the `.env` file configured with Supabase credentials:
   ```
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

2. Install dotenv if not already installed:
   ```bash
   npm install dotenv
   ```

## Data Format

For each ML model test, you need 5 files in a directory:

1. **config.json** - Model configuration
   ```json
   {
     "model_name": "tcn",
     "input_chunk_length": 50,
     "output_chunk_length": 1,
     "n_epochs": 50,
     "batch_size": 1024,
     "train_ratio": 0.5,
     "thresholds": [0.0, 0.00025, 0.0005, 0.001],
     "data_file_path": "../data/sp500_pairs/LLY_UNP.csv",
     "load_model_path": "results/experiment_20251024_025124/saved_model"
   }
   ```

2. **metrics.json** - Prediction errors and portfolio metrics
   ```json
   {
     "prediction_errors": {
       "RMSE": 0.0075929910296090464,
       "MASE": 1.3213001846010504,
       "MAPE": 1.924845251314097,
       "sMAPE": 1.9223240360636373
     },
     "portfolio_metrics": {
       "pure forcasting": {
         "0.0": {
           "Annualized CAGR": 6.332814676611709,
           "Annualized Vol": 11.027584810333884,
           "Sharpe": 0.43197863029694855,
           "Max Drawdown": 7.331418279252983,
           "Turnover": 0.02512820512820513
         },
         ...
       },
       ...
     }
   }
   ```

3. **model_info.json** - Model metadata
   ```json
   {
     "model_name": "tcn",
     "model_params": 0,
     "timestamp": "20251024_025538"
   }
   ```

4. **predictions.csv** - True vs predicted values
   ```csv
   true_values,predicted_values
   0.2713335405657445,0.277540304346282
   0.27418136799752396,0.2735801895715921
   ...
   ```

5. **trading_results.csv** - Trading performance by strategy
   ```csv
   strategy,threshold,total_profit,profit_per_trade,sharpe_ratio,trades_count
   pure forcasting,0.0,892.9059320569253,2.5511598058769294,0,350
   pure forcasting,0.00025,1400.7916874810912,4.2319990558341125,0,331
   ...
   ```

## Upload Process

### Step 1: Organize Your Results

Create a directory structure like this:
```
ml_results/
├── LLY_UNP_tcn/
│   ├── config.json
│   ├── metrics.json
│   ├── model_info.json
│   ├── predictions.csv
│   └── trading_results.csv
├── LLY_UNP_lstm/
│   ├── config.json
│   ├── ...
└── ...
```

### Step 2: Run the Upload Script

For each model result, run:

```bash
node scripts/upload-ml-results.js <pair_name> <model_name> <data_directory>
```

**Examples:**

```bash
# Upload TCN results for LLY_UNP pair
node scripts/upload-ml-results.js LLY_UNP tcn ./ml_results/LLY_UNP_tcn

# Upload LSTM results for MO_WELL pair
node scripts/upload-ml-results.js MO_WELL lstm ./ml_results/MO_WELL_lstm

# Upload all results for a pair
node scripts/upload-ml-results.js LLY_UNP tcn ./ml_results/LLY_UNP_tcn
node scripts/upload-ml-results.js LLY_UNP lstm ./ml_results/LLY_UNP_lstm
node scripts/upload-ml-results.js LLY_UNP gru ./ml_results/LLY_UNP_gru
node scripts/upload-ml-results.js LLY_UNP transformer ./ml_results/LLY_UNP_transformer
node scripts/upload-ml-results.js LLY_UNP nbeats ./ml_results/LLY_UNP_nbeats
```

### Step 3: Verify Upload

The script will output:
- ✓ Success messages for each upload
- ✗ Error messages if something goes wrong

You can verify the data in Supabase dashboard:
1. Go to your Supabase project
2. Navigate to Table Editor
3. Check the `ml_results` table

## Supported Pairs

The database is pre-configured with these 4 pairs:
- `LLY_UNP` (LLY ↔ UNP)
- `MO_WELL` (MO ↔ WELL)
- `RTX_UBER` (RTX ↔ UBER)
- `XOM_LIN` (XOM ↔ LIN)

## Supported ML Models

The frontend supports these 5 ML methods:
- `tcn` - Temporal Convolutional Network
- `lstm` - Long Short-Term Memory
- `gru` - Gated Recurrent Units
- `transformer` - Transformer
- `nbeats` - N-BEATS

## Batch Upload Example

Create a bash script to upload all results:

```bash
#!/bin/bash

PAIRS=("LLY_UNP" "MO_WELL" "RTX_UBER" "XOM_LIN")
MODELS=("tcn" "lstm" "gru" "transformer" "nbeats")

for pair in "${PAIRS[@]}"; do
  for model in "${MODELS[@]}"; do
    dir="./ml_results/${pair}_${model}"
    if [ -d "$dir" ]; then
      echo "Uploading ${pair} - ${model}..."
      node scripts/upload-ml-results.js $pair $model $dir
    else
      echo "Skipping ${pair} - ${model} (directory not found)"
    fi
  done
done

echo "All uploads complete!"
```

Save as `upload_all.sh`, make executable with `chmod +x upload_all.sh`, and run `./upload_all.sh`.

## Troubleshooting

**Error: Missing Supabase credentials**
- Check your `.env` file has the correct Supabase URL and anon key

**Error: pair_name not found**
- Make sure you're using one of the 4 supported pair names exactly as listed above

**Error: File not found**
- Verify the data directory path is correct
- Ensure all 5 required files exist in the directory

**Error: Invalid JSON**
- Validate your JSON files are properly formatted
- Check for trailing commas or syntax errors

## Notes

- The script will automatically create the ML model entry if it doesn't exist
- If results already exist for a pair-model combination, they will be updated
- Large prediction datasets will be stored in JSONB format for efficient querying
- All timestamps are handled automatically by the database
