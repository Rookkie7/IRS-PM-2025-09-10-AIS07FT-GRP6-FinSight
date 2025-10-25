/*
  # Pairs Trading ML Results Schema

  1. New Tables
    - `pair_info`
      - `id` (uuid, primary key)
      - `pair_name` (text) - e.g., 'LLY_UNP'
      - `stock1` (text) - first stock symbol
      - `stock2` (text) - second stock symbol
      - `created_at` (timestamp)
    
    - `ml_models`
      - `id` (uuid, primary key)
      - `model_name` (text) - e.g., 'tcn', 'lstm'
      - `description` (text)
      - `created_at` (timestamp)
    
    - `ml_results`
      - `id` (uuid, primary key)
      - `pair_id` (uuid, foreign key to pair_info)
      - `model_id` (uuid, foreign key to ml_models)
      - `config` (jsonb) - model configuration
      - `prediction_errors` (jsonb) - RMSE, MASE, MAPE, sMAPE
      - `portfolio_metrics` (jsonb) - metrics for different strategies and thresholds
      - `model_info` (jsonb) - model params, timestamp
      - `predictions_data` (jsonb) - array of {true_values, predicted_values}
      - `trading_results` (jsonb) - array of trading results by strategy
      - `created_at` (timestamp)
  
  2. Security
    - Enable RLS on all tables
    - Add policies for public read access (since this is demo data)
*/

-- Create pair_info table
CREATE TABLE IF NOT EXISTS pair_info (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  pair_name text UNIQUE NOT NULL,
  stock1 text NOT NULL,
  stock2 text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE pair_info ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view pair info"
  ON pair_info FOR SELECT
  TO public
  USING (true);

-- Create ml_models table
CREATE TABLE IF NOT EXISTS ml_models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name text UNIQUE NOT NULL,
  description text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view ml models"
  ON ml_models FOR SELECT
  TO public
  USING (true);

-- Create ml_results table
CREATE TABLE IF NOT EXISTS ml_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  pair_id uuid REFERENCES pair_info(id) NOT NULL,
  model_id uuid REFERENCES ml_models(id) NOT NULL,
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  prediction_errors jsonb NOT NULL DEFAULT '{}'::jsonb,
  portfolio_metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
  model_info jsonb NOT NULL DEFAULT '{}'::jsonb,
  predictions_data jsonb NOT NULL DEFAULT '[]'::jsonb,
  trading_results jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz DEFAULT now(),
  UNIQUE(pair_id, model_id)
);

ALTER TABLE ml_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view ml results"
  ON ml_results FOR SELECT
  TO public
  USING (true);

-- Insert the 4 pairs
INSERT INTO pair_info (pair_name, stock1, stock2) VALUES
  ('LLY_UNP', 'LLY', 'UNP'),
  ('MO_WELL', 'MO', 'WELL'),
  ('RTX_UBER', 'RTX', 'UBER'),
  ('XOM_LIN', 'XOM', 'LIN')
ON CONFLICT (pair_name) DO NOTHING;