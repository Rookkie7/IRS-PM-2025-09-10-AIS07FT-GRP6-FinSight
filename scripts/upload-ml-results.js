import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const supabaseUrl = process.env.VITE_SUPABASE_URL;
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing Supabase credentials in .env file');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

function parsePredictionsCSV(csvContent) {
  const lines = csvContent.trim().split('\n');
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const [trueValue, predictedValue] = lines[i].split(',');
    data.push({
      true_values: parseFloat(trueValue),
      predicted_values: parseFloat(predictedValue)
    });
  }

  return data;
}

function parseTradingResultsCSV(csvContent) {
  const lines = csvContent.trim().split('\n');
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const [strategy, threshold, totalProfit, profitPerTrade, sharpeRatio, tradesCount] = lines[i].split(',');
    data.push({
      strategy,
      threshold: parseFloat(threshold),
      total_profit: parseFloat(totalProfit),
      profit_per_trade: parseFloat(profitPerTrade),
      sharpe_ratio: parseFloat(sharpeRatio),
      trades_count: parseInt(tradesCount)
    });
  }

  return data;
}

async function uploadMLResults(pairName, modelName, dataDir) {
  try {
    const configPath = path.join(dataDir, 'config.json');
    const metricsPath = path.join(dataDir, 'metrics.json');
    const modelInfoPath = path.join(dataDir, 'model_info.json');
    const predictionsPath = path.join(dataDir, 'predictions.csv');
    const tradingResultsPath = path.join(dataDir, 'trading_results.csv');

    const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
    const modelInfo = JSON.parse(fs.readFileSync(modelInfoPath, 'utf-8'));
    const predictionsCSV = fs.readFileSync(predictionsPath, 'utf-8');
    const tradingResultsCSV = fs.readFileSync(tradingResultsPath, 'utf-8');

    const predictionsData = parsePredictionsCSV(predictionsCSV);
    const tradingResults = parseTradingResultsCSV(tradingResultsCSV);

    const { data: pairData, error: pairError } = await supabase
      .from('pair_info')
      .select('id')
      .eq('pair_name', pairName)
      .single();

    if (pairError) throw pairError;

    let modelId;
    const { data: existingModel } = await supabase
      .from('ml_models')
      .select('id')
      .eq('model_name', modelName)
      .maybeSingle();

    if (existingModel) {
      modelId = existingModel.id;
    } else {
      const { data: newModel, error: modelError } = await supabase
        .from('ml_models')
        .insert({ model_name: modelName, description: `${modelName.toUpperCase()} model for pairs trading` })
        .select('id')
        .single();

      if (modelError) throw modelError;
      modelId = newModel.id;
    }

    const { data: existingResult } = await supabase
      .from('ml_results')
      .select('id')
      .eq('pair_id', pairData.id)
      .eq('model_id', modelId)
      .maybeSingle();

    const resultData = {
      pair_id: pairData.id,
      model_id: modelId,
      config,
      prediction_errors: metrics.prediction_errors,
      portfolio_metrics: metrics.portfolio_metrics,
      model_info: modelInfo,
      predictions_data: predictionsData,
      trading_results: tradingResults
    };

    if (existingResult) {
      const { error } = await supabase
        .from('ml_results')
        .update(resultData)
        .eq('id', existingResult.id);

      if (error) throw error;
      console.log(`✓ Updated ML results for ${pairName} - ${modelName}`);
    } else {
      const { error } = await supabase
        .from('ml_results')
        .insert(resultData);

      if (error) throw error;
      console.log(`✓ Inserted ML results for ${pairName} - ${modelName}`);
    }

  } catch (error) {
    console.error(`✗ Error uploading results for ${pairName} - ${modelName}:`, error);
  }
}

const args = process.argv.slice(2);

if (args.length < 3) {
  console.log('Usage: node upload-ml-results.js <pair_name> <model_name> <data_directory>');
  console.log('Example: node upload-ml-results.js LLY_UNP tcn ./ml_results/LLY_UNP_tcn');
  process.exit(1);
}

const [pairName, modelName, dataDir] = args;

uploadMLResults(pairName, modelName, dataDir)
  .then(() => {
    console.log('Upload complete!');
    process.exit(0);
  })
  .catch(error => {
    console.error('Upload failed:', error);
    process.exit(1);
  });
