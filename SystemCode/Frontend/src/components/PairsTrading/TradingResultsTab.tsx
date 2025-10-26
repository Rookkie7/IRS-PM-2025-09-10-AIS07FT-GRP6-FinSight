import { useState, useEffect } from 'react';
// import { supabase } from '../../api/http';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, BarChart3, Brain, Activity } from 'lucide-react';

interface PairInfo {
  id: string;
  pair_name: string;
  stock1: string;
  stock2: string;
}

interface MLResult {
  id: string;
  model_name: string;
  config: any;
  prediction_errors: any;
  portfolio_metrics: any;
  model_info: any;
  predictions_data: any[];
  trading_results: any[];
}

export default function TradingResultsTab() {
  const [pairs, setPairs] = useState<PairInfo[]>([]);
  const [selectedPair, setSelectedPair] = useState<PairInfo | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<string>('');
  const [mlResults, setMLResults] = useState<MLResult[]>([]);
  const [loading, setLoading] = useState(true);

  const basicMethods = [
    { id: 'distance', name: 'Distance Method', description: 'Trade based on normalized price distance' },
    { id: 'cointegration', name: 'Cointegration Method', description: 'Use cointegration residuals for signals' },
    { id: 'correlation', name: 'Correlation Method', description: 'Trade on correlation breakdowns' },
    { id: 'kalman', name: 'Kalman Filter', description: 'Dynamic hedge ratio estimation' }
  ];

  const mlMethods = [
    { id: 'tcn', name: 'TCN (Temporal Convolutional Network)', description: 'Deep learning for time series prediction' },
    { id: 'lstm', name: 'LSTM', description: 'Long Short-Term Memory networks' },
    { id: 'gru', name: 'GRU', description: 'Gated Recurrent Units' },
    { id: 'transformer', name: 'Transformer', description: 'Attention-based architecture' },
    { id: 'nbeats', name: 'N-BEATS', description: 'Neural basis expansion analysis' }
  ];

  useEffect(() => {
    fetchPairs();
  }, []);

  useEffect(() => {
    if (selectedPair) {
      fetchMLResults(selectedPair.id);
    }
  }, [selectedPair]);

  const fetchPairs = async () => {
    try {
      const { data, error } = await supabase
        .from('pair_info')
        .select('*')
        .order('pair_name');

      if (error) throw error;
      setPairs(data || []);
      if (data && data.length > 0) {
        setSelectedPair(data[0]);
      }
    } catch (error) {
      console.error('Error fetching pairs:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchMLResults = async (pairId: string) => {
    try {
      const { data, error } = await supabase
        .from('ml_results')
        .select(`
          *,
          model:ml_models(model_name)
        `)
        .eq('pair_id', pairId);

      if (error) throw error;
      setMLResults(data || []);
    } catch (error) {
      console.error('Error fetching ML results:', error);
    }
  };

  const renderBasicMethodCard = (method: any) => (
    <button
      key={method.id}
      onClick={() => setSelectedMethod(method.id)}
      className={`w-full text-left p-6 rounded-lg border-2 transition-all duration-200 ${
        selectedMethod === method.id
          ? 'border-blue-500 bg-blue-50 shadow-md'
          : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-sm'
      }`}
    >
      <div className="flex items-start gap-3">
        <Activity className="w-6 h-6 text-blue-600 mt-1 flex-shrink-0" />
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 mb-1">{method.name}</h4>
          <p className="text-sm text-gray-600">{method.description}</p>
          <div className="mt-3 text-xs text-gray-500 bg-gray-100 px-3 py-1 rounded-full inline-block">
            Click to run analysis
          </div>
        </div>
      </div>
    </button>
  );

  const renderMLMethodCard = (method: any) => {
    const result = mlResults.find(r => r.model_name === method.id);

    return (
      <button
        key={method.id}
        onClick={() => result && setSelectedMethod(method.id)}
        className={`w-full text-left p-6 rounded-lg border-2 transition-all duration-200 ${
          selectedMethod === method.id
            ? 'border-purple-500 bg-purple-50 shadow-md'
            : result
            ? 'border-gray-200 bg-white hover:border-purple-300 hover:shadow-sm'
            : 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-60'
        }`}
        disabled={!result}
      >
        <div className="flex items-start gap-3">
          <Brain className="w-6 h-6 text-purple-600 mt-1 flex-shrink-0" />
          <div className="flex-1">
            <h4 className="font-semibold text-gray-900 mb-1">{method.name}</h4>
            <p className="text-sm text-gray-600">{method.description}</p>
            {result ? (
              <div className="mt-3 text-xs text-green-600 bg-green-100 px-3 py-1 rounded-full inline-block">
                Results available
              </div>
            ) : (
              <div className="mt-3 text-xs text-gray-500 bg-gray-200 px-3 py-1 rounded-full inline-block">
                No pre-computed results
              </div>
            )}
          </div>
        </div>
      </button>
    );
  };

  const renderMLResults = () => {
    const result = mlResults.find(r => r.model_name === selectedMethod);
    if (!result) return null;

    const strategies = ['pure forcasting', 'mean reversion', 'hybrid', 'ground truth'];
    const thresholds = result.config?.thresholds || ['0.0', '0.00025', '0.0005', '0.001'];

    const performanceData = strategies.map(strategy => {
      const metrics = result.portfolio_metrics[strategy];
      if (!metrics) return null;

      return {
        strategy: strategy.charAt(0).toUpperCase() + strategy.slice(1),
        ...Object.keys(metrics).reduce((acc, threshold) => {
          acc[`CAGR_${threshold}`] = metrics[threshold]['Annualized CAGR'];
          acc[`Sharpe_${threshold}`] = metrics[threshold]['Sharpe'];
          acc[`MaxDD_${threshold}`] = metrics[threshold]['Max Drawdown'];
          return acc;
        }, {} as any)
      };
    }).filter(Boolean);

    const predictionData = result.predictions_data?.slice(0, 100).map((item: any, index: number) => ({
      index,
      actual: item.true_values,
      predicted: item.predicted_values
    })) || [];

    return (
      <div className="mt-8 space-y-8 animate-fadeIn">
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl p-6 border border-purple-200">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Model Configuration</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Model</div>
              <div className="font-semibold text-gray-900">{result.config?.model_name?.toUpperCase()}</div>
            </div>
            <div className="bg-white rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Input Length</div>
              <div className="font-semibold text-gray-900">{result.config?.input_chunk_length}</div>
            </div>
            <div className="bg-white rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Epochs</div>
              <div className="font-semibold text-gray-900">{result.config?.n_epochs}</div>
            </div>
            <div className="bg-white rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Batch Size</div>
              <div className="font-semibold text-gray-900">{result.config?.batch_size}</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Prediction Accuracy</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {Object.entries(result.prediction_errors || {}).map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">{key}</div>
                <div className="text-lg font-bold text-blue-600">{(value as number).toFixed(4)}</div>
              </div>
            ))}
          </div>

          {predictionData.length > 0 && (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" label={{ value: 'Time', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Price Spread', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#3b82f6" name="Actual" dot={false} />
                  <Line type="monotone" dataKey="predicted" stroke="#f59e0b" name="Predicted" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Portfolio Performance by Strategy</h3>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="strategy" />
                <YAxis label={{ value: 'Annualized CAGR (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                {thresholds.map((threshold, index) => (
                  <Bar
                    key={threshold}
                    dataKey={`CAGR_${threshold}`}
                    fill={['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'][index]}
                    name={`Threshold ${threshold}`}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {strategies.map(strategy => {
            const metrics = result.portfolio_metrics[strategy];
            if (!metrics) return null;

            return (
              <div key={strategy} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h4 className="font-bold text-gray-900 mb-4 capitalize">{strategy} Strategy</h4>
                <div className="space-y-3">
                  {Object.entries(metrics).map(([threshold, values]: [string, any]) => (
                    <div key={threshold} className="border-l-4 border-blue-500 pl-4 py-2 bg-gray-50 rounded">
                      <div className="text-sm font-semibold text-gray-700 mb-2">Threshold: {threshold}</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-600">CAGR:</span>
                          <span className="ml-2 font-semibold">{values['Annualized CAGR']?.toFixed(2)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Sharpe:</span>
                          <span className="ml-2 font-semibold">{values['Sharpe']?.toFixed(3)}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Max DD:</span>
                          <span className="ml-2 font-semibold">{values['Max Drawdown']?.toFixed(2)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Turnover:</span>
                          <span className="ml-2 font-semibold">{values['Turnover']?.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderBasicMethodPlaceholder = () => (
    <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-xl p-8 text-center">
      <Activity className="w-16 h-16 text-yellow-600 mx-auto mb-4" />
      <h3 className="text-xl font-bold text-gray-900 mb-2">Basic Method Execution</h3>
      <p className="text-gray-600 mb-4">
        Basic methods can be executed in real-time via API calls.
      </p>
      <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors">
        Run Analysis
      </button>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Pairs Trading Analysis</h2>
        <p className="text-gray-600 text-lg">
          Compare different trading methods across selected stock pairs
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Stock Pair</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {pairs.map(pair => (
            <button
              key={pair.id}
              onClick={() => {
                setSelectedPair(pair);
                setSelectedMethod('');
              }}
              className={`p-4 rounded-lg border-2 transition-all duration-200 ${
                selectedPair?.id === pair.id
                  ? 'border-blue-500 bg-blue-50 shadow-md'
                  : 'border-gray-200 bg-white hover:border-blue-300'
              }`}
            >
              <div className="flex items-center gap-2 justify-center mb-2">
                <span className="font-bold text-blue-600">{pair.stock1}</span>
                <span className="text-gray-400">↔</span>
                <span className="font-bold text-blue-600">{pair.stock2}</span>
              </div>
              <div className="text-xs text-gray-500 font-mono">{pair.pair_name}</div>
            </button>
          ))}
        </div>
      </div>

      {selectedPair && (
        <div className="space-y-8">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              Basic Trading Methods
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {basicMethods.map(renderBasicMethodCard)}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-600" />
              Machine Learning Methods
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {mlMethods.map(renderMLMethodCard)}
            </div>
          </div>

          {selectedMethod && (
            mlMethods.find(m => m.id === selectedMethod)
              ? renderMLResults()
              : renderBasicMethodPlaceholder()
          )}
        </div>
      )}
    </div>
  );
}
