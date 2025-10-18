import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Calendar, Target, AlertCircle, BarChart } from 'lucide-react';
import { fetchForecast } from '../../api/forecast';
import { toPredictionData, type PredictionData } from '../../utils/forecastMapper';

const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT'];
const companyNames = ['Apple Inc.', 'Tesla Inc.', 'NVIDIA Corp.', 'Microsoft Corp.'];
const currentPrices = [195.12, 248.33, 612.45, 370.88];
const risks: ('Low' | 'Medium' | 'High')[] = ['Low', 'Medium', 'Low', 'Low'];
const accuracies = [0.89, 0.82, 0.86, 0.91];
const lastUpdates = ['2 minutes ago', '5 minutes ago', '3 minutes ago', '1 minute ago'];



interface PredictionData {
  symbol: string;
  companyName: string;
  currentPrice: number;
  predictions: {
    period: string;
    predictedPrice: number;
    confidence: number;
    change: number;
    changePercent: number;
  }[];
  risk: 'Low' | 'Medium' | 'High';
  accuracy: number;
  lastUpdated: string;
}

function relativeFromNow(iso: string) {
  const sec = Math.max(1, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (sec < 60) return `${sec} seconds ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  return `${hr} hours ago`;
}


const mockPredictions: PredictionData[] = [

  {
    symbol: symbols[0],
    companyName: companyNames[0],
    currentPrice: currentPrices[0],
    predictions: [
      { period: '1 Week', predictedPrice: 198.45, confidence: 0.85, change: 3.33, changePercent: 1.71 },
      { period: '1 Month', predictedPrice: 205.20, confidence: 0.78, change: 10.08, changePercent: 5.17 },
      { period: '3 Months', predictedPrice: 218.90, confidence: 0.65, change: 23.78, changePercent: 12.19 },
      { period: '6 Months', predictedPrice: 235.60, confidence: 0.52, change: 40.48, changePercent: 20.75 },
    ],
    risk: 'Low',
    accuracy: accuracies[0],
    lastUpdated: '2 minutes ago'
  },
  {
    symbol: symbols[1],
    companyName: companyNames[1],
    currentPrice: currentPrices[1],
    predictions: [
      { period: '1 Week', predictedPrice: 252.10, confidence: 0.72, change: 3.60, changePercent: 1.45 },
      { period: '1 Month', predictedPrice: 265.30, confidence: 0.68, change: 16.80, changePercent: 6.76 },
      { period: '3 Months', predictedPrice: 290.75, confidence: 0.58, change: 42.25, changePercent: 17.00 },
      { period: '6 Months', predictedPrice: 315.20, confidence: 0.45, change: 66.70, changePercent: 26.84 },
    ],
    risk: 'High',
    accuracy: accuracies[1],
    lastUpdated: '5 minutes ago'
  },
  {
    symbol: symbols[2],
    companyName: companyNames[2],
    currentPrice: currentPrices[2],
    predictions: [
      { period: '1 Week', predictedPrice: 252.10, confidence: 0.72, change: 3.60, changePercent: 1.45 },
      { period: '1 Month', predictedPrice: 265.30, confidence: 0.68, change: 16.80, changePercent: 6.76 },
      { period: '3 Months', predictedPrice: 290.75, confidence: 0.58, change: 42.25, changePercent: 17.00 },
      { period: '6 Months', predictedPrice: 315.20, confidence: 0.45, change: 66.70, changePercent: 26.84 },
    ],
    risk: 'Medium',
    accuracy: accuracies[2],
    lastUpdated: '5 minutes ago'
  },
  {
    symbol: symbols[3],
    companyName: companyNames[3],
    currentPrice: currentPrices[3],
    predictions: [
      { period: '1 Week', predictedPrice: 252.10, confidence: 0.72, change: 3.60, changePercent: 1.45 },
      { period: '1 Month', predictedPrice: 265.30, confidence: 0.68, change: 16.80, changePercent: 6.76 },
    ],
    risk: 'Low',
    accuracy: accuracies[3],
    lastUpdated: '1 minute ago'
  },

];
// Frontend Website page

export const PredictTab: React.FC = () => {
  // const [selectedStock, setSelectedStock] = useState(mockPredictions[0]);
  // const [selectedPeriod, setSelectedPeriod] = useState('1 Month');
  const [selectedStock, setSelectedStock] = useState<PredictionData>(mockPredictions[1]);
  const [selectedPeriod, setSelectedPeriod] = useState('1 Month');

  const [selectedSymbol, setSelectedSymbol] = useState(symbols[1]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'text-green-600 bg-green-50 border-green-200';
      case 'Medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'High': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const currentPrediction = selectedStock.predictions.find(p => p.period === selectedPeriod) || selectedStock.predictions[1];
  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        const backend = await fetchForecast(selectedSymbol, [7, 30, 90, 180], 'naive-drift');
        // toPredictionData 里没有 lastUpdated 字段，这里补充一下
        const mapped = toPredictionData(backend, companyNames[symbols.indexOf(selectedSymbol)] || '');
        const withUpdated: PredictionData = {
          ...mapped,
          lastUpdated: relativeFromNow(backend.generated_at),
        };
        if (alive) {
          setSelectedStock(withUpdated);
          // 如果当前选中的 period 不在返回里，回退到第一个
          const periods = withUpdated.predictions.map(p => p.period);
          if (!periods.includes(selectedPeriod)) setSelectedPeriod(withUpdated.predictions[0]?.period ?? '1 Week');
        }
      } catch (e: any) {
        if (alive) setErr(e.message || 'Load failed');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [selectedSymbol]);



  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Stock Predictions</h2>
          <p className="text-gray-600">AI-powered price forecasting with confidence intervals</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedStock.symbol}
            // onChange={(e) => {
            //   const stock = mockPredictions.find(p => p.symbol === e.target.value);
            //   if (stock) setSelectedStock(stock);
            // }}
            onChange={(e) => {
              const sym = e.target.value;
              setSelectedSymbol(sym);     // 👉 修改：只更新 symbol，数据由 useEffect 拉取
            }}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {mockPredictions.map((stock) => (
              <option key={stock.symbol} value={stock.symbol}>
                {stock.symbol} - {stock.companyName}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Model Performance */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="h-5 w-5 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">Model Accuracy</span>
          </div>
          <p className="text-2xl font-bold text-blue-800">{Math.round(selectedStock.accuracy * 100)}%</p>
          <p className="text-xs text-blue-600">Last 90 days</p>
        </div>
        
        <div className={`border rounded-lg p-4 ${getRiskColor(selectedStock.risk)}`}>
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-medium">Risk Level</span>
          </div>
          <p className="text-2xl font-bold">{selectedStock.risk}</p>
          <p className="text-xs">Volatility based</p>
        </div>
        
        <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Calendar className="h-5 w-5 text-green-600" />
            <span className="text-sm font-medium text-green-700">Best Timeframe</span>
          </div>
          <p className="text-2xl font-bold text-green-800">1-4 Weeks</p>
          <p className="text-xs text-green-600">Highest accuracy</p>
        </div>
        
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart className="h-5 w-5 text-purple-600" />
            <span className="text-sm font-medium text-purple-700">Data Points</span>
          </div>
          <p className="text-2xl font-bold text-purple-800">2,847</p>
          <p className="text-xs text-purple-600">Updated {selectedStock.lastUpdated}</p>
        </div>
      </div>

      {/* Current Stock Info */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
              <span className="text-lg font-bold text-gray-700">{selectedStock.symbol.charAt(0)}</span>
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900">{selectedStock.symbol}</h3>
              <p className="text-gray-600">{selectedStock.companyName}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-gray-900">${selectedStock.currentPrice.toFixed(2)}</p>
            <p className="text-sm text-gray-600">Current Price</p>
          </div>
        </div>

        {/* Time Period Selector */}
        <div className="flex space-x-2 mb-6">
          {selectedStock.predictions.map((pred) => (
            <button
              key={pred.period}
              onClick={() => setSelectedPeriod(pred.period)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedPeriod === pred.period
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {pred.period}
            </button>
          ))}
        </div>

        {/* Prediction Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Price Prediction ({selectedPeriod})</h4>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-600">Predicted Price</span>
                <span className="text-xl font-bold text-gray-900">
                  ${currentPrediction.predictedPrice.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-600">Expected Change</span>
                <div className="flex items-center space-x-1">
                  {currentPrediction.change > 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-600" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-600" />
                  )}
                  <span className={`font-semibold ${
                    currentPrediction.change > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {currentPrediction.change > 0 ? '+' : ''}${currentPrediction.change.toFixed(2)} ({currentPrediction.changePercent.toFixed(2)}%)
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Confidence Level</h4>
              <div className="flex items-center space-x-3">
                <div className="flex-1 bg-gray-200 rounded-full h-3">
                  <div 
                    className="bg-blue-600 h-3 rounded-full transition-all duration-500" 
                    style={{ width: `${currentPrediction.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="text-lg font-bold text-gray-900">
                  {Math.round(currentPrediction.confidence * 100)}%
                </span>
              </div>
              <p className="text-xs text-gray-600 mt-2">
                Based on historical patterns, market sentiment, and technical indicators
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-3">All Predictions</h4>
              <div className="space-y-3">
                {selectedStock.predictions.map((pred) => (
                  <div key={pred.period} className="flex items-center justify-between py-2 border-b border-gray-200 last:border-b-0">
                    <span className="text-sm text-gray-600">{pred.period}</span>
                    <div className="text-right">
                      <span className="text-sm font-semibold text-gray-900">
                        ${pred.predictedPrice.toFixed(2)}
                      </span>
                      <div className="text-xs text-gray-500">
                        {Math.round(pred.confidence * 100)}% confidence
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-900 mb-2">💡 Key Insights</h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Higher accuracy for shorter-term predictions</li>
                <li>• Model incorporates 15+ technical indicators</li>
                <li>• Real-time sentiment analysis included</li>
                <li>• Risk-adjusted confidence scoring</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};