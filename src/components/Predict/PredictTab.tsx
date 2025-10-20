import React, { useEffect, useMemo, useState } from "react";
import { TrendingUp, TrendingDown, Calendar, Target, AlertCircle, BarChart } from "lucide-react";
import { fetchForecast, fetchForecastBatch } from "../../api/forecast";
import { toPredictionData, type PredictionData } from "../../utils/forecastMapper";

const symbols = ["AAPL", "TSLA", "NVDA"];
const companyNames = ["Apple Inc.", "Tesla Inc.", "NVIDIA Corp.", "Microsoft Corp."];

function relativeFromNow(iso: string) {
  const sec = Math.max(1, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (sec < 60) return `${sec} seconds ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  return `${hr} hours ago`;
}

export const PredictTab: React.FC = () => {
  const [list, setList] = useState<PredictionData[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbols[0]);
  const [selectedStock, setSelectedStock] = useState<PredictionData | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("1 Week");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);


  const companyName = useMemo(() => {
    const idx = symbols.indexOf(selectedSymbol);
    return companyNames[idx] ?? selectedSymbol;
  }, [selectedSymbol]);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setErr(null);
      try {
        // ✅ 一次拿回下拉里的所有股票预测
        const results = await fetchForecastBatch({
          method: 'lstm',
          // symbolsCsv: symbols.join(','),      // 确保包含 AAPL/TSLA/NVDA/MSFT
          horizons: [7, 30, 90, 180],
          // limit: symbols.length,              // 与 symbols 对齐
        }); // results: ForecastResult[]

        // ✅ 映射为 PredictionData[]
        const items: PredictionData[] = results.map(r => {
          const mapped = toPredictionData(r); // 这里 toPredictionData 直接用 r.company_name
          return {
            ...mapped,
            lastUpdated: relativeFromNow(mapped.lastUpdated),
          };
        });

        if (!alive) return;
        setList(items);

        // 选中当前 symbol 对应的那一项
        const match = items.find(x => x.symbol === selectedSymbol) ?? items[0] ?? null;
        setSelectedStock(match);

        // 校正周期按钮
        if (match) {
          const periods = match.predictions.map(p => p.period);
          if (!periods.includes(selectedPeriod)) {
            setSelectedPeriod(match.predictions[0]?.period ?? '1 Week');
          }
        }
      } catch (e: any) {
        if (alive) setErr(e.message || 'Load failed');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []); // ← 只在首次加载时批量拉取

  useEffect(() => {
    if (!list.length) return;
    const match = list.find(x => x.symbol === selectedSymbol) ?? null;
    setSelectedStock(match);
    if (match) {
      const periods = match.predictions.map(p => p.period);
      if (!periods.includes(selectedPeriod)) {
        setSelectedPeriod(match.predictions[0]?.period ?? '1 Week');
      }
    }
  }, [selectedSymbol, list]);


  const currentPrediction =
    selectedStock?.predictions.find((p) => p.period === selectedPeriod) ||
    selectedStock?.predictions[0] || {
      period: "1 Week",
      predictedPrice: selectedStock?.currentPrice ?? 0,
      confidence: 0.7,
      change: 0,
      changePercent: 0,
    };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Low": return "text-green-600 bg-green-50 border-green-200";
      case "Medium": return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "High": return "text-red-600 bg-red-50 border-red-200";
      default: return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Stock Predictions</h2>
          <p className="text-gray-600">AI-powered price forecasting with confidence intervals</p>
        </div>
        <div className="flex items-center space-x-4">
          {/* <select
            value={selectedSymbol}                 // 受控：symbol 决定请求
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {symbols.map((sym, i) => (
              <option key={sym} value={sym}>
                {sym} - {companyNames[i]}
              </option>
            ))}
          </select> */}
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {(list.length ? list.map(x => ({ sym: x.symbol, name: x.companyName })) 
                          : symbols.map((s, i) => ({ sym: s, name: companyNames[i] })))
              .map(item => (
                <option key={item.sym} value={item.sym}>
                  {item.sym} - {item.name}
                </option>
              ))}
          </select>

        </div>
      </div>

      {/* 错误与加载 */}
      {err && <div className="p-3 rounded bg-red-50 border border-red-200 text-red-700">{err}</div>}
      {loading && <div className="text-sm text-gray-500">Loading...</div>}

      {/* 没数据时不渲染后续块 */}
      {!selectedStock ? null : (
        <>
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
                    selectedPeriod === pred.period ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {pred.period}
                </button>
              ))}
            </div>

            {/* Prediction Details & List */}
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
                      <span
                        className={`font-semibold ${
                          currentPrediction.change > 0 ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {currentPrediction.change > 0 ? "+" : ""}${currentPrediction.change.toFixed(2)} ({currentPrediction.changePercent.toFixed(2)}%)
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
                      />
                    </div>
                    <span className="text-lg font-bold text-gray-900">{Math.round(currentPrediction.confidence * 100)}%</span>
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
                          <span className="text-sm font-semibold text-gray-900">${pred.predictedPrice.toFixed(2)}</span>
                          <div className="text-xs text-gray-500">{Math.round(pred.confidence * 100)}% confidence</div>
                        </div>
                      </div>
                    ))}
                    {!selectedStock.predictions.length && <div className="text-sm text-gray-500">No predictions yet.</div>}
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
        </>
      )}
    </div>
  );
};
