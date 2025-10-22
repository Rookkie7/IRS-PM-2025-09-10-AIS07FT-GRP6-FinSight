import React from 'react';
import { X, BarChart3, TrendingUp, DollarSign, Zap, Target } from 'lucide-react';

interface StockRawData {
  symbol: string;
  basic_info?: {
    name?: string;
    sector?: string;
    industry?: string;
    market_cap?: number;
    currency?: string;
    country?: string;
  };
  financials?: {
    key_ratios?: {
      profit_margin?: number;
      revenue_growth?: number;
      return_on_equity?: number;
      debt_to_equity?: number;
      current_ratio?: number;
      operating_margin?: number;
      gross_margin?: number;
      earnings_growth?: number;
      beta?: number;
    };
    valuation_metrics?: {
      market_cap?: number;
      trailing_pe?: number;
      forward_pe?: number;
      price_to_sales?: number;
      price_to_book?: number;
      enterprise_value?: number;
    };
    dividend_info?: {
      dividend_yield?: number;
      payout_ratio?: number;
    };
  };
  historical_data?: {
    time_series?: Array<{
      date: string;
      open: number;
      close: number;
      high: number;
      low: number;
      volume: number;
    }>;
    volatility_30d?: number;
    volatility_90d?: number;
    momentum_1m?: number;
    momentum_3m?: number;
    volume_avg_30d?: number;
  };
  descriptions?: {
    business_summary?: string;
  };
}

interface ComparisonModalProps {
  stocks: StockRawData[];
  onClose: () => void;
}

export const ComparisonModal: React.FC<ComparisonModalProps> = ({ stocks, onClose }) => {
  const formatNumber = (num?: number) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toLocaleString('en-US', { maximumFractionDigits: 2 });
  };

  const formatPercent = (num?: number) => {
    if (num === undefined || num === null) return 'N/A';
    return `${(num * 100).toFixed(2)}%`;
  };

  const formatMarketCap = (marketCap?: number) => {
    if (!marketCap) return 'N/A';
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`;
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`;
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`;
    return `$${marketCap.toFixed(0)}`;
  };

  // 计算指标的最大值用于进度条显示
  const getMaxValue = (metric: keyof any) => {
    const values = stocks.map(stock => {
      if (metric === 'market_cap') return stock.basic_info?.market_cap || 0;
      if (metric === 'pe_ratio') return stock.financials?.valuation_metrics?.trailing_pe || 0;
      if (metric === 'volatility') return stock.historical_data?.volatility_30d || 0;
      if (metric === 'revenue_growth') return stock.financials?.key_ratios?.revenue_growth || 0;
      return 0;
    });
    return Math.max(...values, 1); // 确保不为0
  };

  if (stocks.length === 0) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-6 border-b sticky top-0 bg-white z-10">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center">
              <BarChart3 className="h-6 w-6 mr-2 text-blue-600" />
              Stock Comparison ({stocks.length} stocks)
            </h2>
            <p className="text-gray-600 mt-1">
              Detailed comparison of selected stocks
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="h-6 w-6 text-gray-500" />
          </button>
        </div>

        <div className="overflow-y-auto p-6">
          {/* 关键指标对比表 */}
          <div className="bg-white border border-gray-200 rounded-xl p-6 mb-6">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
              <Target className="h-5 w-5 mr-2 text-blue-600" />
              Key Metrics Comparison
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-4 font-medium text-gray-900">Metric</th>
                    {stocks.map(stock => (
                      <th key={stock.symbol} className="text-center py-4 font-medium text-gray-900 min-w-[120px]">
                        <div className="flex flex-col items-center">
                          <span className="font-bold text-lg">{stock.symbol}</span>
                          <span className="text-xs text-gray-500 mt-1">{stock.basic_info?.sector || 'N/A'}</span>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  <tr>
                    <td className="py-4 font-medium text-gray-700 flex items-center">
                      <DollarSign className="h-4 w-4 mr-2 text-green-600" />
                      Market Cap
                    </td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center font-semibold">
                        {formatMarketCap(stock.basic_info?.market_cap)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">P/E Ratio</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatNumber(stock.financials?.valuation_metrics?.trailing_pe)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">Forward P/E</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatNumber(stock.financials?.valuation_metrics?.forward_pe)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">Dividend Yield</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatPercent(stock.financials?.dividend_info?.dividend_yield)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">Revenue Growth</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        <span className={`font-medium ${
                          (stock.financials?.key_ratios?.revenue_growth || 0) > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatPercent(stock.financials?.key_ratios?.revenue_growth)}
                        </span>
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">Profit Margin</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatPercent(stock.financials?.key_ratios?.profit_margin)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">ROE</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatPercent(stock.financials?.key_ratios?.return_on_equity)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700 flex items-center">
                      <Zap className="h-4 w-4 mr-2 text-red-600" />
                      Beta
                    </td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatNumber(stock.financials?.key_ratios?.beta)}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">1M Momentum</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        <span className={`font-medium ${
                          (stock.historical_data?.momentum_1m || 0) > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatPercent(stock.historical_data?.momentum_1m)}
                        </span>
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">3M Momentum</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        <span className={`font-medium ${
                          (stock.historical_data?.momentum_3m || 0) > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatPercent(stock.historical_data?.momentum_3m)}
                        </span>
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-4 font-medium text-gray-700">30D Volatility</td>
                    {stocks.map(stock => (
                      <td key={stock.symbol} className="py-4 text-center">
                        {formatPercent(stock.historical_data?.volatility_30d)}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* 可视化对比 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* 估值对比 */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100">
              <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <DollarSign className="h-5 w-5 mr-2 text-blue-600" />
                Valuation Comparison
              </h4>
              <div className="space-y-4">
                {stocks.map(stock => {
                  const peRatio = stock.financials?.valuation_metrics?.trailing_pe || 0;
                  const width = Math.min((peRatio / 50) * 100, 100);
                  return (
                    <div key={stock.symbol} className="bg-white rounded-lg p-4 border border-blue-200 shadow-sm">
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-bold text-gray-900">{stock.symbol}</span>
                        <span className="text-sm font-semibold text-blue-600">
                          P/E: {formatNumber(peRatio)}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${width}%` }}
                        ></div>
                      </div>
                      <div className="flex justify-between text-xs text-gray-500 mt-2">
                        <span>0</span>
                        <span>25</span>
                        <span>50+</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 风险对比 */}
            <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-xl p-6 border border-red-100">
              <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2 text-red-600" />
                Risk & Volatility
              </h4>
              <div className="space-y-4">
                {stocks.map(stock => {
                  const volatility = stock.historical_data?.volatility_30d || 0;
                  const width = Math.min(volatility * 500, 100); // 放大显示差异
                  return (
                    <div key={stock.symbol} className="bg-white rounded-lg p-4 border border-red-200 shadow-sm">
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-bold text-gray-900">{stock.symbol}</span>
                        <span className="text-sm font-semibold text-red-600">
                          Vol: {formatPercent(volatility)}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${width}%` }}
                        ></div>
                      </div>
                      <div className="flex justify-between text-xs text-gray-500 mt-2">
                        <span>Low</span>
                        <span>Medium</span>
                        <span>High</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* 增长指标对比 */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-100">
            <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
              <TrendingUp className="h-5 w-5 mr-2 text-green-600" />
              Growth Metrics
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {stocks.map(stock => (
                <div key={stock.symbol} className="bg-white rounded-lg p-4 border border-green-200 text-center">
                  <div className="font-bold text-gray-900 text-lg mb-2">{stock.symbol}</div>
                  <div className="space-y-2">
                    <div>
                      <div className="text-sm text-gray-600">Revenue Growth</div>
                      <div className={`text-lg font-bold ${
                        (stock.financials?.key_ratios?.revenue_growth || 0) > 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatPercent(stock.financials?.key_ratios?.revenue_growth)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Profit Margin</div>
                      <div className="text-lg font-bold text-blue-600">
                        {formatPercent(stock.financials?.key_ratios?.profit_margin)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">ROE</div>
                      <div className="text-lg font-bold text-purple-600">
                        {formatPercent(stock.financials?.key_ratios?.return_on_equity)}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 总结分析 */}
          <div className="bg-gray-50 rounded-xl p-6 mt-6">
            <h4 className="text-lg font-bold text-gray-900 mb-3">Summary Analysis</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="font-medium text-gray-900 mb-2">Highest Valuation</div>
                <div className="text-lg font-bold text-blue-600">
                  {stocks.reduce((max, stock) => {
                    const currentPE = stock.financials?.valuation_metrics?.trailing_pe || 0;
                    const maxPE = max.financials?.valuation_metrics?.trailing_pe || 0;
                    return currentPE > maxPE ? stock : max;
                  }, stocks[0]).symbol}
                </div>
              </div>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="font-medium text-gray-900 mb-2">Lowest Risk</div>
                <div className="text-lg font-bold text-green-600">
                  {stocks.reduce((min, stock) => {
                    const currentVol = stock.historical_data?.volatility_30d || Infinity;
                    const minVol = min.historical_data?.volatility_30d || Infinity;
                    return currentVol < minVol ? stock : min;
                  }, stocks[0]).symbol}
                </div>
              </div>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="font-medium text-gray-900 mb-2">Best Growth</div>
                <div className="text-lg font-bold text-purple-600">
                  {stocks.reduce((max, stock) => {
                    const currentGrowth = stock.financials?.key_ratios?.revenue_growth || 0;
                    const maxGrowth = max.financials?.key_ratios?.revenue_growth || 0;
                    return currentGrowth > maxGrowth ? stock : max;
                  }, stocks[0]).symbol}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};