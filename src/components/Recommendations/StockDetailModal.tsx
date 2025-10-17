import React from 'react';
import { X, DollarSign, TrendingUp, BarChart3, Calendar, Building2 } from 'lucide-react';

interface StockRawData {
  symbol: string;
  basic_info?: {
    name?: string;
    sector?: string;
    industry?: string;
    market_cap?: number;
    country?: string;
    currency?: string;
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

interface StockDetailModalProps {
  rawData: StockRawData | null;
  onClose: () => void;
}

export const StockDetailModal: React.FC<StockDetailModalProps> = ({ rawData, onClose }) => {
  if (!rawData) return null;

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

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-6 border-b sticky top-0 bg-white z-10">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{rawData.symbol}</h2>
            <p className="text-gray-600">{rawData.basic_info?.name || 'N/A'}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="h-6 w-6 text-gray-500" />
          </button>
        </div>

        <div className="overflow-y-auto p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Building2 className="h-5 w-5 text-blue-600" />
                <span className="text-sm font-medium text-blue-900">Basic Info</span>
              </div>
              <div className="space-y-1">
                <p className="text-sm"><span className="font-medium">Sector:</span> {rawData.basic_info?.sector || 'N/A'}</p>
                <p className="text-sm"><span className="font-medium">Industry:</span> {rawData.basic_info?.industry || 'N/A'}</p>
                <p className="text-sm"><span className="font-medium">Country:</span> {rawData.basic_info?.country || 'N/A'}</p>
                <p className="text-sm"><span className="font-medium">Currency:</span> {rawData.basic_info?.currency || 'N/A'}</p>
              </div>
            </div>

            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <DollarSign className="h-5 w-5 text-green-600" />
                <span className="text-sm font-medium text-green-900">Market Cap</span>
              </div>
              <p className="text-2xl font-bold text-green-700">
                {formatMarketCap(rawData.basic_info?.market_cap)}
              </p>
            </div>

            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="h-5 w-5 text-purple-600" />
                <span className="text-sm font-medium text-purple-900">Volatility (30D)</span>
              </div>
              <p className="text-2xl font-bold text-purple-700">
                {formatPercent(rawData.historical_data?.volatility_30d)}
              </p>
            </div>
          </div>

          {rawData.descriptions?.business_summary && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">Business Summary</h3>
              <p className="text-sm text-gray-700 leading-relaxed">{rawData.descriptions.business_summary}</p>
            </div>
          )}

          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-gray-600" />
              Key Financial Ratios
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-gray-500">Profit Margin</p>
                <p className="font-medium">{formatPercent(rawData.financials?.key_ratios?.profit_margin)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Revenue Growth</p>
                <p className="font-medium">{formatPercent(rawData.financials?.key_ratios?.revenue_growth)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Return on Equity</p>
                <p className="font-medium">{formatPercent(rawData.financials?.key_ratios?.return_on_equity)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Debt to Equity</p>
                <p className="font-medium">{formatNumber(rawData.financials?.key_ratios?.debt_to_equity)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Current Ratio</p>
                <p className="font-medium">{formatNumber(rawData.financials?.key_ratios?.current_ratio)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Operating Margin</p>
                <p className="font-medium">{formatPercent(rawData.financials?.key_ratios?.operating_margin)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Gross Margin</p>
                <p className="font-medium">{formatPercent(rawData.financials?.key_ratios?.gross_margin)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Earnings Growth</p>
                <p className="font-medium">{formatPercent(rawData.financials?.key_ratios?.earnings_growth)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Beta</p>
                <p className="font-medium">{formatNumber(rawData.financials?.key_ratios?.beta)}</p>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Valuation Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-gray-500">Trailing P/E</p>
                <p className="font-medium">{formatNumber(rawData.financials?.valuation_metrics?.trailing_pe)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Forward P/E</p>
                <p className="font-medium">{formatNumber(rawData.financials?.valuation_metrics?.forward_pe)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Price to Sales</p>
                <p className="font-medium">{formatNumber(rawData.financials?.valuation_metrics?.price_to_sales)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Price to Book</p>
                <p className="font-medium">{formatNumber(rawData.financials?.valuation_metrics?.price_to_book)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Enterprise Value</p>
                <p className="font-medium">{formatMarketCap(rawData.financials?.valuation_metrics?.enterprise_value)}</p>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Dividend Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-gray-500">Dividend Yield</p>
                <p className="font-medium">{formatPercent(rawData.financials?.dividend_info?.dividend_yield)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Payout Ratio</p>
                <p className="font-medium">{formatPercent(rawData.financials?.dividend_info?.payout_ratio)}</p>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Historical Performance</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
              <div>
                <p className="text-xs text-gray-500">30D Volatility</p>
                <p className="font-medium">{formatPercent(rawData.historical_data?.volatility_30d)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">90D Volatility</p>
                <p className="font-medium">{formatPercent(rawData.historical_data?.volatility_90d)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">1M Momentum</p>
                <p className="font-medium">{formatPercent(rawData.historical_data?.momentum_1m)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">3M Momentum</p>
                <p className="font-medium">{formatPercent(rawData.historical_data?.momentum_3m)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Avg Volume (30D)</p>
                <p className="font-medium">{formatNumber(rawData.historical_data?.volume_avg_30d)}</p>
              </div>
            </div>
          </div>

          {rawData.historical_data?.time_series && rawData.historical_data.time_series.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <Calendar className="h-5 w-5 mr-2 text-gray-600" />
                Historical Price Data (Recent 30 Days)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 border-b">
                    <tr>
                      <th className="px-4 py-2 text-left font-medium text-gray-700">Date</th>
                      <th className="px-4 py-2 text-right font-medium text-gray-700">Open</th>
                      <th className="px-4 py-2 text-right font-medium text-gray-700">High</th>
                      <th className="px-4 py-2 text-right font-medium text-gray-700">Low</th>
                      <th className="px-4 py-2 text-right font-medium text-gray-700">Close</th>
                      <th className="px-4 py-2 text-right font-medium text-gray-700">Volume</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {rawData.historical_data.time_series.slice(-30).reverse().map((data, idx) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-4 py-2">{formatDate(data.date)}</td>
                        <td className="px-4 py-2 text-right">${data.open.toFixed(2)}</td>
                        <td className="px-4 py-2 text-right">${data.high.toFixed(2)}</td>
                        <td className="px-4 py-2 text-right">${data.low.toFixed(2)}</td>
                        <td className="px-4 py-2 text-right font-medium">${data.close.toFixed(2)}</td>
                        <td className="px-4 py-2 text-right">{formatNumber(data.volume)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
