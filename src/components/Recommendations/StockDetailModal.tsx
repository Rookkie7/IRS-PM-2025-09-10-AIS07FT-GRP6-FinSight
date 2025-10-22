import React, { useState, useMemo } from 'react';
import { X, DollarSign, TrendingUp, BarChart3, Calendar, Building2, Volume2, Target, Zap, Shield } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

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
  const [activePriceType, setActivePriceType] = useState<'close' | 'open' | 'high' | 'low'>('close');
  const [hoveredLine, setHoveredLine] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1m' | '3m' | '6m' | '1y'>('1m');

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

  const chartData = useMemo(() => {
    if (!rawData.historical_data?.time_series) return [];
    
    // 根据选择的时间范围过滤数据
    let dataSlice;
    switch (timeRange) {
      case '1m':
        dataSlice = rawData.historical_data.time_series.slice(-30);
        break;
      case '3m':
        dataSlice = rawData.historical_data.time_series.slice(-90);
        break;
      case '6m':
        dataSlice = rawData.historical_data.time_series.slice(-180);
        break;
      case '1y':
        dataSlice = rawData.historical_data.time_series;
        break;
      default:
        dataSlice = rawData.historical_data.time_series.slice(-30);
    }
    
    return dataSlice.map(data => ({
      date: new Date(data.date).toLocaleDateString(),
      fullDate: data.date,
      open: data.open,
      high: data.high,
      low: data.low,
      close: data.close,
      volume: data.volume
    })).reverse();
  }, [rawData.historical_data?.time_series, timeRange]);

  // 计算价格范围用于Y轴
  const priceRange = useMemo(() => {
    if (!chartData.length) return { min: 0, max: 100 };
    
    const allPrices = chartData.flatMap(item => [item.open, item.high, item.low, item.close]);
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    
    // 添加一些边距
    const padding = (maxPrice - minPrice) * 0.1;
    return {
      min: Math.max(0, minPrice - padding),
      max: maxPrice + padding
    };
  }, [chartData]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: ${entry.value.toFixed(2)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const CustomYAxisTick = ({ x, y, payload }: any) => {
    return (
      <g transform={`translate(${x},${y})`}>
        <text x={0} y={0} dy={4} textAnchor="end" fill="#666" fontSize={12}>
          ${payload.value.toFixed(2)}
        </text>
      </g>
    );
  };

  const priceTypes = [
    { key: 'close' as const, name: 'Close', color: '#3B82F6' },
    { key: 'open' as const, name: 'Open', color: '#10B981' },
    { key: 'high' as const, name: 'High', color: '#EF4444' },
    { key: 'low' as const, name: 'Low', color: '#8B5CF6' }
  ];

  const timeRangeOptions = [
    { value: '1m', label: '1 Month' },
    { value: '3m', label: '3 Months' },
    { value: '6m', label: '6 Months' },
    { value: '1y', label: '1 Year' }
  ];

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
          {/* Header Stats */}
          <div className="grid grid-cols-1 md:grid-rows-2 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl p-4 md:row-span-2">
              <div className="flex items-center space-x-2 mb-2">
                <Building2 className="h-5 w-5" />
                <span className="text-sm font-medium">Company Info</span>
              </div>
              <div className="space-y-2">
                <p className="text-sm"><span className="font-medium">Sector:</span> {rawData.basic_info?.sector || 'N/A'}</p>
                <p className="text-sm"><span className="font-medium">Industry:</span> {rawData.basic_info?.industry || 'N/A'}</p>
                <p className="text-sm"><span className="font-medium">Country:</span> {rawData.basic_info?.country || 'N/A'}</p>
                <p className="text-sm"><span className="font-medium">Currency:</span> {rawData.basic_info?.currency || 'N/A'}</p>
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <DollarSign className="h-5 w-5" />
                <span className="text-sm font-medium">Market Cap</span>
              </div>
              <p className="text-2xl font-bold">
                {formatMarketCap(rawData.basic_info?.market_cap)}
              </p>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="h-5 w-5" />
                <span className="text-sm font-medium">30D Volatility</span>
              </div>
              <p className="text-2xl font-bold">
                {formatPercent(rawData.historical_data?.volatility_30d)}
              </p>
            </div>

            <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="h-5 w-5" />
                <span className="text-sm font-medium">P/E Ratio</span>
              </div>
              <p className="text-2xl font-bold">
                {formatNumber(rawData.financials?.valuation_metrics?.trailing_pe)}
              </p>
            </div>

            <div className="bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Zap className="h-5 w-5" />
                <span className="text-sm font-medium">Beta</span>
              </div>
              <p className="text-2xl font-bold">
                {formatNumber(rawData.financials?.key_ratios?.beta)}
              </p>
            </div>
          </div>

          {/* Price Chart */}
          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6 gap-4">
              <h3 className="text-xl font-bold text-gray-900">Price History</h3>
              <div className="flex flex-wrap gap-3">
                {/* 时间范围选择器 */}
                <div className="flex bg-gray-100 rounded-lg p-1">
                  {timeRangeOptions.map(({ value, label }) => (
                    <button
                      key={value}
                      onClick={() => setTimeRange(value as any)}
                      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                        timeRange === value
                          ? 'bg-white text-blue-600 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                
                {/* 价格类型选择器 */}
                <div className="flex space-x-2">
                  {priceTypes.map(({ key, name, color }) => (
                    <button
                      key={key}
                      onClick={() => setActivePriceType(key)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                        activePriceType === key
                          ? 'text-white'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                      style={{
                        backgroundColor: activePriceType === key ? color : 'transparent',
                        border: activePriceType === key ? 'none' : `1px solid ${color}`
                      }}
                      onMouseEnter={() => setHoveredLine(key)}
                      onMouseLeave={() => setHoveredLine(null)}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickMargin={10}
                  />
                  <YAxis 
                    tick={<CustomYAxisTick />}
                    tickMargin={10}
                    domain={[priceRange.min, priceRange.max]}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend 
                    onMouseEnter={(e) => setHoveredLine(e.dataKey)}
                    onMouseLeave={() => setHoveredLine(null)}
                  />
                  {priceTypes.map(({ key, name, color }) => (
                    <Area
                      key={key}
                      type="monotone"
                      dataKey={key}
                      name={name}
                      stroke={color}
                      fill={`${color}20`}
                      strokeWidth={hoveredLine === key || !hoveredLine ? 3 : 1}
                      fillOpacity={0.6}
                      activeDot={{ r: 6, stroke: color, strokeWidth: 2 }}
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Financial Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Valuation Metrics */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <DollarSign className="h-5 w-5 mr-2 text-blue-600" />
                Valuation Metrics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Market Cap</p>
                  <p className="font-semibold text-gray-900">
                    {formatMarketCap(rawData.basic_info?.market_cap)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">P/E Ratio</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.valuation_metrics?.trailing_pe)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Forward P/E</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.valuation_metrics?.forward_pe)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Price/Sales</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.valuation_metrics?.price_to_sales)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Price/Book</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.valuation_metrics?.price_to_book)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Enterprise Value</p>
                  <p className="font-semibold text-gray-900">
                    {formatMarketCap(rawData.financials?.valuation_metrics?.enterprise_value)}
                  </p>
                </div>
              </div>
            </div>

            {/* Financial Ratios */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="h-5 w-5 mr-2 text-green-600" />
                Financial Ratios
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Profit Margin</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.financials?.key_ratios?.profit_margin)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Revenue Growth</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.financials?.key_ratios?.revenue_growth)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">ROE</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.financials?.key_ratios?.return_on_equity)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Debt/Equity</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.key_ratios?.debt_to_equity)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Current Ratio</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.key_ratios?.current_ratio)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Beta</p>
                  <p className="font-semibold text-gray-900">
                    {formatNumber(rawData.financials?.key_ratios?.beta)}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Dividend & Performance */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Dividend Info */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-yellow-600" />
                Dividend Information
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Dividend Yield</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.financials?.dividend_info?.dividend_yield)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">Payout Ratio</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.financials?.dividend_info?.payout_ratio)}
                  </p>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-gray-50 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2 text-red-600" />
                Performance Metrics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">1M Momentum</p>
                  <p className={`font-semibold ${
                    (rawData.historical_data?.momentum_1m || 0) > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercent(rawData.historical_data?.momentum_1m)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">3M Momentum</p>
                  <p className={`font-semibold ${
                    (rawData.historical_data?.momentum_3m || 0) > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercent(rawData.historical_data?.momentum_3m)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">30D Volatility</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.historical_data?.volatility_30d)}
                  </p>
                </div>
                <div className="bg-white rounded-lg p-3 border border-gray-200">
                  <p className="text-sm text-gray-600">90D Volatility</p>
                  <p className="font-semibold text-gray-900">
                    {formatPercent(rawData.historical_data?.volatility_90d)}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Business Summary */}
          {rawData.descriptions?.business_summary && (
            <div className="bg-white border border-gray-200 rounded-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Business Summary</h3>
              <p className="text-gray-700 leading-relaxed">
                {rawData.descriptions.business_summary}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};