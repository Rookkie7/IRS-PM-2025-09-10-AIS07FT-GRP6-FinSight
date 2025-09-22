import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Star, DollarSign, BarChart3 } from 'lucide-react';

interface StockRecommendation {
  id: string;
  symbol: string;
  companyName: string;
  currentPrice: number;
  targetPrice: number;
  recommendation: 'Buy' | 'Sell' | 'Hold';
  confidence: number;
  potentialReturn: number;
  analyst: string;
  reasoning: string;
  sector: string;
  marketCap: string;
}

const mockRecommendations: StockRecommendation[] = [
  {
    id: '1',
    symbol: 'AAPL',
    companyName: 'Apple Inc.',
    currentPrice: 195.12,
    targetPrice: 220.00,
    recommendation: 'Buy',
    confidence: 0.89,
    potentialReturn: 12.75,
    analyst: 'AI Model v2.1',
    reasoning: 'Strong fundamentals, growing services revenue, and AI integration potential.',
    sector: 'Technology',
    marketCap: '$3.01T'
  },
  {
    id: '2',
    symbol: 'TSLA',
    companyName: 'Tesla Inc.',
    currentPrice: 248.50,
    targetPrice: 280.00,
    recommendation: 'Buy',
    confidence: 0.76,
    potentialReturn: 12.67,
    analyst: 'AI Model v2.1',
    reasoning: 'EV market leadership, energy storage growth, and autonomous driving progress.',
    sector: 'Automotive',
    marketCap: '$791B'
  },
  {
    id: '3',
    symbol: 'META',
    companyName: 'Meta Platforms Inc.',
    currentPrice: 342.85,
    targetPrice: 320.00,
    recommendation: 'Hold',
    confidence: 0.65,
    potentialReturn: -6.67,
    analyst: 'AI Model v2.1',
    reasoning: 'Metaverse investments showing mixed results, but core advertising remains strong.',
    sector: 'Technology',
    marketCap: '$871B'
  }
];

export const RecommendationsTab: React.FC = () => {
  const [sortBy, setSortBy] = useState('confidence');

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'Buy': return 'text-green-600 bg-green-50 border-green-200';
      case 'Sell': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    }
  };

  const getRecommendationIcon = (rec: string) => {
    switch (rec) {
      case 'Buy': return <TrendingUp className="h-4 w-4" />;
      case 'Sell': return <TrendingDown className="h-4 w-4" />;
      default: return <BarChart3 className="h-4 w-4" />;
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Stock Recommendations</h2>
        <div className="flex items-center space-x-4">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="confidence">Sort by Confidence</option>
            <option value="return">Sort by Potential Return</option>
            <option value="price">Sort by Price</option>
          </select>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 font-medium">Buy Signals</p>
              <p className="text-2xl font-bold text-green-700">2</p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-600" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-yellow-50 to-yellow-100 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-yellow-600 font-medium">Hold Signals</p>
              <p className="text-2xl font-bold text-yellow-700">1</p>
            </div>
            <BarChart3 className="h-8 w-8 text-yellow-600" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 font-medium">Avg Confidence</p>
              <p className="text-2xl font-bold text-blue-700">77%</p>
            </div>
            <Star className="h-8 w-8 text-blue-600" />
          </div>
        </div>
      </div>

      {/* Recommendations List */}
      <div className="space-y-4">
        {mockRecommendations.map((stock) => (
          <div
            key={stock.id}
            className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all duration-200"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                    <span className="text-lg font-bold text-gray-700">{stock.symbol.charAt(0)}</span>
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{stock.symbol}</h3>
                  <p className="text-sm text-gray-600">{stock.companyName}</p>
                  <p className="text-xs text-gray-500">{stock.sector} • {stock.marketCap}</p>
                </div>
              </div>
              
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-lg border ${getRecommendationColor(stock.recommendation)}`}>
                {getRecommendationIcon(stock.recommendation)}
                <span className="font-medium text-sm">{stock.recommendation}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
              <div className="flex items-center space-x-2">
                <DollarSign className="h-4 w-4 text-gray-400" />
                <div>
                  <p className="text-xs text-gray-500">Current Price</p>
                  <p className="font-semibold">${stock.currentPrice.toFixed(2)}</p>
                </div>
              </div>
              
              <div>
                <p className="text-xs text-gray-500">Target Price</p>
                <p className="font-semibold">${stock.targetPrice.toFixed(2)}</p>
              </div>
              
              <div>
                <p className="text-xs text-gray-500">Potential Return</p>
                <p className={`font-semibold ${stock.potentialReturn > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {stock.potentialReturn > 0 ? '+' : ''}{stock.potentialReturn.toFixed(2)}%
                </p>
              </div>
              
              <div>
                <p className="text-xs text-gray-500">Confidence</p>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${stock.confidence * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium">{Math.round(stock.confidence * 100)}%</span>
                </div>
              </div>
            </div>

            <div className="border-t pt-3">
              <p className="text-sm text-gray-600 mb-2">{stock.reasoning}</p>
              <p className="text-xs text-gray-500">Analysis by {stock.analyst}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};