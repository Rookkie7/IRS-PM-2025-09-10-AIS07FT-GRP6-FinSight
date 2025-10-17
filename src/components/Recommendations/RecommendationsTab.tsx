import React, { useState, useEffect } from 'react';
import { TrendingUp, Star, DollarSign, BarChart3, Heart, ThumbsDown, RefreshCw, Loader2, Search, List, Sparkles, ChevronLeft, ChevronRight } from 'lucide-react';
import { useAuth } from '../Auth/AuthContext';
import { StockDetailModal } from './StockDetailModal';

interface StockRecommendation {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  similarity: number;
  raw_similarity: number;
  updated_at: string;
}

interface StockListItem {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  updated_at: string;
}

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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

type ViewMode = 'list' | 'recommend';

export const RecommendationsTab: React.FC = () => {
  const { user } = useAuth();
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [allStocks, setAllStocks] = useState<StockListItem[]>([]);
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [customTopK, setCustomTopK] = useState('5');
  const [diversityFactor, setDiversityFactor] = useState(0.1);
  const [selectedStock, setSelectedStock] = useState<StockRawData | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 12;

  const fetchAllStocks = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/stocks/list`);

      if (!response.ok) {
        throw new Error(`Failed to fetch stock list: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.ok && data.stocks) {
        setAllStocks(data.stocks);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stock list');
      console.error('Error fetching stock list:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    if (!user?.id) {
      setError('Please login to view recommendations');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `${API_BASE_URL}/stocks/recommend?user_id=${user.id}&top_k=${topK}&diversity_factor=${diversityFactor}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch recommendations: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.ok && data.recommendations) {
        setRecommendations(data.recommendations);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch recommendations');
      console.error('Error fetching recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchStockDetails = async (symbol: string) => {
    try {
      setLoadingDetail(true);

      const response = await fetch(`${API_BASE_URL}/stocks/raw-data/${symbol}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch details for ${symbol}`);
      }

      const data = await response.json();
      if (data.ok && data.raw_data) {
        setSelectedStock(data.raw_data);
      }
    } catch (err) {
      console.error(`Error fetching details for ${symbol}:`, err);
      setError(err instanceof Error ? err.message : 'Failed to fetch stock details');
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleUserBehavior = async (
    symbol: string,
    behaviorType: 'click' | 'favorite' | 'dislike',
    sector: string
  ) => {
    if (!user?.id) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/users/behavior/update?user_id=${user.id}&behavior_type=${behaviorType}&stock_symbol=${symbol}&stock_sector=${sector}&invest_update=true`,
        { method: 'POST' }
      );

      if (response.ok) {
        console.log(`Behavior recorded: ${behaviorType} on ${symbol}`);
        if (behaviorType !== 'click' && viewMode === 'recommend') {
          fetchRecommendations();
        }
      }
    } catch (err) {
      console.error('Error recording behavior:', err);
    }
  };

  const handleStockClick = (symbol: string, sector: string) => {
    fetchStockDetails(symbol);
    if (viewMode === 'recommend') {
      handleUserBehavior(symbol, 'click', sector);
    }
  };

  const handleModeSwitch = (mode: ViewMode) => {
    setViewMode(mode);
    setError(null);
    setSearchQuery('');
    setCurrentPage(1);
    if (mode === 'recommend') {
      fetchRecommendations();
    } else {
      fetchAllStocks();
    }
  };

  const handleTopKChange = (value: string) => {
    setCustomTopK(value);
    const num = parseInt(value);
    if (!isNaN(num) && num > 0 && num <= 100) {
      setTopK(num);
    }
  };

  useEffect(() => {
    fetchAllStocks();
  }, []);

  useEffect(() => {
    if (viewMode === 'recommend' && user?.id) {
      fetchRecommendations();
    }
  }, [user?.id, topK, diversityFactor, viewMode]);

  const filteredStocks = (viewMode === 'list' ? allStocks : recommendations).filter((stock) => {
    const query = searchQuery.toLowerCase();
    return (
      stock.symbol.toLowerCase().includes(query) ||
      stock.name.toLowerCase().includes(query)
    );
  });

  const totalPages = Math.ceil(filteredStocks.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentStocks = filteredStocks.slice(startIndex, endIndex);

  const getSimilarityLevel = (similarity: number) => {
    if (similarity >= 0.9) return { label: 'Excellent Match', color: 'text-green-600 bg-green-50 border-green-200' };
    if (similarity >= 0.8) return { label: 'Great Match', color: 'text-blue-600 bg-blue-50 border-blue-200' };
    if (similarity >= 0.7) return { label: 'Good Match', color: 'text-yellow-600 bg-yellow-50 border-yellow-200' };
    return { label: 'Fair Match', color: 'text-gray-600 bg-gray-50 border-gray-200' };
  };

  const formatMarketCap = (marketCap?: number) => {
    if (!marketCap) return 'N/A';
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`;
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`;
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`;
    return `$${marketCap.toFixed(0)}`;
  };

  if (loading && !currentStocks.length) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <h2 className="text-2xl font-bold text-gray-900">
          {viewMode === 'list' ? 'All Stocks' : 'Personalized Recommendations'}
        </h2>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => handleModeSwitch('list')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                viewMode === 'list'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <List className="h-4 w-4" />
              <span>All Stocks</span>
            </button>
            <button
              onClick={() => handleModeSwitch('recommend')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${
                viewMode === 'recommend'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Sparkles className="h-4 w-4" />
              <span>Recommended</span>
            </button>
          </div>

          {viewMode === 'recommend' && (
            <>
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-600">Top K:</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={customTopK}
                  onChange={(e) => handleTopKChange(e.target.value)}
                  className="w-20 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div className="flex items-center space-x-2">
                <label className="text-sm text-gray-600">Diversity:</label>
                <select
                  value={diversityFactor}
                  onChange={(e) => setDiversityFactor(Number(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={0}>None</option>
                  <option value={0.1}>Low</option>
                  <option value={0.2}>Medium</option>
                  <option value={0.3}>High</option>
                </select>
              </div>
              <button
                onClick={fetchRecommendations}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Refresh</span>
              </button>
            </>
          )}
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search by symbol or name..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div className="text-sm text-gray-600">
          {filteredStocks.length} stocks found
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {viewMode === 'recommend' && recommendations.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-600 font-medium">Total Recommendations</p>
                <p className="text-2xl font-bold text-blue-700">{recommendations.length}</p>
              </div>
              <Star className="h-8 w-8 text-blue-600" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-600 font-medium">Avg Match Score</p>
                <p className="text-2xl font-bold text-green-700">
                  {recommendations.length > 0
                    ? Math.round((recommendations.reduce((sum, r) => sum + r.similarity, 0) / recommendations.length) * 100)
                    : 0}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-purple-600 font-medium">Unique Sectors</p>
                <p className="text-2xl font-bold text-purple-700">
                  {new Set(recommendations.map(r => r.sector)).size}
                </p>
              </div>
              <BarChart3 className="h-8 w-8 text-purple-600" />
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {currentStocks.map((stock) => {
          const isRecommendation = 'similarity' in stock;
          const matchLevel = isRecommendation ? getSimilarityLevel((stock as StockRecommendation).similarity) : null;

          return (
            <div
              key={stock.symbol}
              onClick={() => handleStockClick(stock.symbol, stock.sector)}
              className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-lg hover:border-blue-300 transition-all duration-200 cursor-pointer"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{stock.symbol}</h3>
                  <p className="text-sm text-gray-600 line-clamp-1">{stock.name}</p>
                </div>
                {matchLevel && (
                  <div className={`flex items-center space-x-1 px-2 py-1 rounded text-xs ${matchLevel.color}`}>
                    <Star className="h-3 w-3" />
                    <span>{Math.round((stock as StockRecommendation).similarity * 100)}%</span>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex items-center text-sm">
                  <span className="text-gray-500 w-20">Sector:</span>
                  <span className="font-medium text-gray-700 truncate">{stock.sector}</span>
                </div>
                <div className="flex items-center text-sm">
                  <span className="text-gray-500 w-20">Industry:</span>
                  <span className="text-gray-700 truncate">{stock.industry}</span>
                </div>
              </div>

              {viewMode === 'recommend' && (
                <div className="flex items-center justify-end space-x-2 mt-3 pt-3 border-t">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUserBehavior(stock.symbol, 'favorite', stock.sector);
                    }}
                    className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                    title="I like this"
                  >
                    <Heart className="h-4 w-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUserBehavior(stock.symbol, 'dislike', stock.sector);
                    }}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Not interested"
                  >
                    <ThumbsDown className="h-4 w-4" />
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {filteredStocks.length === 0 && !loading && (
        <div className="text-center py-12">
          <p className="text-gray-500">
            {searchQuery
              ? 'No stocks found matching your search.'
              : viewMode === 'recommend'
              ? 'No recommendations available. Please ensure your profile is set up.'
              : 'No stocks available.'}
          </p>
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
            <span>Previous</span>
          </button>
          <span className="text-gray-600">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <span>Next</span>
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}

      {loadingDetail && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
          <Loader2 className="h-12 w-12 animate-spin text-white" />
        </div>
      )}

      {selectedStock && (
        <StockDetailModal
          rawData={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      )}
    </div>
  );
};
