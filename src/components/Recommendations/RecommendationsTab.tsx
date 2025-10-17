import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Star, DollarSign, BarChart3, Heart, ThumbsDown, RefreshCw, Loader2 } from 'lucide-react';
import { useAuth } from '../Auth/AuthContext';

interface StockRecommendation {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  similarity: number;
  raw_similarity: number;
  updated_at: string;
}

interface StockRawData {
  basic_info?: {
    market_cap?: number;
    currency?: string;
  };
  financials?: {
    valuation_metrics?: {
      trailing_pe?: number;
      forward_pe?: number;
    };
  };
  historical_data?: {
    time_series?: Array<{
      date: string;
      close: number;
    }>;
    volatility_30d?: number;
  };
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const RecommendationsTab: React.FC = () => {
  const { user } = useAuth();
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('similarity');
  const [topK, setTopK] = useState(5);
  const [diversityFactor, setDiversityFactor] = useState(0.1);
  const [stockDetails, setStockDetails] = useState<Record<string, StockRawData>>({});
  const [loadingDetails, setLoadingDetails] = useState<Record<string, boolean>>({});

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
        data.recommendations.forEach((rec: StockRecommendation) => {
          fetchStockDetails(rec.symbol);
        });
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
    if (stockDetails[symbol] || loadingDetails[symbol]) return;

    try {
      setLoadingDetails(prev => ({ ...prev, [symbol]: true }));

      const response = await fetch(`${API_BASE_URL}/stocks/raw-data/${symbol}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch details for ${symbol}`);
      }

      const data = await response.json();
      if (data.ok && data.raw_data) {
        setStockDetails(prev => ({ ...prev, [symbol]: data.raw_data }));
      }
    } catch (err) {
      console.error(`Error fetching details for ${symbol}:`, err);
    } finally {
      setLoadingDetails(prev => ({ ...prev, [symbol]: false }));
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
        if (behaviorType !== 'click') {
          fetchRecommendations();
        }
      }
    } catch (err) {
      console.error('Error recording behavior:', err);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, [user?.id, topK, diversityFactor]);

  const sortedRecommendations = [...recommendations].sort((a, b) => {
    if (sortBy === 'similarity') {
      return b.similarity - a.similarity;
    }
    return 0;
  });

  const formatMarketCap = (marketCap?: number) => {
    if (!marketCap) return 'N/A';
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`;
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`;
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`;
    return `$${marketCap.toFixed(0)}`;
  };

  const getCurrentPrice = (symbol: string) => {
    const details = stockDetails[symbol];
    if (!details?.historical_data?.time_series?.length) return null;
    const latestData = details.historical_data.time_series[details.historical_data.time_series.length - 1];
    return latestData.close;
  };

  const getSimilarityLevel = (similarity: number) => {
    if (similarity >= 0.9) return { label: 'Excellent Match', color: 'text-green-600 bg-green-50 border-green-200' };
    if (similarity >= 0.8) return { label: 'Great Match', color: 'text-blue-600 bg-blue-50 border-blue-200' };
    if (similarity >= 0.7) return { label: 'Good Match', color: 'text-yellow-600 bg-yellow-50 border-yellow-200' };
    return { label: 'Fair Match', color: 'text-gray-600 bg-gray-50 border-gray-200' };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Personalized Recommendations</h2>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Top:</label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={15}>15</option>
            </select>
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
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
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

      <div className="space-y-4">
        {sortedRecommendations.map((stock, index) => {
          const matchLevel = getSimilarityLevel(stock.similarity);
          const currentPrice = getCurrentPrice(stock.symbol);
          const details = stockDetails[stock.symbol];
          const isLoadingDetails = loadingDetails[stock.symbol];

          return (
            <div
              key={stock.symbol}
              className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all duration-200"
              onClick={() => handleUserBehavior(stock.symbol, 'click', stock.sector)}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                      <span className="text-lg font-bold text-white">{index + 1}</span>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{stock.symbol}</h3>
                    <p className="text-sm text-gray-600">{stock.name}</p>
                    <p className="text-xs text-gray-500">
                      {stock.sector} • {stock.industry}
                      {details?.basic_info?.market_cap && ` • ${formatMarketCap(details.basic_info.market_cap)}`}
                    </p>
                  </div>
                </div>

                <div className={`flex items-center space-x-2 px-3 py-1 rounded-lg border ${matchLevel.color}`}>
                  <Star className="h-4 w-4" />
                  <span className="font-medium text-sm">{matchLevel.label}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                {currentPrice && (
                  <div className="flex items-center space-x-2">
                    <DollarSign className="h-4 w-4 text-gray-400" />
                    <div>
                      <p className="text-xs text-gray-500">Current Price</p>
                      <p className="font-semibold">${currentPrice.toFixed(2)}</p>
                    </div>
                  </div>
                )}

                {details?.financials?.valuation_metrics?.trailing_pe && (
                  <div>
                    <p className="text-xs text-gray-500">P/E Ratio</p>
                    <p className="font-semibold">{details.financials.valuation_metrics.trailing_pe.toFixed(2)}</p>
                  </div>
                )}

                {details?.historical_data?.volatility_30d !== undefined && (
                  <div>
                    <p className="text-xs text-gray-500">30D Volatility</p>
                    <p className="font-semibold">{(details.historical_data.volatility_30d * 100).toFixed(2)}%</p>
                  </div>
                )}

                <div>
                  <p className="text-xs text-gray-500">Match Score</p>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${stock.similarity * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">{Math.round(stock.similarity * 100)}%</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between border-t pt-3">
                <p className="text-sm text-gray-600">
                  Based on your preferences and investment profile
                </p>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUserBehavior(stock.symbol, 'favorite', stock.sector);
                    }}
                    className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                    title="I like this"
                  >
                    <Heart className="h-5 w-5" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleUserBehavior(stock.symbol, 'dislike', stock.sector);
                    }}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Not interested"
                  >
                    <ThumbsDown className="h-5 w-5" />
                  </button>
                </div>
              </div>

              {isLoadingDetails && (
                <div className="flex items-center justify-center py-2">
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                  <span className="ml-2 text-sm text-gray-500">Loading details...</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {recommendations.length === 0 && !loading && (
        <div className="text-center py-12">
          <p className="text-gray-500">No recommendations available. Please ensure your profile is set up.</p>
        </div>
      )}
    </div>
  );
};
