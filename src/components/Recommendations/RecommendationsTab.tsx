import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, Star, DollarSign, BarChart3, Heart, ThumbsDown, 
  RefreshCw, Loader2, Search, List, Sparkles, ChevronLeft, 
  ChevronRight, Eye, Volume2, Target, Zap, Shield, TrendingDown,
  Scale, X
} from 'lucide-react';
import { useAuth } from '../Auth/AuthContext';
import { StockDetailModal } from './StockDetailModal';
import { ComparisonModal } from './ComparisonModal';

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
  const [stockDetails, setStockDetails] = useState<{[key: string]: StockRawData}>({});
  const [loadingStocks, setLoadingStocks] = useState<Set<string>>(new Set());
  const [userInteractions, setUserInteractions] = useState<{
    [symbol: string]: {
      favorite: boolean;
      dislike: boolean;
      favoriteCount: number;
      dislikeCount: number;
      animating: 'favorite' | 'dislike' | null;
    }
  }>({});
  const [comparisonList, setComparisonList] = useState<string[]>([]);
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [comparisonData, setComparisonData] = useState<StockRawData[]>([]);
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
        // 预加载前几个股票的详细信息
        data.stocks.slice(0, 6).forEach((stock: StockListItem) => {
          fetchStockDetailsForCard(stock.symbol);
        });
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
        // 预加载推荐股票的详细信息
        data.recommendations.forEach((rec: StockRecommendation) => {
          fetchStockDetailsForCard(rec.symbol);
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

  const fetchStockDetailsForCard = async (symbol: string) => {
    if (stockDetails[symbol] || loadingStocks.has(symbol)) return;

    try {
      setLoadingStocks(prev => new Set(prev).add(symbol));
      
      const response = await fetch(`${API_BASE_URL}/stocks/raw-data/${symbol}`);
      
      if (response.ok) {
        const data = await response.json();
        if (data.ok && data.raw_data) {
          setStockDetails(prev => ({
            ...prev,
            [symbol]: data.raw_data
          }));
        }
      }
    } catch (err) {
      console.error(`Error fetching details for ${symbol}:`, err);
    } finally {
      setLoadingStocks(prev => {
        const newSet = new Set(prev);
        newSet.delete(symbol);
        return newSet;
      });
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
    behaviorType: 'click' | 'favorite' | 'dislike' | 'unfavorite' | 'undislike',
    sector: string
  ) => {
    if (!user?.id) return;

    try {
      // 设置动画状态
      setUserInteractions(prev => ({
        ...prev,
        [symbol]: {
          ...prev[symbol],
          animating: behaviorType === 'favorite' ? 'favorite' : 'dislike'
        }
      }));

      // 更新计数
      setUserInteractions(prev => {
        const current = prev[symbol] || { 
          favorite: false, 
          dislike: false, 
          favoriteCount: 0, 
          dislikeCount: 0,
          animating: null
        };
        
        if (behaviorType === 'favorite') {
          return {
            ...prev,
            [symbol]: {
              ...current,
              favoriteCount: current.favoriteCount + 1,
              favorite: true
            }
          };
        } else if (behaviorType === 'dislike') {
          return {
            ...prev,
            [symbol]: {
              ...current,
              dislikeCount: current.dislikeCount + 1,
              dislike: true
            }
          };
        }
        
        return prev;
      });

      // 发送行为数据到后端
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

      // 清除动画状态
      setTimeout(() => {
        setUserInteractions(prev => ({
          ...prev,
          [symbol]: {
            ...prev[symbol],
            animating: null
          }
        }));
      }, 600);

    } catch (err) {
      console.error('Error recording behavior:', err);
      // 出错时也清除动画状态
      setUserInteractions(prev => ({
        ...prev,
        [symbol]: {
          ...prev[symbol],
          animating: null
        }
      }));
    }
  };

  const handleStockClick = (symbol: string, sector: string) => {
    fetchStockDetails(symbol);
    if (viewMode === 'list') {
      handleUserBehavior(symbol, 'click', sector);
    }
  };

  const toggleComparison = (symbol: string) => {
    setComparisonList(prev => 
      prev.includes(symbol) 
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol].slice(0, 3) // 最多对比3只股票
    );
  };

  const analyzeComparison = async () => {
    if (comparisonList.length < 2) return;
    
    try {
      setLoadingDetail(true);
      
      // 获取所有对比股票的详细数据
      const comparisonPromises = comparisonList.map(symbol => 
        fetch(`${API_BASE_URL}/stocks/raw-data/${symbol}`).then(res => res.json())
      );
      
      const results = await Promise.all(comparisonPromises);
      const validData = results
        .filter(result => result.ok && result.raw_data)
        .map(result => result.raw_data);
      
      setComparisonData(validData);
      setShowComparisonModal(true);
    } catch (err) {
      console.error('Error fetching comparison data:', err);
      setError('Failed to load comparison data');
    } finally {
      setLoadingDetail(false);
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
    if (!isNaN(num) && num > 0 && num <= 20) {
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

  const getSectorColor = (sector: string) => {
    const colors: {[key: string]: string} = {
      'Technology': 'bg-blue-100 text-blue-800',
      'Healthcare': 'bg-green-100 text-green-800',
      'Financial Services': 'bg-purple-100 text-purple-800',
      'Consumer Cyclical': 'bg-yellow-100 text-yellow-800',
      'Consumer Defensive': 'bg-orange-100 text-orange-800',
      'Energy': 'bg-red-100 text-red-800',
      'Industrials': 'bg-indigo-100 text-indigo-800',
      'Real Estate': 'bg-pink-100 text-pink-800',
      'Utilities': 'bg-teal-100 text-teal-800',
      'Communication Services': 'bg-cyan-100 text-cyan-800',
      'Basic Materials': 'bg-amber-100 text-amber-800'
    };
    return colors[sector] || 'bg-gray-100 text-gray-800';
  };

  const renderStockCard = (stock: StockListItem | StockRecommendation) => {
    const isRecommendation = 'similarity' in stock;
    const matchLevel = isRecommendation ? getSimilarityLevel((stock as StockRecommendation).similarity) : null;
    const details = stockDetails[stock.symbol];
    const isLoading = loadingStocks.has(stock.symbol);
    const interactions = userInteractions[stock.symbol] || { 
      favorite: false, 
      dislike: false, 
      favoriteCount: 0, 
      dislikeCount: 0,
      animating: null 
    };
    const isInComparison = comparisonList.includes(stock.symbol);

    return (
      <div
        key={stock.symbol}
        className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-xl hover:border-blue-300 transition-all duration-300 cursor-pointer group"
        onClick={() => handleStockClick(stock.symbol, stock.sector)}
        onMouseEnter={() => fetchStockDetailsForCard(stock.symbol)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h3 className="text-xl font-bold text-gray-900 truncate">{stock.symbol}</h3>
              {isRecommendation && matchLevel && (
                <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs border ${matchLevel.color}`}>
                  <Star className="h-3 w-3" />
                  <span>{Math.round((stock as StockRecommendation).similarity * 100)}%</span>
                </div>
              )}
            </div>
            <p className="text-sm text-gray-600 line-clamp-2 mb-2">{stock.name}</p>
            <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSectorColor(stock.sector)}`}>
              {stock.sector}
            </div>
          </div>
        </div>

        {details && (
          <div className="space-y-3 mt-4">
            <div className="grid grid-cols-2 gap-3 text-sm">
              {details.basic_info?.market_cap && (
                <div className="flex items-center space-x-1">
                  <DollarSign className="h-4 w-4 text-green-600" />
                  <span className="font-medium">{formatMarketCap(details.basic_info.market_cap)}</span>
                </div>
              )}
              {details.financials?.valuation_metrics?.trailing_pe && (
                <div className="flex items-center space-x-1">
                  <Target className="h-4 w-4 text-blue-600" />
                  <span>P/E: {details.financials.valuation_metrics.trailing_pe.toFixed(1)}</span>
                </div>
              )}
              {details.financials?.dividend_info?.dividend_yield && details.financials.dividend_info.dividend_yield > 0 && (
                <div className="flex items-center space-x-1">
                  <TrendingUp className="h-4 w-4 text-yellow-600" />
                  <span>Div: {(details.financials.dividend_info.dividend_yield * 100).toFixed(2)}%</span>
                </div>
              )}
              {details.historical_data?.volatility_30d && (
                <div className="flex items-center space-x-1">
                  <Zap className="h-4 w-4 text-red-600" />
                  <span>Vol: {(details.historical_data.volatility_30d * 100).toFixed(1)}%</span>
                </div>
              )}
            </div>

            {details.historical_data?.momentum_1m && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">1M Momentum</span>
                <span className={`text-sm font-medium ${details.historical_data.momentum_1m > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {(details.historical_data.momentum_1m * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        )}

        {isLoading && (
          <div className="flex justify-center mt-4">
            <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
          </div>
        )}

        <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-100">
          <div className="flex items-center space-x-2 text-xs text-gray-500">
            <Eye className="h-3 w-3" />
            <span>Click for details</span>
          </div>
          <div className="flex items-center space-x-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleComparison(stock.symbol);
              }}
              className={`p-1.5 rounded-lg transition-all duration-300 ${
                isInComparison 
                  ? 'text-blue-600 bg-blue-50 scale-110' 
                  : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
              }`}
              title={isInComparison ? "Remove from comparison" : "Add to comparison"}
            >
              <Scale className={`h-4 w-4 transition-transform ${
                isInComparison ? 'scale-110' : 'group-hover:scale-105'
              }`} />
            </button>
            
            {/* 喜欢按钮 */}
            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleUserBehavior(stock.symbol, 'favorite', stock.sector);
                }}
                className={`p-1.5 rounded-lg transition-all duration-300 group/fav ${
                  interactions.favorite 
                    ? 'text-green-600 bg-green-50' 
                    : 'text-gray-400 hover:text-green-600 hover:bg-green-50'
                } ${interactions.animating === 'favorite' ? 'animate-pulse scale-125' : ''}`}
                title="Like this stock"
              >
                <Heart 
                  className={`h-4 w-4 transition-all ${
                    interactions.favorite ? 'fill-current' : 'group-hover/fav:scale-110'
                  }`} 
                />
              </button>
              {interactions.favoriteCount > 0 && (
                <div className="absolute -top-2 -right-2 bg-green-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center animate-bounce">
                  {interactions.favoriteCount}
                </div>
              )}
            </div>

            {/* 不喜欢按钮 */}
            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleUserBehavior(stock.symbol, 'dislike', stock.sector);
                }}
                className={`p-1.5 rounded-lg transition-all duration-300 group/dislike ${
                  interactions.dislike 
                    ? 'text-red-600 bg-red-50' 
                    : 'text-gray-400 hover:text-red-600 hover:bg-red-50'
                } ${interactions.animating === 'dislike' ? 'animate-pulse scale-125' : ''}`}
                title="Dislike this stock"
              >
                <ThumbsDown 
                  className={`h-4 w-4 transition-all ${
                    interactions.dislike ? 'fill-current' : 'group-hover/dislike:scale-110'
                  }`} 
                />
              </button>
              {interactions.dislikeCount > 0 && (
                <div className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center animate-bounce">
                  {interactions.dislikeCount}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // 骨架屏组件
  const SkeletonCard = () => (
    <div className="bg-white border border-gray-200 rounded-xl p-4 animate-pulse">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="h-6 bg-gray-200 rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-4 bg-gray-200 rounded"></div>
        <div className="h-4 bg-gray-200 rounded w-5/6"></div>
      </div>
    </div>
  );

  if (loading && !currentStocks.length) {
    return (
      <div className="p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            {viewMode === 'list' ? 'Stock Universe' : 'Personalized Recommendations'}
          </h2>
          <p className="text-gray-600 mt-1">
            {viewMode === 'list' 
              ? 'Browse all available stocks and discover investment opportunities' 
              : 'Stocks tailored to your investment preferences and behavior'
            }
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => handleModeSwitch('list')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${
                viewMode === 'list'
                  ? 'bg-white text-blue-600 shadow-sm ring-2 ring-blue-200'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <List className="h-4 w-4" />
              <span>All Stocks</span>
            </button>
            <button
              onClick={() => handleModeSwitch('recommend')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${
                viewMode === 'recommend'
                  ? 'bg-white text-blue-600 shadow-sm ring-2 ring-blue-200'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Sparkles className="h-4 w-4" />
              <span>Recommended</span>
            </button>
          </div>

          {viewMode === 'recommend' && (
            <>
              <div className="flex items-center space-x-2 bg-white border border-gray-300 rounded-lg px-3 py-2">
                <label className="text-sm text-gray-600 font-medium">Top K:</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={customTopK}
                  onChange={(e) => handleTopKChange(e.target.value)}
                  className="w-16 px-2 py-1 border-0 bg-transparent focus:ring-0 text-center font-medium"
                />
                <span className="text-xs text-gray-500">max 20</span>
              </div>
              <div className="flex items-center space-x-2 bg-white border border-gray-300 rounded-lg px-3 py-2">
                <label className="text-sm text-gray-600 font-medium">Diversity:</label>
                <select
                  value={diversityFactor}
                  onChange={(e) => setDiversityFactor(Number(e.target.value))}
                  className="border-0 bg-transparent focus:ring-0 font-medium"
                >
                  <option value={0}>None</option>
                  <option value={0.1}>Low</option>
                  <option value={0.2}>Medium</option>
                  <option value={0.3}>High</option>
                </select>
              </div>
              <button
                onClick={fetchRecommendations}
                className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Refresh</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* 对比面板 */}
      {comparisonList.length > 0 && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-4 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-bold text-blue-900 flex items-center">
              <Scale className="h-5 w-5 mr-2" />
              Stock Comparison ({comparisonList.length}/3)
            </h3>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-blue-600">
                {comparisonList.length} selected
              </span>
              <button
                onClick={() => setComparisonList([])}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium bg-white px-3 py-1 rounded-lg border border-blue-300 transition-colors"
              >
                Clear All
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            {comparisonList.map(symbol => {
              const stock = [...allStocks, ...recommendations].find(s => s.symbol === symbol);
              const details = stockDetails[symbol];
              return (
                <div key={symbol} className="bg-white px-4 py-3 rounded-lg border-2 border-blue-300 shadow-sm flex items-center justify-between min-w-[200px] group hover:border-blue-400 transition-all">
                  <div className="flex-1">
                    <div className="font-bold text-blue-900">{symbol}</div>
                    <div className="text-sm text-gray-600 truncate max-w-[150px]">
                      {stock?.name || 'Loading...'}
                    </div>
                    {details?.basic_info?.market_cap && (
                      <div className="text-xs text-green-600 font-medium mt-1">
                        {formatMarketCap(details.basic_info.market_cap)}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => toggleComparison(symbol)}
                    className="text-gray-400 hover:text-red-500 ml-2 transition-colors opacity-0 group-hover:opacity-100"
                    title="Remove from comparison"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
          {comparisonList.length >= 2 && (
            <div className="mt-3 flex justify-center">
              <button
                onClick={analyzeComparison}
                disabled={loadingDetail}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2 disabled:cursor-not-allowed"
              >
                {loadingDetail ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <BarChart3 className="h-4 w-4" />
                )}
                <span>
                  {loadingDetail ? 'Loading...' : 'Analyze Comparison'}
                </span>
              </button>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center space-x-3 bg-white border border-gray-300 rounded-xl p-3 shadow-sm">
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
            className="w-full pl-10 pr-4 py-2 border-0 bg-transparent focus:ring-0 text-lg"
          />
        </div>
        <div className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-lg font-medium">
          {filteredStocks.length} stocks
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-red-800 font-medium">{error}</p>
        </div>
      )}

      {viewMode === 'recommend' && recommendations.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Total Recommendations</p>
                <p className="text-2xl font-bold">{recommendations.length}</p>
              </div>
              <Star className="h-8 w-8 opacity-90" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Avg Match Score</p>
                <p className="text-2xl font-bold">
                  {recommendations.length > 0
                    ? Math.round((recommendations.reduce((sum, r) => sum + r.similarity, 0) / recommendations.length) * 100)
                    : 0}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 opacity-90" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Unique Sectors</p>
                <p className="text-2xl font-bold">
                  {new Set(recommendations.map(r => r.sector)).size}
                </p>
              </div>
              <BarChart3 className="h-8 w-8 opacity-90" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Avg Volatility</p>
                <p className="text-2xl font-bold">
                  {recommendations.length > 0
                    ? `${(recommendations.reduce((sum, r) => {
                        const details = stockDetails[r.symbol];
                        return sum + (details?.historical_data?.volatility_30d || 0);
                      }, 0) / recommendations.length * 100).toFixed(1)}%`
                    : '0%'}
                </p>
              </div>
              <Zap className="h-8 w-8 opacity-90" />
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {currentStocks.map(renderStockCard)}
      </div>

      {filteredStocks.length === 0 && !loading && (
        <div className="text-center py-12">
          <div className="bg-gray-50 rounded-xl p-8 max-w-md mx-auto">
            <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">
              {searchQuery
                ? 'No stocks found matching your search.'
                : viewMode === 'recommend'
                ? 'No recommendations available. Please ensure your profile is set up.'
                : 'No stocks available.'}
            </p>
          </div>
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

      {loadingDetail && !showComparisonModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
          <div className="bg-white rounded-xl p-6 flex items-center space-x-3">
            <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
            <span>Loading stock details...</span>
          </div>
        </div>
      )}

      {selectedStock && (
        <StockDetailModal
          rawData={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      )}

      {/* 对比分析模态框 */}
      {showComparisonModal && (
        <ComparisonModal
          stocks={comparisonData}
          onClose={() => {
            setShowComparisonModal(false);
            setComparisonData([]);
          }}
        />
      )}
    </div>
  );
};