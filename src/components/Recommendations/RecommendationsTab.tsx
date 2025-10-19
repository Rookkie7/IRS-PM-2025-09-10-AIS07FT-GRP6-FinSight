import React, { useState, useEffect } from 'react';
import { 
  RefreshCw, Loader2, Search, List, Sparkles, ChevronLeft, 
  ChevronRight, Scale, X, Crown 
} from 'lucide-react';
import { useAuth } from '../Auth/AuthContext';
import { StockDetailModal } from './StockDetailModal';
import { ComparisonModal } from './ComparisonModal';
import { AdvancedRecommendations } from './AdvancedRecommendations';
import { BasicRecommendations } from './BasicRecommendations';
import { StockList } from './StockList';

interface StockRecommendation {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  similarity: number;
  raw_similarity: number;
  updated_at: string;
}

interface AdvancedRecommendation {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  final_score: number;
  component_scores: {
    preference: number;
    risk_adjusted: number;
    diversification: number;
    market_timing: number;
  };
  explanation: string;
  weight_used: {
    preference: number;
    risk_return: number;
    diversification: number;
    timing: number;
  };
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

type ViewMode = 'list' | 'recommend' | 'advanced';

export const RecommendationsTab: React.FC = () => {
  const { user } = useAuth();
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [allStocks, setAllStocks] = useState<StockListItem[]>([]);
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [advancedRecommendations, setAdvancedRecommendations] = useState<AdvancedRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [customTopK, setCustomTopK] = useState('5');
  const [diversityFactor, setDiversityFactor] = useState(0.1);
  const [riskProfile, setRiskProfile] = useState<'conservative' | 'balanced' | 'aggressive'>('balanced');
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

  // API 调用函数
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

  const fetchAdvancedRecommendations = async () => {
    if (!user?.id) {
      setError('Please login to view advanced recommendations');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `${API_BASE_URL}/stocks/recommend/v2?user_id=${user.id}&top_k=${topK}&risk_profile=${riskProfile}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch advanced recommendations: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.ok && data.recommendations) {
        setAdvancedRecommendations(data.recommendations);
        // 预加载推荐股票的详细信息
        data.recommendations.forEach((rec: AdvancedRecommendation) => {
          fetchStockDetailsForCard(rec.symbol);
        });
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch advanced recommendations');
      console.error('Error fetching advanced recommendations:', err);
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
        } else if (behaviorType !== 'click' && viewMode === 'advanced') {
          fetchAdvancedRecommendations();
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
    } else if (mode === 'advanced') {
      fetchAdvancedRecommendations();
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

  const handleDiversityChange = (factor: number) => {
    setDiversityFactor(factor);
  };

  const handleRiskProfileChange = (profile: 'conservative' | 'balanced' | 'aggressive') => {
    setRiskProfile(profile);
  };

  useEffect(() => {
    fetchAllStocks();
  }, []);

  useEffect(() => {
    if (viewMode === 'recommend' && user?.id) {
      fetchRecommendations();
    } else if (viewMode === 'advanced' && user?.id) {
      fetchAdvancedRecommendations();
    }
  }, [user?.id, topK, diversityFactor, riskProfile, viewMode]);

  // 工具函数
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

  const getSimilarityLevel = (similarity: number) => {
    if (similarity >= 0.9) return { label: 'Excellent Match', color: 'text-green-600 bg-green-50 border-green-200' };
    if (similarity >= 0.8) return { label: 'Great Match', color: 'text-blue-600 bg-blue-50 border-blue-200' };
    if (similarity >= 0.7) return { label: 'Good Match', color: 'text-yellow-600 bg-yellow-50 border-yellow-200' };
    return { label: 'Fair Match', color: 'text-gray-600 bg-gray-50 border-gray-200' };
  };

  // 过滤和分页逻辑
  const filteredStocks = (() => {
    if (viewMode === 'list') return allStocks;
    if (viewMode === 'recommend') return recommendations;
    if (viewMode === 'advanced') return advancedRecommendations.map(rec => ({
      symbol: rec.symbol,
      name: rec.name,
      sector: rec.sector,
      industry: rec.industry,
      updated_at: new Date().toISOString()
    }));
    return [];
  })().filter((stock) => {
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
      {/* 头部控制面板 */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-gray-900 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            {viewMode === 'list' ? 'Stock Universe' : 
             viewMode === 'recommend' ? 'Personalized Recommendations' : 
             'Advanced Recommendations'}
          </h2>
          <p className="text-gray-600 mt-1">
            {viewMode === 'list' 
              ? 'Browse all available stocks and discover investment opportunities' 
              : viewMode === 'recommend'
              ? 'Stocks tailored to your investment preferences and behavior'
              : 'Multi-objective optimized recommendations with risk adjustment'
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
            <button
              onClick={() => handleModeSwitch('advanced')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${
                viewMode === 'advanced'
                  ? 'bg-white text-purple-600 shadow-sm ring-2 ring-purple-200'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Crown className="h-4 w-4" />
              <span>Advanced</span>
            </button>
          </div>
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
              const stock = [...allStocks, ...recommendations, ...advancedRecommendations].find(s => s.symbol === symbol);
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
                  <Scale className="h-4 w-4" />
                )}
                <span>
                  {loadingDetail ? 'Loading...' : 'Analyze Comparison'}
                </span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* 搜索栏 */}
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

      {/* 错误显示 */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-red-800 font-medium">{error}</p>
        </div>
      )}

      {/* 内容区域 */}
      {viewMode === 'advanced' && (
        <AdvancedRecommendations
          recommendations={advancedRecommendations}
          stockDetails={stockDetails}
          loadingStocks={loadingStocks}
          userInteractions={userInteractions}
          comparisonList={comparisonList}
          topK={topK}
          customTopK={customTopK}
          riskProfile={riskProfile}
          onStockClick={handleStockClick}
          onFetchStockDetails={fetchStockDetailsForCard}
          onToggleComparison={toggleComparison}
          onUserBehavior={handleUserBehavior}
          onTopKChange={handleTopKChange}
          onRiskProfileChange={handleRiskProfileChange}
          onRefresh={fetchAdvancedRecommendations}
          formatMarketCap={formatMarketCap}
          getSectorColor={getSectorColor}
        />
      )}

      {viewMode === 'recommend' && (
        <BasicRecommendations
          recommendations={recommendations}
          stockDetails={stockDetails}
          loadingStocks={loadingStocks}
          userInteractions={userInteractions}
          comparisonList={comparisonList}
          topK={topK}
          customTopK={customTopK}
          diversityFactor={diversityFactor}
          onStockClick={handleStockClick}
          onFetchStockDetails={fetchStockDetailsForCard}
          onToggleComparison={toggleComparison}
          onUserBehavior={handleUserBehavior}
          onTopKChange={handleTopKChange}
          onDiversityChange={handleDiversityChange}
          onRefresh={fetchRecommendations}
          formatMarketCap={formatMarketCap}
          getSectorColor={getSectorColor}
          getSimilarityLevel={getSimilarityLevel}
        />
      )}

      {viewMode === 'list' && (
        <StockList
          stocks={currentStocks}
          stockDetails={stockDetails}
          loadingStocks={loadingStocks}
          userInteractions={userInteractions}
          comparisonList={comparisonList}
          onStockClick={handleStockClick}
          onFetchStockDetails={fetchStockDetailsForCard}
          onToggleComparison={toggleComparison}
          onUserBehavior={handleUserBehavior}
          formatMarketCap={formatMarketCap}
          getSectorColor={getSectorColor}
        />
      )}

      {/* 分页 */}
      {viewMode === 'list' && totalPages > 1 && (
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

      {/* 空状态 */}
      {filteredStocks.length === 0 && !loading && (
        <div className="text-center py-12">
          <div className="bg-gray-50 rounded-xl p-8 max-w-md mx-auto">
            <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">
              {searchQuery
                ? 'No stocks found matching your search.'
                : viewMode === 'recommend'
                ? 'No recommendations available. Please ensure your profile is set up.'
                : viewMode === 'advanced'
                ? 'No advanced recommendations available. Please ensure your profile is set up.'
                : 'No stocks available.'}
            </p>
          </div>
        </div>
      )}

      {/* 加载状态 */}
      {loadingDetail && !showComparisonModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
          <div className="bg-white rounded-xl p-6 flex items-center space-x-3">
            <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
            <span>Loading stock details...</span>
          </div>
        </div>
      )}

      {/* 模态框 */}
      {selectedStock && (
        <StockDetailModal
          rawData={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      )}

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