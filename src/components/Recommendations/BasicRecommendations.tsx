import React from 'react';
import { Sparkles, Star, TrendingUp, Scale, Zap, Heart, ThumbsDown, Eye, DollarSign, Target, RefreshCw, Loader2 } from 'lucide-react';

interface StockRecommendation {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  similarity: number;
  raw_similarity: number;
  updated_at: string;
}

interface BasicRecommendationsProps {
  recommendations: StockRecommendation[];
  stockDetails: { [key: string]: any };
  loadingStocks: Set<string>;
  userInteractions: { [symbol: string]: any };
  comparisonList: string[];
  topK: number;
  customTopK: string;
  diversityFactor: number;
  onStockClick: (symbol: string, sector: string) => void;
  onFetchStockDetails: (symbol: string) => void;
  onToggleComparison: (symbol: string) => void;
  onUserBehavior: (symbol: string, behaviorType: string, sector: string) => void;
  onTopKChange: (value: string) => void;
  onDiversityChange: (factor: number) => void;
  onRefresh: () => void;
  formatMarketCap: (marketCap?: number) => string;
  getSectorColor: (sector: string) => string;
  getSimilarityLevel: (similarity: number) => { label: string; color: string };
}

export const BasicRecommendations: React.FC<BasicRecommendationsProps> = ({
  recommendations,
  stockDetails,
  loadingStocks,
  userInteractions,
  comparisonList,
  topK,
  customTopK,
  diversityFactor,
  onStockClick,
  onFetchStockDetails,
  onToggleComparison,
  onUserBehavior,
  onTopKChange,
  onDiversityChange,
  onRefresh,
  formatMarketCap,
  getSectorColor,
  getSimilarityLevel,
}) => {
  const renderStockCard = (rec: StockRecommendation) => {
    const matchLevel = getSimilarityLevel(rec.similarity);
    const details = stockDetails[rec.symbol];
    const isLoading = loadingStocks.has(rec.symbol);
    const interactions = userInteractions[rec.symbol] || { 
      favorite: false, 
      dislike: false, 
      favoriteCount: 0, 
      dislikeCount: 0,
      animating: null 
    };
    const isInComparison = comparisonList.includes(rec.symbol);

    return (
      <div
        key={rec.symbol}
        className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-xl hover:border-blue-300 transition-all duration-300 cursor-pointer group"
        onClick={() => onStockClick(rec.symbol, rec.sector)}
        onMouseEnter={() => onFetchStockDetails(rec.symbol)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h3 className="text-xl font-bold text-gray-900 truncate">{rec.symbol}</h3>
              <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs border ${matchLevel.color}`}>
                <Star className="h-3 w-3" />
                <span>{Math.round(rec.similarity * 100)}%</span>
              </div>
            </div>
            <p className="text-sm text-gray-600 line-clamp-2 mb-2">{rec.name}</p>
            <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSectorColor(rec.sector)}`}>
              {rec.sector}
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
                onToggleComparison(rec.symbol);
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
            
            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onUserBehavior(rec.symbol, 'favorite', rec.sector);
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

            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onUserBehavior(rec.symbol, 'dislike', rec.sector);
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

  return (
    <div className="space-y-6">
      {/* 控制面板 */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center space-x-2 bg-white border border-gray-300 rounded-lg px-3 py-2">
          <label className="text-sm text-gray-600 font-medium">Top K:</label>
          <input
            type="number"
            min="1"
            max="20"
            value={customTopK}
            onChange={(e) => onTopKChange(e.target.value)}
            className="w-16 px-2 py-1 border-0 bg-transparent focus:ring-0 text-center font-medium"
          />
          <span className="text-xs text-gray-500">max 20</span>
        </div>
        <div className="flex items-center space-x-2 bg-white border border-gray-300 rounded-lg px-3 py-2">
          <label className="text-sm text-gray-600 font-medium">Diversity:</label>
          <select
            value={diversityFactor}
            onChange={(e) => onDiversityChange(Number(e.target.value))}
            className="border-0 bg-transparent focus:ring-0 font-medium"
          >
            <option value={0}>None</option>
            <option value={0.1}>Low</option>
            <option value={0.2}>Medium</option>
            <option value={0.3}>High</option>
          </select>
        </div>
        <button
          onClick={onRefresh}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* 统计信息 */}
      {recommendations.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Total Recommendations</p>
                <p className="text-2xl font-bold">{recommendations.length}</p>
              </div>
              <Sparkles className="h-8 w-8 opacity-90" />
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
              <Scale className="h-8 w-8 opacity-90" />
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

      {/* 推荐列表 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {recommendations.map(renderStockCard)}
      </div>

      {recommendations.length === 0 && (
        <div className="text-center py-12">
          <div className="bg-gray-50 rounded-xl p-8 max-w-md mx-auto">
            <Sparkles className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">
              暂无推荐结果
            </p>
            <p className="text-gray-400 text-sm mt-2">
              请确保您的用户画像已设置完成
            </p>
          </div>
        </div>
      )}
    </div>
  );
};