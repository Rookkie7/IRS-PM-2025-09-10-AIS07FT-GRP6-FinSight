import React from 'react';
import { Crown, Target, TrendingUp, Scale, Zap, Heart, ThumbsDown, Eye, DollarSign, RefreshCw, Loader2, BarChart3 } from 'lucide-react';

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

interface AdvancedRecommendationsProps {
  recommendations: AdvancedRecommendation[];
  stockDetails: { [key: string]: any };
  loadingStocks: Set<string>;
  userInteractions: { [symbol: string]: any };
  comparisonList: string[];
  topK: number;
  customTopK: string;
  riskProfile: 'conservative' | 'balanced' | 'aggressive';
  onStockClick: (symbol: string, sector: string) => void;
  onFetchStockDetails: (symbol: string) => void;
  onToggleComparison: (symbol: string) => void;
  onUserBehavior: (symbol: string, behaviorType: string, sector: string) => void;
  onTopKChange: (value: string) => void;
  onRiskProfileChange: (profile: 'conservative' | 'balanced' | 'aggressive') => void;
  onRefresh: () => void;
  formatMarketCap: (marketCap?: number) => string;
  getSectorColor: (sector: string) => string;
}

export const AdvancedRecommendations: React.FC<AdvancedRecommendationsProps> = ({
  recommendations,
  stockDetails,
  loadingStocks,
  userInteractions,
  comparisonList,
  topK,
  customTopK,
  riskProfile,
  onStockClick,
  onFetchStockDetails,
  onToggleComparison,
  onUserBehavior,
  onTopKChange,
  onRiskProfileChange,
  onRefresh,
  formatMarketCap,
  getSectorColor,
}) => {
  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 0.6) return 'text-blue-600 bg-blue-50 border-blue-200';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const getWeightColor = (weight: number) => {
    if (weight >= 0.4) return 'text-purple-600 bg-purple-50';
    if (weight >= 0.3) return 'text-blue-600 bg-blue-50';
    return 'text-gray-600 bg-gray-50';
  };

  const renderStockCard = (rec: AdvancedRecommendation) => {
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
        className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-xl hover:border-purple-300 transition-all duration-300 cursor-pointer group"
        onClick={() => onStockClick(rec.symbol, rec.sector)}
        onMouseEnter={() => onFetchStockDetails(rec.symbol)}
      >
        {/* Header Information */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-2">
              <h3 className="text-xl font-bold text-gray-900 truncate">{rec.symbol}</h3>
              <div className="flex items-center space-x-1 px-2 py-1 rounded-full text-xs border border-purple-200 bg-purple-50 text-purple-700">
                <Crown className="h-3 w-3" />
                <span>{Math.round(rec.final_score * 100)}%</span>
              </div>
            </div>
            <p className="text-sm text-gray-600 line-clamp-2 mb-2">{rec.name}</p>
            <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSectorColor(rec.sector)}`}>
              {rec.sector}
            </div>
          </div>
        </div>

        {/* Weight Configuration */}
        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart3 className="h-4 w-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Weight Configuration</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className={`flex items-center justify-between px-2 py-1 rounded ${getWeightColor(rec.weight_used.preference)}`}>
              <span>Preference</span>
              <span className="font-medium">{Math.round(rec.weight_used.preference * 100)}%</span>
            </div>
            <div className={`flex items-center justify-between px-2 py-1 rounded ${getWeightColor(rec.weight_used.risk_return)}`}>
              <span>Risk Return</span>
              <span className="font-medium">{Math.round(rec.weight_used.risk_return * 100)}%</span>
            </div>
            <div className={`flex items-center justify-between px-2 py-1 rounded ${getWeightColor(rec.weight_used.diversification)}`}>
              <span>Diversification</span>
              <span className="font-medium">{Math.round(rec.weight_used.diversification * 100)}%</span>
            </div>
            <div className={`flex items-center justify-between px-2 py-1 rounded ${getWeightColor(rec.weight_used.timing)}`}>
              <span>Market Timing</span>
              <span className="font-medium">{Math.round(rec.weight_used.timing * 100)}%</span>
            </div>
          </div>
        </div>

        {/* Score Details */}
        <div className="mb-4 p-3 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-100">
          <div className="grid grid-cols-2 gap-3">
            <div className={`flex items-center justify-between p-2 rounded border ${getScoreColor(rec.component_scores.preference)}`}>
              <div className="flex items-center space-x-1">
                <Target className="h-3 w-3" />
                <span className="text-xs">Preference</span>
              </div>
              <span className="text-sm font-bold">{Math.round(rec.component_scores.preference * 100)}%</span>
            </div>
            <div className={`flex items-center justify-between p-2 rounded border ${getScoreColor(rec.component_scores.risk_adjusted)}`}>
              <div className="flex items-center space-x-1">
                <TrendingUp className="h-3 w-3" />
                <span className="text-xs">Risk Adj</span>
              </div>
              <span className="text-sm font-bold">{Math.round(rec.component_scores.risk_adjusted * 100)}%</span>
            </div>
            <div className={`flex items-center justify-between p-2 rounded border ${getScoreColor(rec.component_scores.diversification)}`}>
              <div className="flex items-center space-x-1">
                <Scale className="h-3 w-3" />
                <span className="text-xs">Diversify</span>
              </div>
              <span className="text-sm font-bold">{Math.round(rec.component_scores.diversification * 100)}%</span>
            </div>
            <div className={`flex items-center justify-between p-2 rounded border ${getScoreColor(rec.component_scores.market_timing)}`}>
              <div className="flex items-center space-x-1">
                <Zap className="h-3 w-3" />
                <span className="text-xs">Timing</span>
              </div>
              <span className="text-sm font-bold">{Math.round(rec.component_scores.market_timing * 100)}%</span>
            </div>
          </div>
        </div>

        {/* Recommendation Explanation */}
        <div className="mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-sm font-medium text-gray-700">Recommendation Explanation</span>
          </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3 max-h-32 overflow-y-auto">
            <p className="text-xs text-gray-600 leading-relaxed">
              {rec.explanation}
            </p>
          </div>
        </div>

        {/* Stock Basic Information */}
        {details && (
          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
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
          </div>
        )}

        {isLoading && (
          <div className="flex justify-center">
            <Loader2 className="h-4 w-4 animate-spin text-purple-600" />
          </div>
        )}

        {/* Interaction Buttons */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-100">
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
      {/* Control Panel */}
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
          <label className="text-sm text-gray-600 font-medium">Risk:</label>
          <select
            value={riskProfile}
            onChange={(e) => onRiskProfileChange(e.target.value as any)}
            className="border-0 bg-transparent focus:ring-0 font-medium"
          >
            <option value="conservative">Conservative</option>
            <option value="balanced">Balanced</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>
        <button
          onClick={onRefresh}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg hover:shadow-xl"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Statistics */}
      {recommendations.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Total Recommendations</p>
                <p className="text-2xl font-bold">{recommendations.length}</p>
              </div>
              <Crown className="h-8 w-8 opacity-90" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Avg Final Score</p>
                <p className="text-2xl font-bold">
                  {recommendations.length > 0
                    ? Math.round((recommendations.reduce((sum, r) => sum + r.final_score, 0) / recommendations.length) * 100)
                    : 0}%
                </p>
              </div>
              <Target className="h-8 w-8 opacity-90" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Risk Profile</p>
                <p className="text-2xl font-bold capitalize">{riskProfile}</p>
              </div>
              <TrendingUp className="h-8 w-8 opacity-90" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-xl p-4 shadow-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Strategy</p>
                <p className="text-2xl font-bold">Multi-Obj</p>
              </div>
              <Scale className="h-8 w-8 opacity-90" />
            </div>
          </div>
        </div>
      )}

      {/* Recommendations List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {recommendations.map(renderStockCard)}
      </div>

      {recommendations.length === 0 && (
        <div className="text-center py-12">
          <div className="bg-gray-50 rounded-xl p-8 max-w-md mx-auto">
            <Crown className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">
              No advanced recommendations available
            </p>
            <p className="text-gray-400 text-sm mt-2">
              Please ensure your user profile is properly set up
            </p>
          </div>
        </div>
      )}
    </div>
  );
};