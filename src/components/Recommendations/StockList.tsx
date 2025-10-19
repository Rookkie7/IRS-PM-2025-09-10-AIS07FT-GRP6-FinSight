import React from 'react';
import { Heart, ThumbsDown, Eye, DollarSign, Target, TrendingUp, Zap, Scale, Loader2 } from 'lucide-react';

interface StockListItem {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  updated_at: string;
}

interface StockListProps {
  stocks: StockListItem[];
  stockDetails: { [key: string]: any };
  loadingStocks: Set<string>;
  userInteractions: { [symbol: string]: any };
  comparisonList: string[];
  onStockClick: (symbol: string, sector: string) => void;
  onFetchStockDetails: (symbol: string) => void;
  onToggleComparison: (symbol: string) => void;
  onUserBehavior: (symbol: string, behaviorType: string, sector: string) => void;
  formatMarketCap: (marketCap?: number) => string;
  getSectorColor: (sector: string) => string;
}

export const StockList: React.FC<StockListProps> = ({
  stocks,
  stockDetails,
  loadingStocks,
  userInteractions,
  comparisonList,
  onStockClick,
  onFetchStockDetails,
  onToggleComparison,
  onUserBehavior,
  formatMarketCap,
  getSectorColor,
}) => {
  const renderStockCard = (stock: StockListItem) => {
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
        onClick={() => onStockClick(stock.symbol, stock.sector)}
        onMouseEnter={() => onFetchStockDetails(stock.symbol)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h3 className="text-xl font-bold text-gray-900 truncate">{stock.symbol}</h3>
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
                onToggleComparison(stock.symbol);
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
                  onUserBehavior(stock.symbol, 'favorite', stock.sector);
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
                  onUserBehavior(stock.symbol, 'dislike', stock.sector);
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
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {stocks.map(renderStockCard)}
    </div>
  );
};