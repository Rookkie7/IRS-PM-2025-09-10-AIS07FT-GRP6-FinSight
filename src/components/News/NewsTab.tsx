import React, { useState } from 'react';
import { Clock, ExternalLink, Filter, Search } from 'lucide-react';

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  timestamp: string;
  category: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  relevanceScore: number;
}

const mockNews: NewsItem[] = [
  {
    id: '1',
    title: 'Fed Signals Potential Rate Cut in Q2 2024',
    summary: 'Federal Reserve officials hint at monetary policy adjustments amid changing economic conditions...',
    source: 'Reuters',
    timestamp: '2 hours ago',
    category: 'Monetary Policy',
    sentiment: 'positive',
    relevanceScore: 0.95
  },
  {
    id: '2',
    title: 'Tech Stocks Rally on AI Infrastructure Investments',
    summary: 'Major technology companies announce significant investments in artificial intelligence infrastructure...',
    source: 'Bloomberg',
    timestamp: '4 hours ago',
    category: 'Technology',
    sentiment: 'positive',
    relevanceScore: 0.88
  },
  {
    id: '3',
    title: 'Energy Sector Faces Regulatory Headwinds',
    summary: 'New environmental regulations could impact traditional energy companies profitability...',
    source: 'Financial Times',
    timestamp: '6 hours ago',
    category: 'Energy',
    sentiment: 'negative',
    relevanceScore: 0.82
  }
];

export const NewsTab: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600 bg-green-50';
      case 'negative': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Market News</h2>
        <div className="flex items-center space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search news..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <button className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors">
            <Filter className="h-4 w-4" />
            <span>Filter</span>
          </button>
        </div>
      </div>

      {/* News Feed */}
      <div className="space-y-4">
        {mockNews.map((news) => (
          <div
            key={news.id}
            className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-3">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(news.sentiment)}`}>
                  {news.sentiment.charAt(0).toUpperCase() + news.sentiment.slice(1)}
                </span>
                <span className="text-sm text-gray-500">{news.category}</span>
                <div className="flex items-center text-gray-400">
                  <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                </div>
                <span className="text-sm text-gray-500">Relevance: {Math.round(news.relevanceScore * 100)}%</span>
              </div>
              <button className="text-gray-400 hover:text-gray-600">
                <ExternalLink className="h-4 w-4" />
              </button>
            </div>
            
            <h3 className="text-lg font-semibold text-gray-900 mb-2 hover:text-blue-600 cursor-pointer">
              {news.title}
            </h3>
            
            <p className="text-gray-600 mb-3 line-clamp-2">
              {news.summary}
            </p>
            
            <div className="flex items-center justify-between text-sm text-gray-500">
              <div className="flex items-center space-x-4">
                <span className="font-medium">{news.source}</span>
                <div className="flex items-center space-x-1">
                  <Clock className="h-3 w-3" />
                  <span>{news.timestamp}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};