import React from 'react';
import { Newspaper, Target, Brain, TrendingUp, GitCompare } from 'lucide-react';

interface Tab {
  id: string;
  label: string;
  icon: React.ReactNode;
}

interface TabNavigationProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

const tabs: Tab[] = [
  { id: 'news', label: 'News', icon: <Newspaper className="h-5 w-5" /> },
  { id: 'recommendations', label: 'Stock Recommend', icon: <Target className="h-5 w-5" /> },
  { id: 'analyst', label: 'AI Analyst', icon: <Brain className="h-5 w-5" /> },
  { id: 'predict', label: 'Stock Predict', icon: <TrendingUp className="h-5 w-5" /> },
  { id: 'pairstrading', label: 'Pairs Trading', icon: <GitCompare className="h-5 w-5" /> },
];

export const TabNavigation: React.FC<TabNavigationProps> = ({ activeTab, onTabChange }) => {
  return (
    <nav className="bg-white border-b border-gray-200 px-6">
      <div className="flex space-x-8">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === tab.id
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
};