import React, { useState } from 'react';
import { Header } from './components/Layout/Header';
import { TabNavigation } from './components/Layout/TabNavigation';
import { NewsTab } from './components/News/NewsTab';
import { RecommendationsTab } from './components/Recommendations/RecommendationsTab';
import { AnalystTab } from './components/Analyst/AnalystTab';
import { PredictTab } from './components/Predict/PredictTab';

function App() {
  const [activeTab, setActiveTab] = useState('news');

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'news':
        return <NewsTab />;
      case 'recommendations':
        return <RecommendationsTab />;
      case 'analyst':
        return <AnalystTab />;
      case 'predict':
        return <PredictTab />;
      default:
        return <NewsTab />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="min-h-screen">
        {renderActiveTab()}
      </main>
    </div>
  );
}

export default App;