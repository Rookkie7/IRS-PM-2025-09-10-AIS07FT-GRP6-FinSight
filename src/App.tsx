import { useState } from 'react';
import { Header } from './components/Layout/Header';
import { TabNavigation } from './components/Layout/TabNavigation';
import { NewsTab } from './components/News/NewsTab';
import { RecommendationsTab } from './components/Recommendations/RecommendationsTab';
import { AnalystTab } from './components/Analyst/AnalystTab';
import { PredictTab } from './components/Predict/PredictTab';
import PairsTradingTab from './components/PairsTrading/PairsTradingTab';
import { useAuth } from './components/Auth/AuthContext.tsx';
import { LoginPage } from './components/Auth/LoginPage';
import { RegisterPage } from './components/Auth/RegisterPage';

type TabKey = 'news' | 'recommendations' | 'analyst' | 'predict' | 'pairstrading';

function App() {
    const [activeTab, setActiveTab] = useState<TabKey>('news');
    const [authView, setAuthView] = useState<'login' | 'register'>('login');

    // 关键：用全局 AuthContext 的 token/user 判断是否已登录
    const { token } = useAuth();

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
            case 'pairstrading':
                return <PairsTradingTab />;
            default:
                return <NewsTab />;
        }
    };

    // 未登录：显示登录/注册（不再传 onLoginSuccess/onRegisterSuccess）
    if (!token) {
        return authView === 'login' ? (
            <LoginPage onSwitchToRegister={() => setAuthView('register')} />
        ) : (
            <RegisterPage onSwitchToLogin={() => setAuthView('login')} />
        );
    }

    // 已登录：显示主应用
    return (
        <div className="min-h-screen bg-gray-50">
            <Header />
            <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
            <main className="min-h-screen">{renderActiveTab()}</main>
        </div>
    );
}

export default App;