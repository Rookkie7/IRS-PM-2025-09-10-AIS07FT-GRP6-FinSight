import { useState, useEffect } from 'react';
import { Header } from './components/Layout/Header';
import { TabNavigation } from './components/Layout/TabNavigation';
import { NewsTab } from './components/News/NewsTab';
import { RecommendationsTab } from './components/Recommendations/RecommendationsTab';
import { AnalystTab } from './components/Analyst/AnalystTab';
import { PredictTab } from './components/Predict/PredictTab';
import { LoginPage } from './components/Auth/LoginPage';
import { RegisterPage } from './components/Auth/RegisterPage';

function App() {
    const [activeTab, setActiveTab] = useState('news');
    const [authView, setAuthView] = useState<'login' | 'register'>('login');
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [, setUser] = useState<unknown>(null);

    useEffect(() => {
        const token = localStorage.getItem('auth_token');
        const userData = localStorage.getItem('user_data');
        if (token && userData) {
            setIsAuthenticated(true);
            setUser(JSON.parse(userData));
        }
    }, []);

    const handleLoginSuccess = (token: string, userData: any) => {
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_data', JSON.stringify(userData));
        setIsAuthenticated(true);
        setUser(userData);
    };

    const handleRegisterSuccess = (token: string, userData: any) => {
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_data', JSON.stringify(userData));
        setIsAuthenticated(true);
        setUser(userData);
    };

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

    if (!isAuthenticated) {
        if (authView === 'login') {
            return (
                <LoginPage
                    onSwitchToRegister={() => setAuthView('register')}
                    onLoginSuccess={handleLoginSuccess}
                />
            );
        }
        return (
            <RegisterPage
                onSwitchToLogin={() => setAuthView('login')}
                onRegisterSuccess={handleRegisterSuccess}
            />
        );
    }

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