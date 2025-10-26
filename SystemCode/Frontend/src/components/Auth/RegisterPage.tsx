import React, { useState } from 'react';
import { UserPlus } from 'lucide-react';
import { useAuth} from "./AuthContext.tsx";

const SECTOR_LIST = [
    'Utilities', 'Technology', 'Consumer Defensive', 'Healthcare',
    'Basic Materials', 'Real Estate', 'Energy', 'Industrials',
    'Consumer Cyclical', 'Communication Services', 'Financial Services'
];

const INTEREST_OPTIONS = [
    'AI', 'Cloud', 'Semiconductor', 'Fintech', 'E-commerce',
    'Blockchain', 'Green Energy', 'Biotechnology', 'Cybersecurity',
    'IoT', '5G', 'Automotive', 'Space', 'Robotics'
];

const MOCK_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    'JPM', 'V', 'WMT', 'JNJ', 'PG', 'MA', 'HD', 'BAC'
];

const INVESTMENT_PREFERENCES = [
    { key: 'market_cap', label: 'Market Cap Preference (0=Small-cap, 1=Large-cap)' },
    { key: 'growth_value', label: 'Growth-Value Preference (0=Value, 1=Growth)' },
    { key: 'dividend', label: 'Dividend Preference (0=Low Priority, 1=High Priority)' },
    { key: 'risk_tolerance', label: 'Risk Tolerance (0=Conservative, 1=Aggressive)' },
    { key: 'liquidity', label: 'Liquidity Requirement (0=Low, 1=High)' },
    { key: 'quality', label: 'Quality Preference (0=Low Priority, 1=High Priority)' },
    { key: 'valuation_safety', label: 'Valuation Safety (0=Accept High Valuation, 1=Require Safety Margin)' },
    { key: 'momentum', label: 'Momentum Preference (0=No Momentum Chasing, 1=Believe in Momentum)' },
    { key: 'efficiency', label: 'Efficiency Preference (0=Not Concerned, 1=High Priority)' },
];

interface RegisterPageProps {
    onSwitchToLogin: () => void;
    // onRegisterSuccess: (token: string, user: any) => void;
}

export function RegisterPage({ onSwitchToLogin }: RegisterPageProps) {
    const {setToken, setUser} = useAuth();
    const [formData, setFormData] = useState({
        email: '',
        username: '',
        password: '',
        full_name: '',
        bio: '',
        interests: [] as string[],
        sectors: [] as string[],
        tickers: [] as string[],
        investment_preference: {
            market_cap: 0.4,
            growth_value: 0.4,
            dividend: 0.4,
            risk_tolerance: 0.4,
            liquidity: 0.4,
            quality: 0.4,
            valuation_safety: 0.4,
            momentum: 0.4,
            efficiency: 0.4,
        },
    });

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const toggleArrayItem = (array: string[], item: string) => {
        if (array.includes(item)) {
            return array.filter(i => i !== item);
        }
        return [...array, item];
    };

    const handleInterestToggle = (interest: string) => {
        setFormData({
            ...formData,
            interests: toggleArrayItem(formData.interests, interest),
        });
    };

    const handleSectorToggle = (sector: string) => {
        setFormData({
            ...formData,
            sectors: toggleArrayItem(formData.sectors, sector),
        });
    };

    const handleTickerToggle = (ticker: string) => {
        setFormData({
            ...formData,
            tickers: toggleArrayItem(formData.tickers, ticker),
        });
    };

    const handlePreferenceChange = (key: string, value: number) => {
        setFormData({
            ...formData,
            investment_preference: {
                ...formData.investment_preference,
                [key]: value,
            },
        });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const apiUrl = `${import.meta.env.VITE_BACKEND_BASE_URL}/auth/register`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ payload: formData }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Registration failed');
            }

            setToken(data.token);
            setUser(data.user);
            // onRegisterSuccess(data.token, data.user);
        } catch (err: any) {
            setError(err.message || 'An error occurred during registration');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center p-4">
            <div className="w-full max-w-2xl bg-white rounded-lg shadow-lg p-8">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-full mb-4">
                        <UserPlus className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900">Create Account</h1>
                    <p className="text-gray-600 mt-2">Join our investment community</p>
                </div>

                {error && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Email *
                            </label>
                            <input
                                type="email"
                                name="email"
                                value={formData.email}
                                onChange={handleInputChange}
                                required
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                placeholder="your@email.com"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Username *
                            </label>
                            <input
                                type="text"
                                name="username"
                                value={formData.username}
                                onChange={handleInputChange}
                                required
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                placeholder="username"
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Full Name *
                            </label>
                            <input
                                type="text"
                                name="full_name"
                                value={formData.full_name}
                                onChange={handleInputChange}
                                required
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                placeholder="John Doe"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Password *
                            </label>
                            <input
                                type="password"
                                name="password"
                                value={formData.password}
                                onChange={handleInputChange}
                                required
                                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                placeholder="••••••••"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Bio (optional)
                        </label>
                        <textarea
                            name="bio"
                            value={formData.bio}
                            onChange={handleInputChange}
                            rows={3}
                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            placeholder="Tell us about your investment interests..."
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-3">
                            Interests
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {INTEREST_OPTIONS.map(interest => (
                                <button
                                    key={interest}
                                    type="button"
                                    onClick={() => handleInterestToggle(interest)}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                        formData.interests.includes(interest)
                                            ? 'bg-blue-600 text-white'
                                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                >
                                    {interest}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-3">
                            Sectors
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {SECTOR_LIST.map(sector => (
                                <button
                                    key={sector}
                                    type="button"
                                    onClick={() => handleSectorToggle(sector)}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                        formData.sectors.includes(sector)
                                            ? 'bg-green-600 text-white'
                                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                >
                                    {sector}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-3">
                            Tickers
                        </label>
                        <div className="flex flex-wrap gap-2">
                            {MOCK_TICKERS.map(ticker => (
                                <button
                                    key={ticker}
                                    type="button"
                                    onClick={() => handleTickerToggle(ticker)}
                                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                        formData.tickers.includes(ticker)
                                            ? 'bg-purple-600 text-white'
                                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                    }`}
                                >
                                    {ticker}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-4">
                            Investment Preferences
                        </label>
                        <div className="space-y-6">
                            {INVESTMENT_PREFERENCES.map(({ key, label }) => (
                                <div key={key}>
                                    <div className="flex justify-between items-center mb-2">
                                        <label className="text-sm text-gray-600">{label}</label>
                                        <span className="text-sm font-medium text-blue-600">
                                            {formData.investment_preference[key as keyof typeof formData.investment_preference].toFixed(1)}
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.2"
                                        value={formData.investment_preference[key as keyof typeof formData.investment_preference]}
                                        onChange={(e) => handlePreferenceChange(key, parseFloat(e.target.value))}
                                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                                    />
                                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                                        <span>0.0</span>
                                        <span>0.2</span>
                                        <span>0.4</span>
                                        <span>0.6</span>
                                        <span>0.8</span>
                                        <span>1.0</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Creating Account...' : 'Register'}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <p className="text-gray-600">
                        Already have an account?{' '}
                        <button
                            onClick={onSwitchToLogin}
                            className="text-blue-600 hover:text-blue-700 font-medium"
                        >
                            Sign in
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
}
