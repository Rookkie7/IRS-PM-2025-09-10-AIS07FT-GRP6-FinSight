import React, { useState } from 'react';
import { LogIn } from 'lucide-react';
import { useAuth } from './AuthContext'; // ← 路径按你的项目实际调整
// 如需路由跳转可解开：
// import { useNavigate } from 'react-router-dom';

interface LoginPageProps {
    onSwitchToRegister: () => void;
}

export function LoginPage({ onSwitchToRegister }: LoginPageProps) {
    const { setToken, setUser } = useAuth();
    // const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (loading) return;
        setError('');
        setLoading(true);

        try {
            const base = import.meta.env.VITE_BACKEND_BASE_URL as string | undefined;
            if (!base) throw new Error('VITE_BACKEND_BASE_URL is not set in .env(.local)');

            // 1) 登录：拿 access_token
            const loginUrl = new URL('/auth/login', base).toString();
            const loginResp = await fetch(loginUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // 如果后端需要 {username, password} 就改字段名
                body: JSON.stringify({ email, password }),
            });

            const loginText = await loginResp.text();
            let loginData: any = {};
            try { loginData = loginText ? JSON.parse(loginText) : {}; } catch { /* ignore */ }

            if (!loginResp.ok) {
                const msg = loginData?.error || loginData?.message || `Login failed (HTTP ${loginResp.status})`;
                throw new Error(msg);
            }

            const token: string | null =
                loginData?.access_token || loginData?.token || loginData?.jwt || null;

            if (!token) throw new Error('No access token in response');

            // 先保存 token（让后续请求能带鉴权）
            setToken(token);

            // 2) 用 token 拉取当前用户
            const meUrl = new URL('/users/me', base).toString();
            const meResp = await fetch(meUrl, {
                headers: { Authorization: `Bearer ${token}` }, // token_type=bearer
            });

            if (!meResp.ok) {
                // 如果 /users/me 失败，仍允许登录，但给出提示
                console.warn('Fetching /users/me failed:', meResp.status);
            }

            let meData: any = null;
            try { meData = await meResp.json(); } catch { /* ignore */ }
            console.log(meData);
            // 3) 写入用户（没有就给占位，防止依赖 user 的地方炸）
            setUser(meData ?? { id: 'me', username: email.split('@')[0] || 'User' });

            // 如需路由跳转：
            // navigate('/app', { replace: true });
            // 如果用条件渲染，App.tsx 里看到 token/user 更新就会切到主界面
        } catch (err: any) {
            setError(err?.message || 'An error occurred during login');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center p-4">
            <div className="w-full max-w-md bg-white rounded-lg shadow-lg p-8">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-full mb-4">
                        <LogIn className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900">Welcome Back</h1>
                    <p className="text-gray-600 mt-2">Sign in to your account</p>
                </div>

                {error && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            placeholder="your@email.com"
                            autoComplete="email"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            placeholder="••••••••"
                            autoComplete="current-password"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Signing in...' : 'Sign In'}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <p className="text-gray-600">
                        Don't have an account?{' '}
                        <button
                            onClick={onSwitchToRegister}
                            className="text-blue-600 hover:text-blue-700 font-medium"
                        >
                            Create one
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
}