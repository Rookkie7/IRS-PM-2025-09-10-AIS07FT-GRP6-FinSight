import React, { createContext, useContext, useEffect, useState } from 'react';

export interface User {
    id?: string;
    email: string;
    username: string;
    full_name?: string;
    bio?: string;
    interests?: string[];
    sectors?: string[];
    tickers?: string[];
}

interface AuthContextType {
    user: User | null;
    token: string | null;
    setUser: (u: User | null) => void;
    setToken: (t: string | null) => void;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
    user: null,
    token: null,
    setUser: () => {},
    setToken: () => {},
    logout: () => {},
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);

    // 刷新时恢复会话
    useEffect(() => {
        const t = localStorage.getItem('token');
        const u = localStorage.getItem('user');
        if (t) setToken(t);
        if (u) setUser(JSON.parse(u));
    }, []);

    const logout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        setToken(null);
        setUser(null);
    };

    // 也可以在这里加一个 effect，把 token 改变时自动持久化
    useEffect(() => {
        if (token) localStorage.setItem('token', token);
        else localStorage.removeItem('token');
    }, [token]);

    useEffect(() => {
        if (user) localStorage.setItem('user', JSON.stringify(user));
        else localStorage.removeItem('user');
    }, [user]);

    return (
        <AuthContext.Provider value={{ user, token, setUser, setToken, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);