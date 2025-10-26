import React from 'react';
import { TrendingUp, Bell, User } from 'lucide-react';
import {useAuth} from "../Auth/AuthContext.tsx";

export const Header: React.FC = () => {
    const { user, logout } = useAuth();
    const displayName = user?.full_name || user?.username || 'Guest';
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-900">FinSight</h1>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="relative p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">
            <Bell className="h-5 w-5" />
            <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full"></span>
          </button>
          <button className="flex items-center space-x-2 p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors">
            <User className="h-5 w-5" />
            <span className="text-sm font-medium">{displayName}</span>
          </button>
           <button onClick={logout} className="text-sm px-3 py-1 rounded bg-gray-100 hover:bg-gray-200">
            Logout
          </button>
        </div>
      </div>
    </header>
  );
};