import React from 'react'
import { AuthProvider } from './components/Auth/AuthContext.tsx' // 路径按你的项目调整
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
      <AuthProvider>
          <App />
      </AuthProvider>
  </React.StrictMode>
);
