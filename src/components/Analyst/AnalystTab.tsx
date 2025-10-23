import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Send, Bot, User, TrendingUp, DollarSign, BarChart3, RefreshCw, Sparkles, Database } from 'lucide-react';

type Role = 'user' | 'ai';

interface ChatMessage {
    id: string;
    type: Role;
    content: string;
    timestamp: string;
    citations?: Array<Record<string, any>>;
}

const suggestedQuestions = [
    "Analyze NVDA's recent earnings impact",
    "Compare energy vs tech sector performance",
    "What are the top ESG investment opportunities?",
    "Explain current market volatility patterns"
];

const API_BASE = import.meta.env.VITE_BACKEND_BASE_URL || 'http://localhost:8000';

export const AnalystTab: React.FC = () => {
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            id: 'welcome',
            type: 'ai',
            content:
                "Hello! I'm your AI Financial Analyst. Ask me about sectors, stocks, earnings, or macro trends.",
            timestamp: new Date().toLocaleTimeString()
        }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    const userId = useMemo(() => undefined as string | undefined, []);

    const [datasets, setDatasets] = useState<Array<{ id: string; name?: string }>>([]);
    const [models, setModels] = useState<string[]>([]);
    const [selectedModel, setSelectedModel] = useState<string | null>(null);
    const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);

    useEffect(() => {
        (async () => {
            try {
                const [dsRes, mdlRes] = await Promise.all([
                    fetch(`${API_BASE}/rag/datasets`),
                    fetch(`${API_BASE}/rag/models`)
                ]);

                if (dsRes.ok) {
                    const j = await dsRes.json();
                    setDatasets(Array.isArray(j?.items) ? j.items : []);
                }
                if (mdlRes.ok) {
                    const j = await mdlRes.json();
                    setModels(Array.isArray(j?.items) ? j.items : []);
                    if (!selectedModel && Array.isArray(j?.items) && j.items.length > 0) {
                        setSelectedModel(j.items[0]);
                    }
                }
            } catch (e) {
                console.warn('load datasets/models failed', e);
            }
        })();
    }, []);

    useEffect(() => {
        const saved = localStorage.getItem('rag_session_id');
        if (saved) setSessionId(saved);
    }, []);

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }, [messages, isTyping]);

    const nowStr = () =>
        new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    const pushUser = (content: string) => {
        const m: ChatMessage = {
            id: `u_${Date.now()}`,
            type: 'user',
            content,
            timestamp: nowStr()
        };
        setMessages(prev => [...prev, m]);
    };

    const pushAI = (content: string, citations?: ChatMessage['citations']) => {
        const m: ChatMessage = {
            id: `a_${Date.now()}`,
            type: 'ai',
            content,
            timestamp: nowStr(),
            citations
        };
        setMessages(prev => [...prev, m]);
    };

    const handleSendMessage = async () => {
        const q = inputValue.trim();
        if (!q || isTyping) return;

        pushUser(q);
        setInputValue('');
        setIsTyping(true);

        try {
            const res = await fetch(`${API_BASE}/rag/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: q,
                    session_id: sessionId,
                    user_id: userId,
                    dataset_ids: selectedDatasetIds.length ? selectedDatasetIds : undefined
                })
            });

            if (!res.ok) {
                const detail = await safeDetail(res);
                throw new Error(detail || `HTTP ${res.status}`);
            }

            type ChatResp = {
                session_id: string;
                answer: string;
                citations?: Array<Record<string, any>>;
            };
            const data = (await res.json()) as ChatResp;

            if (data.session_id && data.session_id !== sessionId) {
                setSessionId(data.session_id);
                localStorage.setItem('rag_session_id', data.session_id);
            }

            const answer = data.answer?.trim() || '(No answer returned from RAG)';
            pushAI(answer, data.citations);
        } catch (err: any) {
            pushAI(`⚠️ Error: ${err?.message || 'RAG request failed'}`);
        } finally {
            setIsTyping(false);
        }
    };

    const handleSuggestedQuestion = (question: string) => {
        setInputValue(question);
    };

    const handleNewChat = () => {
        localStorage.removeItem('rag_session_id');
        setSessionId(null);
        setMessages([
            {
                id: 'welcome',
                type: 'ai',
                content:
                    "Started a new session. What would you like to explore? I can retrieve fresh market info and reason over it.",
                timestamp: nowStr()
            }
        ]);
    };

    return (
        <div className="min-h-screen flex flex-col">
            <div className="max-w-7xl mx-auto w-full flex-1 flex flex-col p-4 md:p-8 gap-6">
                {/* Header */}
                <div className="bg-white/80 backdrop-blur-sm border border-slate-200 rounded-2xl p-6 shadow-sm">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div className="flex items-center gap-4">
                            <div className="relative">
                                <div className="w-14 h-14 bg-gradient-to-br from-blue-500 via-blue-600 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                                    <Bot className="h-7 w-7 text-white" />
                                </div>
                                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 border-2 border-white rounded-full animate-pulse"></div>
                            </div>
                            <div>
                                <h1 className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight">
                                    AI Financial Analyst
                                </h1>
                                <p className="text-sm text-slate-600 mt-1 flex items-center gap-2">
                                    <Sparkles className="h-3.5 w-3.5 text-blue-500" />
                                    Powered by RAG •
                                    <span className="font-mono text-xs bg-slate-100 px-2 py-0.5 rounded">
                                        {sessionId ? `${sessionId.slice(0, 8)}...` : 'new session'}
                                    </span>
                                </p>
                            </div>
                        </div>

                        <button
                            onClick={handleNewChat}
                            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-all hover:scale-105 active:scale-95"
                        >
                            <RefreshCw className="h-4 w-4" />
                            <span>New Chat</span>
                        </button>
                    </div>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <QuickStat
                        icon={<TrendingUp className="h-5 w-5" />}
                        title="Market Sentiment"
                        value="Bullish"
                        sub="Multi-source data"
                        gradient="from-emerald-500 to-teal-500"
                    />
                    <QuickStat
                        icon={<DollarSign className="h-5 w-5" />}
                        title="S&P 500"
                        value="—"
                        sub="Live via backend"
                        gradient="from-blue-500 to-cyan-500"
                    />
                    <QuickStat
                        icon={<BarChart3 className="h-5 w-5" />}
                        title="VIX Index"
                        value="—"
                        sub="Volatility measure"
                        gradient="from-orange-500 to-red-500"
                    />
                </div>

                {/* Configuration */}
                <div className="bg-white/80 backdrop-blur-sm border border-slate-200 rounded-2xl p-6 shadow-sm">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Model Selection */}
                        <div>
                            <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-3">
                                <Sparkles className="h-4 w-4 text-blue-500" />
                                AI Model
                            </label>
                            <select
                                className="w-full px-4 py-2.5 bg-white border border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all text-sm"
                                value={selectedModel ?? ''}
                                onChange={(e) => {
                                    const next = e.target.value || null;
                                    setSelectedModel(next);
                                    localStorage.removeItem('rag_session_id');
                                    setSessionId(null);
                                }}
                            >
                                {models.length === 0 && <option value="">Loading models...</option>}
                                {models.map((m) => (
                                    <option key={m} value={m}>{m}</option>
                                ))}
                            </select>
                        </div>

                        {/* Knowledge Bases */}
                        <div>
                            <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-3">
                                <Database className="h-4 w-4 text-blue-500" />
                                Knowledge Bases
                            </label>
                            <div className="flex flex-wrap gap-2">
                                {datasets.length === 0 ? (
                                    <span className="text-sm text-slate-500">Loading datasets...</span>
                                ) : (
                                    datasets.map((ds) => {
                                        const checked = selectedDatasetIds.includes(ds.id);
                                        return (
                                            <label
                                                key={ds.id}
                                                className={`px-3 py-1.5 text-sm rounded-lg border-2 cursor-pointer select-none transition-all ${
                                                    checked
                                                        ? 'bg-blue-500 text-white border-blue-500 shadow-md shadow-blue-500/30'
                                                        : 'bg-white text-slate-700 border-slate-300 hover:border-blue-300 hover:bg-blue-50'
                                                }`}
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={checked}
                                                    onChange={() => {
                                                        setSelectedDatasetIds((prev) =>
                                                            checked
                                                                ? prev.filter((x) => x !== ds.id)
                                                                : [...prev, ds.id]
                                                        );
                                                        localStorage.removeItem('rag_session_id');
                                                        setSessionId(null);
                                                    }}
                                                    className="hidden"
                                                />
                                                {ds.name || ds.id}
                                            </label>
                                        );
                                    })
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chat Area */}
                <div className="flex-1 bg-white/80 backdrop-blur-sm border border-slate-200 rounded-2xl shadow-sm flex flex-col overflow-hidden">
                    <div ref={scrollRef} className="flex-1 p-6 space-y-6 overflow-y-auto">
                        {messages.map((m) => (
                            <Bubble key={m.id} msg={m} />
                        ))}

                        {isTyping && (
                            <div className="flex items-start gap-3 animate-in fade-in slide-in-from-bottom-4">
                                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 via-blue-600 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30 flex-shrink-0">
                                    <Bot className="h-5 w-5 text-white" />
                                </div>
                                <div className="bg-slate-100 px-5 py-3 rounded-2xl rounded-tl-sm">
                                    <div className="flex gap-1.5">
                                        <Dot />
                                        <Dot delay="0.2s" />
                                        <Dot delay="0.4s" />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Suggested Questions */}
                    {messages.length <= 2 && (
                        <div className="p-6 border-t border-slate-200 bg-slate-50/50">
                            <p className="text-sm font-medium text-slate-700 mb-3">Suggested questions:</p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                {suggestedQuestions.map((q, i) => (
                                    <button
                                        key={i}
                                        onClick={() => handleSuggestedQuestion(q)}
                                        className="text-left text-sm text-slate-700 hover:text-blue-600 bg-white hover:bg-blue-50 p-3 rounded-xl border border-slate-200 hover:border-blue-300 transition-all hover:shadow-md hover:scale-[1.02] active:scale-100"
                                    >
                                        {q}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Input Area */}
                    <div className="p-6 border-t border-slate-200 bg-white">
                        <div className="flex gap-3">
                            <input
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                                placeholder="Ask me anything about the markets..."
                                className="flex-1 px-5 py-3 border-2 border-slate-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all text-sm placeholder:text-slate-400"
                            />
                            <button
                                onClick={handleSendMessage}
                                disabled={!inputValue.trim() || isTyping}
                                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl hover:shadow-lg hover:shadow-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none transition-all hover:scale-105 active:scale-95 inline-flex items-center justify-center font-medium"
                            >
                                <Send className="h-4 w-4" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

function QuickStat({
                       icon,
                       title,
                       value,
                       sub,
                       gradient
                   }: {
    icon: React.ReactNode;
    title: string;
    value: string;
    sub: string;
    gradient: string;
}) {
    return (
        <div className="bg-white/80 backdrop-blur-sm border border-slate-200 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow">
            <div className={`inline-flex items-center justify-center w-10 h-10 bg-gradient-to-br ${gradient} rounded-xl shadow-lg mb-3 text-white`}>
                {icon}
            </div>
            <div>
                <p className="text-sm font-medium text-slate-600 mb-1">{title}</p>
                <p className="text-2xl font-bold text-slate-900 mb-0.5">{value}</p>
                <p className="text-xs text-slate-500">{sub}</p>
            </div>
        </div>
    );
}

function Bubble({ msg }: { msg: ChatMessage }) {
    const isUser = msg.type === 'user';
    return (
        <div className={`flex items-start gap-3 ${isUser ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
            {!isUser && (
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 via-blue-600 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30 flex-shrink-0">
                    <Bot className="h-5 w-5 text-white" />
                </div>
            )}

            <div className="max-w-lg lg:max-w-xl flex flex-col gap-1.5">
                <div
                    className={`px-5 py-3 rounded-2xl shadow-sm ${
                        isUser
                            ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-tr-sm'
                            : 'bg-slate-100 text-slate-900 rounded-tl-sm'
                    }`}
                >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    {msg.citations && msg.citations.length > 0 && (
                        <details className="mt-3 text-xs">
                            <summary className={`cursor-pointer font-medium ${isUser ? 'text-blue-100' : 'text-slate-600'}`}>
                                {msg.citations.length} citation{msg.citations.length > 1 ? 's' : ''}
                            </summary>
                            <div className={`mt-2 p-2 rounded-lg overflow-x-auto ${isUser ? 'bg-blue-600/30' : 'bg-white'}`}>
                                <pre className={`text-[10px] ${isUser ? 'text-blue-50' : 'text-slate-700'}`}>
                                    {JSON.stringify(msg.citations, null, 2)}
                                </pre>
                            </div>
                        </details>
                    )}
                </div>
                <p className={`text-xs ${isUser ? 'text-right text-slate-500' : 'text-left text-slate-500'} px-1`}>
                    {msg.timestamp}
                </p>
            </div>

            {isUser && (
                <div className="w-10 h-10 bg-gradient-to-br from-slate-600 to-slate-700 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0">
                    <User className="h-5 w-5 text-white" />
                </div>
            )}
        </div>
    );
}

function Dot({ delay = '0s' }: { delay?: string }) {
    return (
        <div
            className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
            style={{ animationDelay: delay }}
        />
    );
}

async function safeDetail(res: Response): Promise<string | null> {
    try {
        const j = await res.json();
        return j?.detail || j?.message || null;
    } catch {
        return null;
    }
}
