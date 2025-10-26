import axios from "axios";
// import { createClient } from '@supabase/supabase-js';

export const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  timeout: 15000,
});
// 统一 fetch 包装（带基础地址与错误处理）
const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
// const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
export async function fetchJson<T>(
    path: string,
    init?: RequestInit & { token?: string }
): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (init?.token) headers.Authorization = `Bearer ${init.token}`;

    const res = await fetch(`${BASE}${path}`, { ...init, headers });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    return res.json() as Promise<T>;
}

// export const supabase = createClient(supabaseUrl, supabaseAnonKey);

