// src/api/http.ts（可选：统一封装 axios 实例）
import axios from "axios";
export const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  timeout: 15000,
});
