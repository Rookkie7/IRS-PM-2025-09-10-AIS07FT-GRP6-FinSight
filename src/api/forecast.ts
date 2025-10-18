// src/api/forecast.ts（axios 版本）
import { http } from "./http";

export async function fetchForecast(
  ticker: string,
  horizons: number[] = [7, 30, 90, 180],
  method = "naive-drift"
) {
  const { data } = await http.get(`/forecast/${encodeURIComponent(ticker)}`, {
    params: { horizons: horizons.join(","), method },
  });
  return data;
}
