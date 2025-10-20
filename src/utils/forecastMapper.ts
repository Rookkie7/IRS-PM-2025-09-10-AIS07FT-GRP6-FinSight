// src/utils/forecastMapper.ts
import type { ForecastResult } from "../api/forecast";

export type OnePeriod = {
  period: string;
  predictedPrice: number;
  confidence: number;
  change: number;
  changePercent: number;
};

export type PredictionData = {
  symbol: string;
  companyName: string;
  currentPrice: number;
  predictions: OnePeriod[];
  risk: "Low" | "Medium" | "High";
  accuracy: number;
  lastUpdated: string;
};

function labelOf(days: number): string {
  if (days === 7) return "1 Week";
  if (days === 30) return "1 Month";
  if (days === 90) return "3 Months";
  if (days === 180) return "6 Months";
  return `${days} Days`;
}

export function toPredictionData(resp: ForecastResult): PredictionData {
  const curr = resp.current_price;
  const preds = (resp.predictions ?? [])
    .slice()
    .sort((a, b) => a.horizon_days - b.horizon_days)
    .map((p) => {
      const predicted = p.predicted ?? curr;
      const change = predicted - curr;
      const changePercent = curr ? (change / curr) * 100 : 0;
      const conf = typeof p.confidence === "number" ? Math.max(0, Math.min(1, p.confidence)) : 0.7;
      return {
        period: labelOf(p.horizon_days),
        predictedPrice: predicted,
        confidence: conf,
        change,
        changePercent,
      };
    });

  const avgConf = preds.length ? preds.reduce((s, x) => s + x.confidence, 0) / preds.length : 0.7;
  const risk: "Low" | "Medium" | "High" = avgConf >= 0.8 ? "Low" : avgConf < 0.6 ? "High" : "Medium";

  return {
    symbol: resp.ticker,
    companyName: resp.company_name || resp.ticker, // 👈 后端名优先，缺失兜底 ticker
    currentPrice: curr,
    predictions: preds,
    risk,
    accuracy: avgConf,
    lastUpdated: resp.generated_at,
  };
}
