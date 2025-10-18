// // src/utils/forecastMapper.ts
// import { BackendForecastResult } from "../api/forecast";

// const PERIOD_LABEL: Record<number, string> = {
//   7: "1 Week",
//   30: "1 Month",
//   90: "3 Months",
//   180: "6 Months",
// };

// export function toPredictionData(
//   b: BackendForecastResult,
//   companyName = ""
// ) {
//   const current = b.current_price;
//   const preds = b.predictions
//     .sort((a, b) => a.horizon_days - b.horizon_days)
//     .map(p => {
//       const change = +(p.predicted - current).toFixed(2);
//       const changePercent = +(((p.predicted - current) / current) * 100).toFixed(2);
//       return {
//         period: PERIOD_LABEL[p.horizon_days] || `${p.horizon_days} Days`,
//         predictedPrice: +p.predicted.toFixed(2),
//         confidence: p.confidence ?? 0.5,
//         change,
//         changePercent
//       };
//     });

//   // 这些可先用简单规则占位
//   const risk = Math.max(...preds.map(p => Math.abs(p.changePercent))) > 15 ? "High"
//              : Math.max(...preds.map(p => Math.abs(p.changePercent))) > 7  ? "Medium"
//              : "Low";
//   const accuracy = 0.75; // 先写死/或从后端 method 给出基准

//   return {
//     symbol: b.ticker,
//     companyName,
//     currentPrice: +current.toFixed(2),
//     predictions: preds,
//     risk,
//     accuracy
//   };
// }


// src/utils/forecastMapper.ts
export interface BackendForecastPoint {
  horizon_days: number;
  predicted: number;
  confidence?: number;
}
export interface BackendForecastResult {
  ticker: string;
  method: string;
  generated_at: string;
  current_price: number;
  predictions: BackendForecastPoint[];
}

// 你的 UI 结构（和 PredictTab.tsx 中一致）
export interface PredictionRow {
  period: string;
  predictedPrice: number;
  confidence: number;
  change: number;
  changePercent: number;
}
export interface PredictionData {
  symbol: string;
  companyName: string;
  currentPrice: number;
  predictions: PredictionRow[];
  risk: 'Low' | 'Medium' | 'High';
  accuracy: number;
}

const PERIOD_LABEL: Record<number, string> = {
  7: '1 Week',
  30: '1 Month',
  90: '3 Months',
  180: '6 Months',
};

export function toPredictionData(
  backend: BackendForecastResult,
  companyName = ''
): PredictionData {
  const current = backend.current_price;

  const rows: PredictionRow[] = backend.predictions
    .slice() // 避免原地排序
    .sort((a, b) => a.horizon_days - b.horizon_days)
    .map(p => {
      const change = +(p.predicted - current).toFixed(2);
      const changePercent = +(((p.predicted - current) / current) * 100).toFixed(2);
      return {
        period: PERIOD_LABEL[p.horizon_days] || `${p.horizon_days} Days`,
        predictedPrice: +p.predicted.toFixed(2),
        confidence: p.confidence ?? 0.5,
        change,
        changePercent,
      };
    });

  const maxAbsPct = rows.reduce((m, r) => Math.max(m, Math.abs(r.changePercent)), 0);
  const risk: PredictionData['risk'] =
    maxAbsPct > 15 ? 'High' : maxAbsPct > 7 ? 'Medium' : 'Low';

  return {
    symbol: backend.ticker,
    companyName,
    currentPrice: +current.toFixed(2),
    predictions: rows,
    risk,
    accuracy: 0.75, // 占位，可按模型评估替换
  };
}
