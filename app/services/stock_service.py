import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from pymongo.database import Database
import logging
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime

from app.model.models import StockVector, StockRawData

logger = logging.getLogger(__name__)

# 获取标普500中的前100市值symbol列表
sp500_top100 = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'ORCL', 'JPM', 'WMT', 'LLY',
                'V', 'NFLX', 'MA', 'XOM', 'JNJ', 'PLTR', 'COST', 'ABBV', 'HD', 'BAC', 'PG', 'AMD', 'UNH', 'GE', 'CVX',
                'KO', 'CSCO', 'IBM', 'TMUS', 'PM', 'WFC', 'MS', 'GS', 'ABT', 'CAT', 'CRM', 'AXP', 'MRK', 'LIN', 'MCD',
                'RTX', 'PEP', 'MU', 'TMO', 'DIS', 'UBER', 'ANET', 'APP', 'BX', 'T', 'NOW', 'INTU', 'BLK', 'INTC', 'C',
                'NEE', 'VZ', 'BKNG', 'AMAT', 'SCHW', 'QCOM', 'LRCX', 'GEV', 'BA', 'TJX', 'AMGN', 'TXN', 'ISRG', 'ACN',
                'APH', 'SPGI', 'GILD', 'DHR', 'ETN', 'BSX', 'ADBE', 'PANW', 'PFE', 'PGR', 'SYK', 'UNP', 'LOW', 'COF',
                'KLAC', 'HON', 'CRWD', 'HOOD', 'MDT', 'DE', 'LMT', 'IBKR', 'ADP', 'CEG', 'DASH', 'CB', 'MO', 'WELL']

SECTOR_LIST =['Utilities', 'Technology', 'Consumer Defensive', 'Healthcare', 'Basic Materials', 'Real Estate', 'Energy',
              'Industrials', 'Consumer Cyclical', 'Communication Services', 'Financial Services']


class StockService:
    def __init__(self, postgres_db: Session, mongo_db: Database):
        self.postgres_db = postgres_db
        self.mongo_db = mongo_db
        self.stock_raw_collection = mongo_db["stocks"]
        self.sector_list = SECTOR_LIST

    async def fetch_stock_data(self, symbols: List[str]) -> Dict[str, StockRawData]:
        """从yfinance获取股票数据并存入MongoDB"""
        stock_data_dict = {}

        for symbol in symbols:
            try:
                logger.info(f"获取股票数据: {symbol}")
                ticker = yf.Ticker(symbol)

                stock_data = StockRawData(symbol)

                # 1. 获取基本信息
                stock_data.basic_info = self._fetch_basic_info(ticker)

                # 2. 获取财务数据
                stock_data.financials = self._fetch_financial_ratios(ticker)

                # 3. 获取历史数据
                stock_data.historical_data = self._fetch_historical_data(ticker)

                # 4. 获取新闻和描述
                stock_data.descriptions = self._fetch_descriptions(ticker)

                stock_data_dict[symbol] = stock_data

                # 存入MongoDB
                await self._save_to_mongodb(stock_data)

                logger.info(f"成功获取 {symbol} 的数据")

            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {str(e)}")
                continue

        return stock_data_dict

    def _fetch_basic_info(self, ticker) -> Dict[str, Any]:
        """获取股票基本信息"""
        try:
            info = ticker.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'country': info.get('country', 'US'),
                'currency': info.get('currency', 'USD'),
            }
        except Exception as e:
            logger.error(f"获取基本信息失败: {e}")
            return {}

    def _fetch_financial_ratios(self, ticker) -> Dict[str, Any]:
        """获取财务比率"""
        try:
            info = ticker.info

            key_ratios = {
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'gross_margin': info.get('grossMargins', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'beta': info.get('beta', 1.0)
            }

            valuation_metrics = {
                'market_cap': info.get('marketCap', 0),
                'trailing_pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'price_to_book': info.get('priceToBook', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
            }

            dividend_info = {
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0)
            }

            return {
                'key_ratios': key_ratios,
                'valuation_metrics': valuation_metrics,
                'dividend_info': dividend_info
            }
        except Exception as e:
            logger.error(f"获取财务比率失败: {e}")
            return {}

    def _fetch_historical_data(self, ticker) -> Dict[str, Any]:
        """获取历史价格数据"""
        try:
            hist = ticker.history(period="1y", interval="1d")

            if hist.empty:
                return {}

            returns = hist['Close'].pct_change()

            # 转换时间序列数据为列表格式
            time_series_data = []
            for date, row in hist.iterrows():
                # 将时区敏感的Timestamp转换为可序列化的格式
                timestamp = date.tz_convert(None) if hasattr(date, 'tz') else date

                time_series_data.append({
                    'date': timestamp,
                    'open': float(row['Open']),
                    'close': float(row['Close']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                })

            return {
                'time_series': time_series_data,
                'volatility_30d': returns.tail(30).std() * np.sqrt(252),
                'volatility_90d': returns.tail(90).std() * np.sqrt(252),
                'momentum_1m': hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1,
                'momentum_3m': hist['Close'].iloc[-1] / hist['Close'].iloc[-66] - 1,
                'volume_avg_30d': hist['Volume'].tail(30).mean()
            }
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return {}

    def _fetch_descriptions(self, ticker) -> Dict[str, str]:
        """获取公司描述信息"""
        try:
            info = ticker.info
            return {
                'business_summary': info.get('longBusinessSummary', ''),
            }
        except:
            return {}

    async def _save_to_mongodb(self, stock_data: StockRawData):
        """保存到MongoDB"""
        try:
            await self.stock_raw_collection.update_one(
                {'symbol': stock_data.symbol},
                {'$set': stock_data.to_dict()},
                upsert=True
            )
        except Exception as e:
            logger.error(f"保存到MongoDB失败 {stock_data.symbol}: {e}")

    async def compute_stock_vector_20d(self, symbol: str) -> np.ndarray:
        """计算20维股票向量"""
        try:
            # 使用 await 进行异步查询
            raw_data = await self.stock_raw_collection.find_one({'symbol': symbol})
            if not raw_data:
                logger.warning(f"未找到 {symbol} 的原始数据")
                return self._create_default_vector_20d()

            vector_20d = np.zeros(20)

            # 1. 行业特征 (0-10维)
            sector_vector = self._compute_sector_vector(raw_data)
            vector_20d[0:11] = sector_vector

            # 2. 投资特征 (11-19维)
            investment_vector = self._compute_investment_features(raw_data)
            vector_20d[11:20] = investment_vector

            # 归一化向量，确保与用户向量计算相似度的一致性
            norm = np.linalg.norm(vector_20d)
            if norm > 0:
                vector_20d = vector_20d / norm
            logger.info(f"stock_vector: {vector_20d}")
            return vector_20d

        except Exception as e:
            logger.error(f"计算 {symbol} 向量失败: {e}")
            return self._create_default_vector_20d()

    def _compute_sector_vector(self, raw_data: Dict) -> np.ndarray:
        """计算行业特征向量 (11维)"""
        sector_vector = np.zeros(11)
        basic_info = raw_data.get('basic_info', {})
        sector = basic_info.get('sector', 'Unknown')

        # 找到行业在列表中的索引
        try:
            sector_index = self.sector_list.index(sector)
            sector_vector[sector_index] = 1.0
        except ValueError:
            # 如果行业不在预定义列表中，均匀分布
            sector_vector[:] = 1.0 / len(self.sector_list)

        return sector_vector

    def _compute_investment_features(self, raw_data: Dict) -> np.ndarray:
        """计算投资特征 (9维)"""
        features = np.zeros(9)
        basic_info = raw_data.get('basic_info', {})
        financials = raw_data.get('financials', {})
        historical_data = raw_data.get('historical_data', {})

        try:
            key_ratios = financials.get('key_ratios', {})
            valuation_metrics = financials.get('valuation_metrics', {})
            dividend_info = financials.get('dividend_info', {})

            # 11. 市值得分 (0=小盘, 1=大盘)
            market_cap = basic_info.get('market_cap', 0)
            log_market_cap = np.log1p(market_cap)
            features[0] = min(log_market_cap / np.log1p(3e12), 1.0)

            # 12. 成长价值得分 (0=价值型, 1=成长型)
            pe_ratio = valuation_metrics.get('trailing_pe', 0)
            revenue_growth = key_ratios.get('revenue_growth', 0)
            if pe_ratio > 0 and revenue_growth > 0:
                # 简化PEG计算，低PEG偏向价值，高PEG偏向成长
                peg_ratio = pe_ratio / (revenue_growth * 100)
                features[1] = min(peg_ratio / 2.0, 1.0)  # 假设PEG=2为成长型上限
            else:
                features[1] = 0.5

            # 13. 股息吸引力 (0=无股息, 1=高股息)
            dividend_yield = dividend_info.get('dividend_yield', 0)
            payout_ratio = dividend_info.get('payout_ratio', 0)
            if payout_ratio > 0 and payout_ratio < 0.8:  # 派息率合理
                features[2] = min(dividend_yield * 10, 1.0)  # 10%股息率为上限
            else:
                features[2] = 0.0

            # 14. 风险水平 (0=低风险, 1=高风险)
            volatility = historical_data.get('volatility_30d', 0)
            beta = key_ratios.get('beta', 1.0)  # 假设有beta数据
            risk_score = (volatility + abs(beta - 1.0)) / 2.0
            features[3] = min(risk_score, 1.0)

            # 15. 流动性得分 (0=低流动性, 1=高流动性)
            volume_avg = historical_data.get('volume_avg_30d', 0)
            features[4] = min(np.log1p(volume_avg) / 20, 1.0)

            # 16. 质量得分 (0=低质量, 1=高质量)
            profit_margin = key_ratios.get('profit_margin', 0)
            roe = key_ratios.get('return_on_equity', 0)
            quality_score = (max(profit_margin, 0) + max(roe, 0)) / 2.0
            features[5] = min(quality_score, 1.0)

            # 17. 估值安全边际 (0=高估, 1=低估)
            pe_ratio = valuation_metrics.get('trailing_pe', 0)
            pb_ratio = valuation_metrics.get('price_to_book', 0)
            # 简化估值安全计算
            if pe_ratio > 0 and pb_ratio > 0:
                valuation_score = 1.0 / (pe_ratio * 0.01 + pb_ratio)  # 低PE/PB更安全
                features[6] = min(valuation_score * 10, 1.0)
            else:
                features[6] = 0.5

            # 18. 动量强度 (0=弱势, 1=强势)
            momentum_3m = historical_data.get('momentum_3m', 0)
            momentum_1m = historical_data.get('momentum_1m', 0)
            momentum_score = (momentum_3m + momentum_1m) / 2.0 + 0.5  # 归一化到0-1
            features[7] = max(0, min(momentum_score, 1.0))

            # 19. 经营效率得分 (0=低效, 1=高效)
            operating_margin = key_ratios.get('operating_margin', 0)
            asset_turnover = 2.0  # 简化假设，实际需要计算
            efficiency_score = (max(operating_margin, 0) + min(asset_turnover / 3.0, 1.0)) / 2.0
            features[8] = min(efficiency_score, 1.0)

        except Exception as e:
            logger.error(f"计算投资特征失败: {e}")

        return features

    def _create_default_vector_20d(self) -> np.ndarray:
        """创建默认股票向量"""
        vector = np.zeros(20)
        # 行业特征均匀分布
        vector[0:11] = 1.0 / 11
        # 投资特征设为中性
        vector[11:20] = 0.5

        # 归一化向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    async def update_stock_vectors(self, symbols: List[str]) -> Dict[str, Any]:
        """更新股票向量（使用20维向量）"""
        results = {
            "updated": 0,
            "failed": [],
            "success_symbols": []
        }

        for symbol in symbols:
            try:
                # 计算20维向量 - 使用 await
                vector_20d = await self.compute_stock_vector_20d(symbol)
                # logger.info(f"vector_20d: {vector_20d}")
                # 从MongoDB获取基本信息 - 使用 await
                raw_data = await self.stock_raw_collection.find_one({'symbol': symbol})
                basic_info = raw_data.get('basic_info', {}) if raw_data else {}

                # 更新或创建PostgreSQL记录
                existing = self.postgres_db.query(StockVector).filter(StockVector.symbol == symbol).first()
                # logger.info(f"existing: {existing}")
                if existing:
                    existing.set_vector_20d(vector_20d)
                    existing.name = basic_info.get('name', symbol)
                    existing.sector = basic_info.get('sector', 'Unknown')
                    existing.industry = basic_info.get('industry', 'Unknown')
                    existing.updated_at = datetime.utcnow()
                else:
                    stock_vector = StockVector(
                        symbol=symbol,
                        name=basic_info.get('name', symbol),
                        sector=basic_info.get('sector', 'Unknown'),
                        industry=basic_info.get('industry', 'Unknown')
                    )
                    stock_vector.set_vector_20d(vector_20d)
                    self.postgres_db.add(stock_vector)

                results["updated"] += 1
                results["success_symbols"].append(symbol)
                logger.info(f"更新 {symbol} 向量成功")

            except Exception as e:
                logger.error(f"更新 {symbol} 向量失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                results["failed"].append(symbol)

        self.postgres_db.commit()
        return results

    def recommend_stocks(self, user_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """推荐股票（基于20维向量相似度）"""
        stock_vectors = self.postgres_db.query(StockVector).all()

        if not stock_vectors:
            return []

        recommendations = []
        user_vec_2d = user_vector.reshape(1, -1)

        for stock in stock_vectors:
            try:
                stock_vec = stock.get_vector_20d()
                # 确保向量维度匹配
                if len(stock_vec) != len(user_vector):
                    logger.warning(f"股票 {stock.symbol} 向量维度不匹配: {len(stock_vec)} vs {len(user_vector)}")
                    continue

                stock_vec_2d = stock_vec.reshape(1, -1)
                similarity = cosine_similarity(user_vec_2d, stock_vec_2d)[0][0]

                recommendations.append({
                    'symbol': stock.symbol,
                    'name': stock.name,
                    'sector': stock.sector,
                    'industry': stock.industry,
                    'similarity': float(similarity),
                    'updated_at': stock.updated_at.isoformat()
                })
            except Exception as e:
                logger.error(f"计算 {stock.symbol} 相似度失败: {e}")
                continue

        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        return recommendations[:top_k]

    def get_all_stocks(self) -> List[Dict]:
        """获取所有股票信息"""
        stocks = self.postgres_db.query(StockVector).all()
        return [
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "sector": stock.sector,
                "industry": stock.industry,
                "updated_at": stock.updated_at.isoformat()
            }
            for stock in stocks
        ]