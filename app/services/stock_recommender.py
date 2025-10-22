import numpy as np
import logging
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from pymongo.database import Database
from datetime import datetime, timedelta
import yfinance as yf
import asyncio

from app.model.models import StockVector
from app.services.stock_service import StockService

logger = logging.getLogger(__name__)


class MultiObjectiveRecommender:
    def __init__(self, postgres_db: Session, mongo_db: Database):
        self.postgres_db = postgres_db
        self.mongo_db = mongo_db
        self.stock_service = StockService(postgres_db, mongo_db)
        self.stock_raw_collection = mongo_db["stocks"]
        self._market_regime_cache = None
        self._market_regime_cache_time = None

    async def recommend_stocks(self, user_vector: np.ndarray, top_k: int = 10,
                               user_risk_profile: str = "balanced") -> List[Dict]:
        """Multi-objective stock recommendation main function (performance optimized)"""
        try:
            # Get all stocks
            all_stocks = self.postgres_db.query(StockVector).all()

            if not all_stocks:
                logger.warning("No stock data found")
                return []

            # Batch get all stock data (avoid repeated queries)
            stock_symbols = [stock.symbol for stock in all_stocks]
            stock_data_map = await self._batch_get_stock_data(stock_symbols)

            # Pre-calculate market regime (avoid repeated calculations)
            market_regime = await self._get_current_market_regime_cached()

            # Parallel calculation of various scores
            base_similarities = self._calculate_base_similarity(user_vector, all_stocks)

            # Use parallel processing for risk scores and market timing scores
            risk_scores, timing_scores = await asyncio.gather(
                self._calculate_risk_adjusted_return_batch(all_stocks, stock_data_map),
                self._calculate_market_timing_batch(all_stocks, stock_data_map, market_regime)
            )

            diversification_scores = self._calculate_diversification_benefit(all_stocks)

            # Dynamic weight adjustment
            weights = self._calculate_weights(user_risk_profile)

            # Batch generate recommendation explanations
            recommendations = await self._generate_recommendations_batch(
                all_stocks, base_similarities, risk_scores,
                diversification_scores, timing_scores, weights,
                stock_data_map, user_risk_profile, market_regime
            )

            # Sort by final score and return top_k
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            return recommendations[:top_k]

        except Exception as e:
            logger.error(f"Multi-objective recommendation failed: {e}")
            return []

    async def _batch_get_stock_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Batch get stock data"""
        try:
            cursor = self.stock_raw_collection.find({'symbol': {'$in': symbols}})
            stock_data_map = {}
            async for doc in cursor:
                stock_data_map[doc['symbol']] = doc
            logger.info(f"Batch retrieved data for {len(stock_data_map)} stocks")
            return stock_data_map
        except Exception as e:
            logger.error(f"Batch stock data retrieval failed: {e}")
            return {}

    async def _calculate_risk_adjusted_return_batch(self, stocks: List[StockVector],
                                                    stock_data_map: Dict) -> List[float]:
        """Batch calculate risk-adjusted returns"""
        risk_scores = []

        for stock in stocks:
            try:
                raw_data = stock_data_map.get(stock.symbol)
                if not raw_data:
                    risk_scores.append(0.0)
                    continue

                expected_return = self._estimate_expected_return(raw_data)
                volatility = self._calculate_volatility(raw_data)

                # Sharpe ratio style calculation (simplified)
                if volatility > 0.01:
                    risk_adjusted = expected_return / volatility
                else:
                    risk_adjusted = expected_return

                # Normalize to 0-1
                normalized_score = min(max(risk_adjusted, 0), 2) / 2.0
                risk_scores.append(normalized_score)

            except Exception as e:
                logger.error(f"Failed to calculate risk-adjusted return for {stock.symbol}: {e}")
                risk_scores.append(0.0)

        return risk_scores

    async def _calculate_market_timing_batch(self, stocks: List[StockVector],
                                             stock_data_map: Dict,
                                             market_regime: str) -> List[float]:
        """Batch calculate market timing scores"""
        timing_scores = []

        for stock in stocks:
            try:
                raw_data = stock_data_map.get(stock.symbol)
                if not raw_data:
                    timing_scores.append(0.0)
                    continue

                financials = raw_data.get('financials', {})
                key_ratios = financials.get('key_ratios', {})
                historical_data = raw_data.get('historical_data', {})
                dividend_info = financials.get('dividend_info', {})

                beta = key_ratios.get('beta', 1.0) or 1.0
                volatility = historical_data.get('volatility_30d', 0.2) or 0.2
                dividend_yield = dividend_info.get('dividend_yield', 0) or 0

                if market_regime == 'bull':
                    # Bull market prefers high beta, high growth
                    growth_score = min(beta * 0.8, 1.0)
                    timing_score = growth_score
                elif market_regime == 'bear':
                    # Bear market prefers low volatility, high dividend
                    defense_score = (1 - min(volatility, 1.0)) * 0.6 + min(dividend_yield * 5, 1.0) * 0.4
                    timing_score = defense_score
                else:  # sideways market
                    # Prefers reasonable valuation, medium volatility
                    valuation_score = 0.5
                    timing_score = (1 - min(volatility, 1.0)) * 0.5 + valuation_score * 0.5

                timing_scores.append(timing_score)

            except Exception as e:
                logger.error(f"Failed to calculate market timing for {stock.symbol}: {e}")
                timing_scores.append(0.0)

        return timing_scores

    async def _generate_recommendations_batch(self, stocks: List[StockVector],
                                              base_similarities: List[float],
                                              risk_scores: List[float],
                                              diversification_scores: List[float],
                                              timing_scores: List[float],
                                              weights: Dict[str, float],
                                              stock_data_map: Dict,
                                              user_risk_profile: str,
                                              market_regime: str) -> List[Dict]:
        """Batch generate recommendation results"""
        recommendations = []

        # Use async task list to parallel generate recommendation explanations
        tasks = []
        for i, stock in enumerate(stocks):
            task = self._create_recommendation_task(
                stock, i, base_similarities, risk_scores,
                diversification_scores, timing_scores, weights,
                stock_data_map, user_risk_profile, market_regime
            )
            tasks.append(task)

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Recommendation generation failed: {result}")
                continue
            if result:
                recommendations.append(result)

        return recommendations

    async def _create_recommendation_task(self, stock: StockVector, index: int,
                                          base_similarities: List[float],
                                          risk_scores: List[float],
                                          diversification_scores: List[float],
                                          timing_scores: List[float],
                                          weights: Dict[str, float],
                                          stock_data_map: Dict,
                                          user_risk_profile: str,
                                          market_regime: str) -> Optional[Dict]:
        """Create single recommendation task"""
        try:
            final_score = (
                    base_similarities[index] * weights['preference'] +
                    risk_scores[index] * weights['risk_return'] +
                    diversification_scores[index] * weights['diversification'] +
                    timing_scores[index] * weights['timing']
            )

            # Generate detailed recommendation explanation
            detailed_explanation = await self._generate_detailed_explanation(
                stock.symbol,
                {
                    'pref_score': base_similarities[index],
                    'risk_score': risk_scores[index],
                    'div_score': diversification_scores[index],
                    'timing_score': timing_scores[index],
                    'stock_sector': stock.sector,
                    'stock_name': stock.name
                },
                user_risk_profile,
                stock_data_map.get(stock.symbol),
                market_regime
            )

            return {
                'symbol': stock.symbol,
                'name': stock.name,
                'sector': stock.sector,
                'industry': stock.industry,
                'final_score': round(final_score, 4),
                'component_scores': {
                    'preference': round(base_similarities[index], 4),
                    'risk_adjusted': round(risk_scores[index], 4),
                    'diversification': round(diversification_scores[index], 4),
                    'market_timing': round(timing_scores[index], 4)
                },
                'explanation': detailed_explanation,
                'weight_used': weights
            }
        except Exception as e:
            logger.error(f"Failed to create recommendation task for {stock.symbol}: {e}")
            return None

    async def _get_current_market_regime_cached(self) -> str:
        """Market regime judgment with caching"""
        current_time = datetime.now()

        # Cache for 5 minutes
        if (self._market_regime_cache and
                self._market_regime_cache_time and
                (current_time - self._market_regime_cache_time).total_seconds() < 300):
            return self._market_regime_cache

        # Use thread pool to execute synchronous yfinance operations, avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            regime = await loop.run_in_executor(None, self._get_current_market_regime_sync)
            self._market_regime_cache = regime
            self._market_regime_cache_time = current_time
            return regime
        except Exception as e:
            logger.error(f"Failed to get market regime: {e}")
            return "sideways"

    def _get_current_market_regime_sync(self) -> str:
        """Synchronous version of market regime judgment"""
        try:
            # Use SPY as market benchmark
            spy = yf.Ticker("SPY")
            hist = spy.history(period="3mo")  # Shorter period for faster speed

            if len(hist) < 20:
                return "sideways"

            # Calculate recent returns and volatility
            returns = hist['Close'].pct_change().dropna()
            recent_return = returns.tail(20).mean() * 252
            recent_volatility = returns.tail(20).std() * np.sqrt(252)

            if recent_return > 0.1 and recent_volatility < 0.2:
                return "bull"
            elif recent_return < -0.05 and recent_volatility > 0.25:
                return "bear"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Failed to get market regime synchronously: {e}")
            return "sideways"

    def _calculate_base_similarity(self, user_vector: np.ndarray, stocks: List[StockVector]) -> List[float]:
        """Calculate base similarity (user preference)"""
        similarities = []
        user_vec_2d = user_vector.reshape(1, -1)

        for stock in stocks:
            try:
                stock_vec = stock.get_vector_20d()
                if len(stock_vec) != len(user_vector):
                    logger.warning(f"Vector dimension mismatch: {stock.symbol}")
                    similarities.append(0.0)
                    continue

                stock_vec_2d = stock_vec.reshape(1, -1)
                similarity = cosine_similarity(user_vec_2d, stock_vec_2d)[0][0]
                similarities.append(float(similarity))
            except Exception as e:
                logger.error(f"Failed to calculate similarity for {stock.symbol}: {e}")
                similarities.append(0.0)

        return similarities

    def _estimate_expected_return(self, raw_data: Dict) -> float:
        """Estimate expected return"""
        try:
            if not raw_data or not isinstance(raw_data, dict):
                logger.warning("Raw data is empty or not dictionary type")
                return 0.05  # Default 5% annual return

            financials = raw_data.get('financials', {})
            historical_data = raw_data.get('historical_data', {})
            key_ratios = financials.get('key_ratios', {})
            valuation_metrics = financials.get('valuation_metrics', {})

            # Multi-factor expected return model
            historical_return = historical_data.get('momentum_3m', 0) or 0
            growth_projection = key_ratios.get('revenue_growth', 0) or 0
            pe_ratio = valuation_metrics.get('trailing_pe', 15) or 15

            # Valuation reversal effect: low PE may have higher expected return
            valuation_effect = 1.0 / max(pe_ratio, 5)

            # Comprehensive expected return (annualized)
            expected_return = (
                    historical_return * 0.3 +
                    growth_projection * 0.4 +
                    valuation_effect * 0.3
            )

            result = max(expected_return, 0)  # Ensure non-negative
            return result

        except Exception as e:
            logger.error(f"Failed to estimate expected return: {e}")
            return 0.05  # Default 5% annual return

    def _calculate_volatility(self, raw_data: Dict) -> float:
        """Calculate volatility"""
        try:
            if not raw_data or not isinstance(raw_data, dict):
                logger.warning("Raw data is empty or not dictionary type")
                return 0.2  # Default 20% volatility

            historical_data = raw_data.get('historical_data', {})
            volatility_30d = historical_data.get('volatility_30d', 0.2) or 0.2

            # Ensure volatility is not too small
            result = max(volatility_30d, 0.1)
            return result

        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return 0.2  # Default 20% volatility

    def _calculate_diversification_benefit(self, stocks: List[StockVector]) -> List[float]:
        """Calculate diversification benefit"""
        try:
            # Get all sectors
            all_sectors = list(set([stock.sector for stock in stocks if stock.sector]))
            sector_counts = {sector: 0 for sector in all_sectors}

            # Count stocks in each sector
            for stock in stocks:
                if stock.sector in sector_counts:
                    sector_counts[stock.sector] += 1

            # Calculate sector concentration
            total_stocks = len(stocks)
            sector_concentration = {
                sector: count / total_stocks
                for sector, count in sector_counts.items()
            }

            # Calculate diversification benefit for each stock
            diversification_scores = []
            for stock in stocks:
                if stock.sector in sector_concentration:
                    # More concentrated sectors have lower diversification benefit
                    concentration = sector_concentration[stock.sector]
                    diversity_score = 1.0 - concentration
                else:
                    diversity_score = 1.0  # New sector, high diversification benefit

                diversification_scores.append(diversity_score)

            return diversification_scores

        except Exception as e:
            logger.error(f"Failed to calculate diversification benefit: {e}")
            return [0.0] * len(stocks)

    def _calculate_weights(self, user_risk_profile: str) -> Dict[str, float]:
        """Dynamically adjust weights based on user risk preference"""
        weight_profiles = {
            'conservative': {'preference': 0.2, 'risk_return': 0.4, 'diversification': 0.3, 'timing': 0.1},
            'balanced': {'preference': 0.3, 'risk_return': 0.4, 'diversification': 0.2, 'timing': 0.1},
            'aggressive': {'preference': 0.3, 'risk_return': 0.5, 'diversification': 0.1, 'timing': 0.1}
        }
        return weight_profiles.get(user_risk_profile, weight_profiles['balanced'])

    async def _generate_detailed_explanation(self, stock_symbol: str,
                                             component_scores: Dict,
                                             user_risk_profile: str,
                                             raw_data: Optional[Dict] = None,
                                             market_regime: str = "sideways") -> str:
        """Generate detailed explanation based on specific stock characteristics"""

        try:
            if not raw_data:
                return self._generate_basic_explanation(component_scores)

            basic_info = raw_data.get('basic_info', {})
            financials = raw_data.get('financials', {})
            historical_data = raw_data.get('historical_data', {})

            stock_name = basic_info.get('name', stock_symbol)
            sector = basic_info.get('sector', 'Unknown Sector')

            key_ratios = financials.get('key_ratios', {})
            valuation_metrics = financials.get('valuation_metrics', {})

            reasons = []

            # 1. User preference matching details
            pref_score = component_scores['pref_score']
            if pref_score > 0.8:
                reasons.append(f"Highly aligned with your investment preferences ({pref_score * 100:.1f}% match)")
            elif pref_score > 0.6:
                reasons.append(f"Matches your investment style ({pref_score * 100:.1f}% match)")
            elif pref_score > 0.4:
                reasons.append(f"Has some alignment with your preferences")
            else:
                reasons.append("While preference match is limited, excels in other dimensions")

            # 2. Fundamental analysis
            revenue_growth = key_ratios.get('revenue_growth', 0)
            profit_margin = key_ratios.get('profit_margin', 0)
            roe = key_ratios.get('return_on_equity', 0)

            if revenue_growth > 0.15:
                reasons.append(f"Strong revenue growth ({revenue_growth * 100:.1f}%), has growth potential")
            elif revenue_growth > 0.05:
                reasons.append(f"Stable revenue growth ({revenue_growth * 100:.1f}%)")

            if profit_margin > 0.2:
                reasons.append(f"Excellent profitability (margin {profit_margin * 100:.1f}%)")
            elif profit_margin > 0.1:
                reasons.append(f"Good profitability")

            if roe > 0.15:
                reasons.append(f"High shareholder return (ROE {roe * 100:.1f}%)")

            # 3. Valuation analysis
            pe_ratio = valuation_metrics.get('trailing_pe', 0)
            pb_ratio = valuation_metrics.get('price_to_book', 0)

            if 0 < pe_ratio < 15:
                reasons.append(f"Reasonable valuation (PE {pe_ratio:.1f})")
            elif pe_ratio >= 15:
                reasons.append(f"Higher valuation (PE {pe_ratio:.1f}), but growth may justify it")

            # 4. Risk characteristics
            risk_score = component_scores['risk_score']
            beta = key_ratios.get('beta', 1.0)
            volatility = historical_data.get('volatility_30d', 0.2)

            if risk_score > 0.7:
                reasons.append("Excellent risk-reward ratio, expected returns significantly exceed risk")
            elif risk_score > 0.5:
                reasons.append("Balanced risk-reward, provides good returns with controlled risk")
            elif risk_score > 0.3:
                reasons.append("Risk-adjusted returns at reasonable levels")
            else:
                reasons.append("Conservative risk-reward profile, suitable for risk-averse investors")

            if beta > 1.2:
                reasons.append("High beta coefficient, more volatile than market, suitable for risk-tolerant investors")
            elif beta < 0.8:
                reasons.append("Low beta coefficient, defensive characteristics, relatively stable volatility")

            # 5. Industry position and diversification value
            div_score = component_scores['div_score']
            market_cap = basic_info.get('market_cap', 0)

            if div_score > 0.7:
                reasons.append(f"As a {sector} sector representative, effectively diversifies portfolio risk")
            elif div_score > 0.5:
                reasons.append(f"Helps enhance portfolio allocation in {sector} sector")
            else:
                reasons.append("While sector concentration is higher, excels in other dimensions")

            if market_cap > 100e9:  # $100B
                reasons.append(f"Industry leader, large market cap, strong risk resistance")
            elif market_cap > 10e9:  # $10B
                reasons.append(f"Mid-cap company, has growth potential with reasonable stability")
            else:
                reasons.append(f"Small-cap company, higher growth potential but relatively higher risk")

            # 6. Match with user risk preference
            if user_risk_profile == "conservative" and beta < 0.9 and volatility < 0.2:
                reasons.append("Highly matches your conservative investment style")
            elif user_risk_profile == "aggressive" and (beta > 1.1 or revenue_growth > 0.2):
                reasons.append("Aligns with your aggressive investment preference")
            elif user_risk_profile == "balanced" and 0.4 <= risk_score <= 0.7:
                reasons.append("Well-suited to your balanced investment strategy")

            # 7. Market timing adaptation
            timing_score = component_scores['timing_score']

            if timing_score > 0.7:
                if market_regime == "bull":
                    reasons.append("Strong upward momentum in current bull market environment")
                elif market_regime == "bear":
                    reasons.append("Good defensive characteristics in bear market conditions")
                else:
                    reasons.append("Relatively stable performance in current sideways market")
            elif timing_score > 0.5:
                reasons.append("Good adaptation to current market environment")

            # 8. Comprehensive rating
            final_score = (pref_score + risk_score + div_score + timing_score) / 4
            if final_score > 0.75:
                reasons.append("Overall Rating: Strongly Recommended")
            elif final_score > 0.6:
                reasons.append("Overall Rating: Recommended")
            elif final_score > 0.5:
                reasons.append("Overall Rating: Cautiously Recommended")
            else:
                reasons.append("Overall Rating: Watch")

            return "; ".join(
                reasons) if reasons else "Comprehensive assessment shows good performance, recommended for attention"

        except Exception as e:
            logger.error(f"Failed to generate detailed explanation: {e}")
            return self._generate_basic_explanation(component_scores)

    def _generate_basic_explanation(self, component_scores: Dict) -> str:
        """Basic recommendation explanation (fallback)"""
        pref_score = component_scores['pref_score']
        risk_score = component_scores['risk_score']
        div_score = component_scores['div_score']
        timing_score = component_scores['timing_score']

        reasons = []

        if pref_score > 0.7:
            reasons.append("Highly matches your investment preferences")
        elif pref_score > 0.5:
            reasons.append("Matches your investment style")

        if risk_score > 0.6:
            reasons.append("Excellent risk-reward ratio")
        elif risk_score < 0.4:
            reasons.append("Balanced risk-reward profile")

        if div_score > 0.7:
            reasons.append("Significantly enhances portfolio diversification")
        elif div_score > 0.5:
            reasons.append("Helps with investment portfolio diversification")

        if timing_score > 0.6:
            reasons.append("Suitable for current market environment")

        return "; ".join(reasons) if reasons else "Comprehensive assessment shows good performance"