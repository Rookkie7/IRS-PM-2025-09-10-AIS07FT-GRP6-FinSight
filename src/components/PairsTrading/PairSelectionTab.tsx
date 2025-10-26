import { useState } from 'react';
import { TrendingUp, ArrowRight } from 'lucide-react';

interface PairSelectionTabProps {
  onProceed: () => void;
}

export default function PairSelectionTab({ onProceed }: PairSelectionTabProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const selectedPairs = [
    { name: 'LLY_UNP', stock1: 'LLY', stock2: 'UNP', score: 0.89 },
    { name: 'MO_WELL', stock1: 'MO', stock2: 'WELL', score: 0.85 },
    { name: 'RTX_UBER', stock1: 'RTX', stock2: 'UBER', score: 0.82 },
    { name: 'XOM_LIN', stock1: 'XOM', stock2: 'LIN', score: 0.78 }
  ];

  const handleExecute = () => {
    setIsRunning(true);
    setTimeout(() => {
      setIsRunning(false);
      setShowResults(true);
    }, 2000);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Pair Selection Algorithm</h2>
        <p className="text-gray-600 text-lg">
          Our pair selection methodology identifies optimal stock pairs for statistical arbitrage trading.
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 mb-8">
        <h3 className="text-xl font-semibold text-gray-900 mb-6">Selection Methodology</h3>

        <div className="space-y-6">
          <div className="border-l-4 border-blue-500 pl-6">
            <h4 className="font-semibold text-gray-900 mb-2">Step 1: Cointegration Analysis</h4>
            <p className="text-gray-600">
              We use the Engle-Granger two-step method to test for cointegration between stock pairs.
              Pairs with significant cointegration (p-value &lt; 0.05) are selected as candidates.
            </p>
          </div>

          <div className="border-l-4 border-green-500 pl-6">
            <h4 className="font-semibold text-gray-900 mb-2">Step 2: Correlation Coefficient</h4>
            <p className="text-gray-600">
              Calculate Pearson correlation coefficients for all candidate pairs. Higher correlation
              indicates stronger historical price relationships.
            </p>
          </div>

          <div className="border-l-4 border-purple-500 pl-6">
            <h4 className="font-semibold text-gray-900 mb-2">Step 3: Half-Life of Mean Reversion</h4>
            <p className="text-gray-600">
              Estimate the half-life of mean reversion for the spread. Shorter half-lives indicate
              faster mean reversion, making the pair more suitable for trading.
            </p>
          </div>

          <div className="border-l-4 border-orange-500 pl-6">
            <h4 className="font-semibold text-gray-900 mb-2">Step 4: Ranking and Selection</h4>
            <p className="text-gray-600">
              Rank pairs based on combined scores from cointegration strength, correlation, and
              mean reversion speed. Select the top 4 pairs for trading.
            </p>
          </div>
        </div>
      </div>

      <div className="text-center mb-8">
        <button
          onClick={handleExecute}
          disabled={isRunning || showResults}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold px-8 py-4 rounded-lg transition-colors duration-200 inline-flex items-center gap-2 text-lg"
        >
          {isRunning ? (
            <>
              <div className="w-5 h-5 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
              Executing Pair Selection...
            </>
          ) : showResults ? (
            'Selection Complete'
          ) : (
            <>
              <TrendingUp className="w-5 h-5" />
              Execute Pair Selection
            </>
          )}
        </button>
      </div>

      {showResults && (
        <div className="animate-fadeIn">
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-8 mb-8 border border-green-200">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-green-600" />
              Selected Pairs
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {selectedPairs.map((pair, index) => (
                <div key={pair.name} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-semibold text-gray-500">Rank #{index + 1}</span>
                    <span className="text-sm font-semibold text-green-600">Score: {pair.score}</span>
                  </div>
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-lg font-bold text-blue-600">{pair.stock1}</span>
                    <span className="text-gray-400">↔</span>
                    <span className="text-lg font-bold text-blue-600">{pair.stock2}</span>
                  </div>
                  <div className="text-sm text-gray-500 font-mono">{pair.name}</div>
                </div>
              ))}
            </div>

            <div className="text-center">
              <button
                onClick={onProceed}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold px-8 py-4 rounded-lg transition-all duration-200 inline-flex items-center gap-2 text-lg shadow-lg hover:shadow-xl"
              >
                Proceed to Trading Methods
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
