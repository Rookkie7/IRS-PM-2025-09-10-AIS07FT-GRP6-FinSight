import { useState } from 'react';
import PairSelectionTab from './PairSelectionTab';
import TradingResultsTab from './TradingResultsTab';

export default function PairsTradingTab() {
  const [showResults, setShowResults] = useState(false);

  return (
    <div className="p-6">
      {!showResults ? (
        <PairSelectionTab onProceed={() => setShowResults(true)} />
      ) : (
        <div>
          <button
            onClick={() => setShowResults(false)}
            className="mb-6 text-blue-600 hover:text-blue-700 font-semibold flex items-center gap-2"
          >
            ← Back to Pair Selection
          </button>
          <TradingResultsTab />
        </div>
      )}
    </div>
  );
}
