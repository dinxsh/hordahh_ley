import React, { useState } from 'react';

const TakeProfitHandler = ({ rowId, initialValue = 1.16 }) => {
  const [value, setValue] = useState(initialValue);
  const [status, setStatus] = useState('');

  const updateTakeProfit = async () => {
    try {
      // Extract numeric row ID from "row-X" format
      const numericRowId = rowId.replace('row-', '');
      
      const response = await fetch('/order/update_take_profit', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          row_id: numericRowId,
          take_profit: parseFloat(value)
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to update take profit');
      }

      setStatus('Updated');
      setTimeout(() => setStatus(''), 3000);

    } catch (err) {
      console.error('Take profit update error:', err);
      setStatus(`Error: ${err.message}`);
      setTimeout(() => setStatus(''), 5000);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <input
        type="number"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="w-16 px-2 py-1 bg-gray-800 border border-gray-600 rounded"
        min="0.1"
        max="10"
        step="0.1"
      />
      <button 
        onClick={updateTakeProfit}
        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded"
      >
        Update
      </button>
      {status && (
        <span className={status.includes('Error') ? 'text-red-500' : 'text-green-500'}>
          {status}
        </span>
      )}
    </div>
  );
};

export default TakeProfitHandler;