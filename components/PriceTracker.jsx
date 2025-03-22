import React, { useState, useEffect } from 'react';

const PriceTracker = ({ rowId, initialPrice = 0, orderType }) => {
  const [priceDiff, setPriceDiff] = useState(0);
  const [isActive, setIsActive] = useState(true);

  useEffect(() => {
    if (!isActive) return;

    const fetchPrice = async () => {
      try {
        const response = await fetch(`/order/status/${rowId}`);
        const data = await response.json();
        
        if (data.status === 'sell_open' && data.current_price_diff !== undefined) {
          setPriceDiff(data.current_price_diff);
        } else if (data.status === 'buy_open' && data.buy_price_diff !== undefined) {
          setPriceDiff(data.buy_price_diff);
        } else if (['completed', 'cancelled', 'error'].includes(data.status)) {
          setIsActive(false);
        }
      } catch (error) {
        console.error('Error fetching price:', error);
      }
    };

    const interval = setInterval(fetchPrice, 3000);
    fetchPrice(); // Initial fetch

    return () => clearInterval(interval);
  }, [rowId, isActive]);

  const getColor = () => {
    if (orderType === 'buy') return 'text-yellow-400';
    return priceDiff >= 0 ? 'text-green-500' : 'text-red-500';
  };

  if (!isActive) return null;

  return (
    <span className={`ml-2 ${getColor()}`}>
      [{priceDiff >= 0 ? '+' : ''}{priceDiff.toFixed(2)}%]
    </span>
  );
};

export default PriceTracker;