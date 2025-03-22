import React, { useState, useEffect } from 'react';

const OrderMonitor = () => {
  const [orderState, setOrderState] = useState({
    status: 'initializing',
    lastUpdate: null,
    history: []
  });

  useEffect(() => {
    const pollOrder = async () => {
      try {
        // Find current auto-trade row ID from window.currentAutoTradeId
        const currentId = window.currentAutoTradeId;
        if (!currentId) {
          setOrderState(prev => ({
            ...prev,
            status: 'waiting',
            lastUpdate: new Date()
          }));
          return;
        }

        const response = await fetch(`/order/status/${currentId}`);
        const data = await response.json();
        
        // Add validation to confirm order completion
        if (data.status === 'closed' || data.status === 'completed') {
          const krakenCheck = await fetch('/api/kraken/order/' + data.orderId);
          const krakenData = await krakenCheck.json();
          
          if (krakenData.status === 'closed') {
            setOrderState(prev => ({
              ...prev,
              status: 'completed',
              lastUpdate: new Date(),
              history: [
                ...prev.history,
                {
                  timestamp: new Date(),
                  event: 'Order completion verified',
                  details: krakenData
                }
              ]
            }));
            return;
          }
        }
        
        setOrderState(prev => ({
          ...prev,
          status: data.status,
          lastUpdate: new Date(),
          history: [
            ...prev.history,
            {
              timestamp: new Date(),
              event: 'Status update',
              details: data
            }
          ]
        }));
      } catch (error) {
        console.error('Error polling order:', error);
      }
    };

    const interval = setInterval(pollOrder, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white mb-2">Order Monitor</h2>
      </div>
      <div className="space-y-4">
        <div className="flex justify-between items-center p-4 bg-gray-700 rounded">
          <span className="font-medium text-white">Current Status:</span>
          <span className={`px-3 py-1 rounded ${
            orderState.status === 'completed' ? 'bg-green-700 text-white' :
            orderState.status === 'error' ? 'bg-red-700 text-white' :
            'bg-blue-700 text-white'
          }`}>
            {orderState.status}
          </span>
        </div>
        
        <div className="border border-gray-600 rounded p-4">
          <h3 className="font-medium mb-2 text-white">Order History</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {orderState.history.map((entry, idx) => (
              <div key={idx} className="text-sm border-b border-gray-600 pb-2">
                <div className="flex justify-between text-gray-300">
                  <span>{entry.timestamp.toLocaleTimeString()}</span>
                  <span>{entry.event}</span>
                </div>
                {entry.details && (
                  <pre className="mt-1 text-xs bg-gray-900 text-gray-300 p-2 rounded">
                    {JSON.stringify(entry.details, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrderMonitor;