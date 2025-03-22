import React, { useState, useEffect } from 'react';

const PricePrecisionHandler = ({ pair, initialPrice, onPrecisionUpdate }) => {
  const [precision, setPrecision] = useState(null);
  const [attempts, setAttempts] = useState(0);
  const MAX_ATTEMPTS = 3;

  // Function to extract precision from Kraken error message
  const extractPrecision = (errorMessage) => {
    const match = /up to (\d+) decimal/.exec(errorMessage);
    return match ? parseInt(match[1]) : null;
  };

  // Function to format price with proper precision
  const formatPrice = (price, precisionValue) => {
    try {
      return Number(price).toFixed(precisionValue);
    } catch (error) {
      console.error('Error formatting price:', error);
      return price;
    }
  };

  // Function to validate price format with Kraken
  const validateWithKraken = async (formattedPrice) => {
    try {
      const response = await fetch('/validate-price', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          pair,
          price: formattedPrice
        })
      });

      const data = await response.json();
      if (data.error) {
        // Extract new precision from error message if available
        const newPrecision = extractPrecision(data.error);
        if (newPrecision !== null) {
          setPrecision(newPrecision);
          return { success: false, newPrecision };
        }
      }
      return { success: true };
    } catch (error) {
      console.error('Validation error:', error);
      return { success: false, error };
    }
  };

  // Effect to handle price precision updates
  useEffect(() => {
    const updatePrecision = async () => {
      if (attempts >= MAX_ATTEMPTS) {
        console.error('Max attempts reached for precision adjustment');
        return;
      }

      const currentPrecision = precision || 4; // Default to 4 decimals
      const formattedPrice = formatPrice(initialPrice, currentPrecision);
      
      const validation = await validateWithKraken(formattedPrice);
      if (!validation.success && validation.newPrecision) {
        setPrecision(validation.newPrecision);
        setAttempts(prev => prev + 1);
        onPrecisionUpdate(validation.newPrecision);
      }
    };

    if (initialPrice) {
      updatePrecision();
    }
  }, [initialPrice, precision, attempts]);

  return null; // This is a utility component, no UI needed
};

export default PricePrecisionHandler;