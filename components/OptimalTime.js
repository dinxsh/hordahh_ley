const OptimalTimeDisplay = ({ pair }) => {
    const [optimalTime, setOptimalTime] = React.useState(null);
    const [isNearCurrentTime, setIsNearCurrentTime] = React.useState(false);
const [frequency, setFrequency] = React.useState(0);
  
    React.useEffect(() => {
        const fetchOptimalTime = async () => {
            try {
                const response = await fetch(`https://api.kraken.com/0/public/OHLC?pair=${pair}USD&interval=5`);
                const data = await response.json();
                if (!data.result || !data.result[`${pair}USD`]) {
                    console.error('No data found for', pair);
                    return;
                }
  
                const candles = data.result[`${pair}USD`];
                const hourlyMoves = new Array(24).fill(0);  // Count of significant moves per hour
  
                // Track the last 7 days
                const sevenDaysAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
  
                for (let i = 1; i < candles.length; i++) {
                    const currentCandle = candles[i];
                    const timestamp = currentCandle[0] * 1000;
                    
                    // Only consider last 7 days
                    if (timestamp < sevenDaysAgo) continue;
  
                    const date = new Date(timestamp);
                    const hour = (date.getUTCHours() - 8 + 24) % 24;  // Convert UTC to PST
  
                    const open = parseFloat(currentCandle[1]);
                    const close = parseFloat(currentCandle[4]);
                    const gain = ((close - open) / open) * 100;
                    
// Count significant upward moves (>0.5%)
                    if (gain > 0.5) {
                        hourlyMoves[hour]++;
                    }
                }
  
                // Find hour with most frequent significant moves
                let bestHour = 0;
                let maxMoves = 0;
                
                hourlyMoves.forEach((moves, hour) => {
                    if (moves > maxMoves) {
                        maxMoves = moves;
                        bestHour = hour;
                    }
                });
  
                const formattedHour = bestHour % 12 || 12;
                const period = bestHour < 12 ? 'AM' : 'PM';
                const timeString = `${formattedHour}:00 ${period}`;
  
// Check if current time is near optimal time
                const now = new Date();
                const currentHourPST = (now.getUTCHours() - 8 + 24) % 24;
                const hourDiff = Math.abs(currentHourPST - bestHour);
                const isNear = hourDiff <= 1 || hourDiff >= 23;
  
                setOptimalTime(timeString);
                setIsNearCurrentTime(isNear);
setFrequency(maxMoves);
  
            } catch (error) {
                console.error('Error fetching optimal time:', error);
                setOptimalTime('N/A');
            }
        };
  
        fetchOptimalTime();
    }, [pair]);
  
    return React.createElement('span', {
        style: {
            color: isNearCurrentTime ? '#22c55e' : 'inherit',
            fontWeight: isNearCurrentTime ? 'bold' : 'normal'
        }
    }, optimalTime ? `${optimalTime}` : 'Loading...');
  };