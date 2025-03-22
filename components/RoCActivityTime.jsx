import React from 'react';

const RoCActivityTime = ({ time, isRecentTime }) => {
    return (
        <div className="flex items-center">
            <span 
                className={`${isRecentTime ? 'text-green-500' : 'text-gray-400'} text-sm`}
            >
                {time || 'N/A'}
            </span>
        </div>
    );
};

export default RoCActivityTime;