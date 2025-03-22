import json
import time
import threading
import pytz
from datetime import datetime
from pathlib import Path
from transaction_history import HISTORY_FILE, load_transaction_history, save_transaction_history

# Define time zone constants
UTC_TZ = pytz.timezone('UTC')
PACIFIC_TZ = pytz.timezone('America/Los_Angeles')

def convert_to_pacific_time(timestamp):
    """
    Convert a UTC timestamp to Pacific Time (Los Angeles)
    
    Args:
        timestamp (float): UTC timestamp in seconds
        
    Returns:
        str: Formatted datetime string in Pacific Time
    """
    if isinstance(timestamp, str):
        try:
            # Try to parse as ISO format first
            dt = datetime.fromisoformat(timestamp)
            # Make it timezone aware if it's not
            if dt.tzinfo is None:
                dt = UTC_TZ.localize(dt)
        except ValueError:
            # If that fails, try as timestamp
            dt = datetime.fromtimestamp(float(timestamp), UTC_TZ)
    else:
        # Handle numeric timestamp
        dt = datetime.fromtimestamp(timestamp, UTC_TZ)
    
    # Convert to Pacific time
    pacific_dt = dt.astimezone(PACIFIC_TZ)
    return pacific_dt.isoformat()

class TransactionRecorder:
    """
    Enhanced transaction recorder that uses the existing transaction_history.py
    with Pacific Time zone support
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.pending_file = HISTORY_FILE + ".pending"
        self._ensure_pending_file()

    def _ensure_pending_file(self):
        """Ensure pending file exists"""
        if not Path(self.pending_file).exists():
            with open(self.pending_file, 'w') as f:
                json.dump([], f)

    def _load_pending(self):
        """Load pending transactions"""
        try:
            with open(self.pending_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted pending file, resetting")
            return []

    def _save_pending(self, transactions):
        """Save pending transactions"""
        try:
            with open(self.pending_file, 'w') as f:
                json.dump(transactions, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving pending transactions: {e}")
            return False

    def is_duplicate(self, crypto, percentage, timestamp, window_seconds=300):
        """Check if transaction already exists"""
        with self.lock:
            history = load_transaction_history()
            cutoff_time = timestamp - window_seconds
            
            for tx in history:
                # Convert date string to timestamp, handling potential timezone info
                try:
                    tx_date = datetime.fromisoformat(tx['date'])
                    # If timezone info is present, convert to UTC for comparison
                    if tx_date.tzinfo is not None:
                        tx_date = tx_date.astimezone(UTC_TZ)
                    tx_time = tx_date.timestamp()
                except ValueError:
                    # Fallback to simple parsing if isoformat fails
                    tx_time = datetime.strptime(tx['date'], "%Y-%m-%d %H:%M:%S").timestamp()
                
                if (tx_time > cutoff_time and 
                    tx['crypto'] == crypto and 
                    abs(tx['percentage'] - percentage) < 0.01):
                    return True
            return False

    def record_transaction(self, crypto, percentage, timestamp=None, max_retries=3):
        """
        Record transaction with verification, using Pacific Time
        
        Args:
            crypto (str): Cryptocurrency name
            percentage (float): Gain/loss percentage
            timestamp (float, optional): UTC timestamp in seconds. If None, current time is used.
            max_retries (int): Maximum number of retries
            
        Returns:
            tuple: (success, message)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Convert timestamp to Pacific Time for storage
        pacific_time = convert_to_pacific_time(timestamp)
        
        if self.is_duplicate(crypto, percentage, timestamp):
            return False, "Duplicate transaction detected"

        transaction = {
            'date': pacific_time,  # Store in Pacific Time format
            'crypto': crypto.strip(),
            'percentage': round(float(percentage), 2)
        }

        with self.lock:
            # Save to pending first
            pending = self._load_pending()
            pending.append(transaction)
            if not self._save_pending(pending):
                return False, "Failed to save pending transaction"

            # Try to add to main history
            for attempt in range(max_retries):
                try:
                    history = load_transaction_history()
                    history.append(transaction)
                    save_transaction_history(history)
                    
                    # Clear pending after successful save
                    self._save_pending([])
                    print(f"Successfully recorded transaction: {transaction}")
                    return True, "Transaction recorded successfully"
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        return False, f"Failed to record after {max_retries} attempts: {str(e)}"

            return False, "Failed to record transaction"

    def verify_transaction(self, crypto, percentage, timestamp, tolerance_seconds=60):
        """
        Verify transaction was recorded
        
        Args:
            crypto (str): Cryptocurrency name
            percentage (float): Gain/loss percentage
            timestamp (float): UTC timestamp in seconds
            tolerance_seconds (int): Time tolerance in seconds
            
        Returns:
            bool: True if transaction verified, False otherwise
        """
        with self.lock:
            history = load_transaction_history()
            
            # Convert input timestamp to both UTC and Pacific for comparison
            timestamp_utc = timestamp
            pacific_time = convert_to_pacific_time(timestamp)
            
            for tx in history:
                # Parse transaction date, handling potential timezone info
                try:
                    tx_date = datetime.fromisoformat(tx['date'])
                    # If timezone info is present, convert to UTC for comparison
                    if tx_date.tzinfo is not None:
                        tx_date = tx_date.astimezone(UTC_TZ)
                    tx_time = tx_date.timestamp()
                except ValueError:
                    # Fallback to simple parsing if isoformat fails
                    tx_time = datetime.strptime(tx['date'], "%Y-%m-%d %H:%M:%S").timestamp()
                
                if (abs(tx_time - timestamp_utc) <= tolerance_seconds and
                    tx['crypto'] == crypto and
                    abs(tx['percentage'] - percentage) < 0.01):
                    return True
                    
            # Check pending transactions
            pending = self._load_pending()
            for tx in pending:
                # Parse transaction date, handling potential timezone info
                try:
                    tx_date = datetime.fromisoformat(tx['date'])
                    # If timezone info is present, convert to UTC for comparison
                    if tx_date.tzinfo is not None:
                        tx_date = tx_date.astimezone(UTC_TZ)
                    tx_time = tx_date.timestamp()
                except ValueError:
                    # Fallback to simple parsing if isoformat fails
                    tx_time = datetime.strptime(tx['date'], "%Y-%m-%d %H:%M:%S").timestamp()
                
                if (abs(tx_time - timestamp_utc) <= tolerance_seconds and
                    tx['crypto'] == crypto and
                    abs(tx['percentage'] - percentage) < 0.01):
                    # Found in pending, try to recover
                    return self.record_transaction(crypto, percentage, timestamp_utc)[0]
                    
            return False

    def recover_pending_transactions(self):
        """Recover any stuck transactions"""
        with self.lock:
            pending = self._load_pending()
            if not pending:
                return 0
                
            recovered = 0
            history = load_transaction_history()
            
            for tx in pending:
                # Parse transaction date, handling potential timezone info
                try:
                    tx_date = datetime.fromisoformat(tx['date'])
                    # If timezone info is present, convert to UTC for comparison
                    if tx_date.tzinfo is not None:
                        tx_date = tx_date.astimezone(UTC_TZ)
                    tx_time = tx_date.timestamp()
                except ValueError:
                    # Fallback to simple parsing if isoformat fails
                    tx_time = datetime.strptime(tx['date'], "%Y-%m-%d %H:%M:%S").timestamp()
                
                if not self.is_duplicate(tx['crypto'], tx['percentage'], tx_time):
                    history.append(tx)
                    recovered += 1
                    
            if recovered > 0:
                save_transaction_history(history)
            self._save_pending([])  # Clear pending
            
            return recovered