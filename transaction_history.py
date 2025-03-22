import json
from datetime import datetime
from pathlib import Path

HISTORY_FILE = "transaction_history.json"

def load_transaction_history():
    """Load transaction history from JSON file"""
    try:
        if Path(HISTORY_FILE).exists():
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading transaction history: {e}")
        return []

def save_transaction_history(history):
    """Save transaction history to JSON file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving transaction history: {e}")

def add_transaction(crypto, percentage):
    """Add a new transaction to history"""
    if not crypto or crypto.strip() == "":
        print("Warning: Attempting to save transaction with empty crypto name")
        return None
        
    transaction = {
        'date': datetime.now().isoformat(),
        'crypto': crypto.strip(),  # Remove any whitespace
        'percentage': float(percentage)  # Ensure percentage is a number
    }
    print(f"Adding transaction: {transaction}")  # Debug log
    
    history = load_transaction_history()
    history.append(transaction)
    save_transaction_history(history)
    return transaction