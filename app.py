import os
import re
import sys
import time
import hmac
import base64
import hashlib
import threading
import webbrowser
import subprocess
import urllib.parse
import pytz
from functools import wraps, lru_cache
from datetime import datetime

import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from flask import send_from_directory
from datetime import datetime

from transaction_history import load_transaction_history, add_transaction
from transaction_recorder import TransactionRecorder
from rate_limiter import get_global_rate_limiter, rate_limit_decorator
import logging

import json
import shutil
import os.path
import atexit

UPDATE_CONFIG = {
    "update_url": "http://jeremie.tv/bot",
    "version_file": "version.json",
    "app_files": ["app.py", "index.html", "transaction_history.py", "transaction_recorder.py", "rate_limiter.py", "components/OptimalTime.js"],
    "backup_dir": "backups"
}

last_nonce = int(time.time() * 1000)

# Time zone configuration 
UTC_TZ = pytz.timezone('UTC')
PACIFIC_TZ = pytz.timezone('America/Los_Angeles')

# Configure logging for fee adjustments
fee_logger = logging.getLogger('fee_adjustments')
fee_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('fee_adjustments.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
fee_logger.addHandler(file_handler)

# Kraken fee structure (maker/taker fees in percentage)
KRAKEN_FEES = {
    'default': {
        'maker': 0.16,
        'taker': 0.26
    },
    # Volume-based tiers
    '50k+': {
        'maker': 0.14,
        'taker': 0.24
    },
    '100k+': {
        'maker': 0.12,
        'taker': 0.22
    },
    '250k+': {
        'maker': 0.10,
        'taker': 0.20
    },
    '500k+': {
        'maker': 0.08,
        'taker': 0.18
    },
    '1m+': {
        'maker': 0.06,
        'taker': 0.16
    },
    '2.5m+': {
        'maker': 0.04,
        'taker': 0.14
    },
    '5m+': {
        'maker': 0.02,
        'taker': 0.12
    },
    '10m+': {
        'maker': 0.00,
        'taker': 0.10
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create global rate limiter
limiter = get_global_rate_limiter()
can_request, wait_time = limiter.check_rate_limit(is_private=True)

load_dotenv()  # Load .env

class OrderStatus:
    """Constants for order status tracking"""
    VALIDATING = "validating"
    PLACING_BUY = "placing_buy"
    BUY_OPEN = "buy_open"
    BUY_FILLED = "buy_filled"
    PLACING_SELL = "placing_sell"
    SELL_OPEN = "sell_open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

app = Flask(__name__, template_folder=".")
CORS(app)

# KRAKEN credentials from .env
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

KRAKEN_API_URL = "https://api.kraken.com/0/public/Ticker"
OHLC_API_URL = "https://api.kraken.com/0/public/OHLC"

TICKER_CACHE_DURATION = 30
OHLC_CACHE_DURATION = 60

PAIR_PRICE_PRECISION = {
    "CRVUSD": 3,
    "ENAUSD": 4,
    "ADAUSD": 4,
    "AVAXUSD": 2,
    "FTMUSD": 4,
    "ALGOUSD": 5,
    "APTUSD": 4,
    "TREMPUSD": 4,
    "TIAUSD": 4,
    "ARBUSD": 4,
    "PEPEUSD": 8,
    "LDOUSD": 3,
    "SOLUSD": 2,
    "MOGUSD": 8,
    "BTCUSD": 1,  # Bitcoin
    "ETHUSD": 2,  # Ethereum
    "XRPUSD": 5,  # Ripple
    "DOGEUSD": 7, # Dogecoin
    "MATICUSD": 4, # Polygon
    "LINKUSD": 3, # Chainlink
    "UNIUSD": 3,  # Uniswap
    "AAVEUSD": 2, # Aave
    "ATOMUSD": 3, # Cosmos
    "FILUSD": 3,  # Filecoin
    "DOTUSD": 3,  # Polkadot
    "SANDUSD": 4, # The Sandbox
    "MANAUSD": 4, # Decentraland
    "SNXUSD": 3,  # Synthetix
    "GRAMUSD": 4, # Telegram
    "LTCUSD": 2,  # Litecoin
    "DASHUSD": 2, # Dash
    "ZECUSD": 2,  # Zcash
    "XMRUSD": 2,  # Monero
    "SUIUSD": 4,  # Sui
    "SHIBUSD": 8, # Shiba Inu
    "COMPUSD": 2, # Compound
    "OPUSD": 4,   # Optimism
    "NEARUSD": 3, # NEAR Protocol
    "ICPUSD": 2,  # Internet Computer
    "BLURUSD": 4, # Blur
    "INJUSD": 3,  # Injective
    "PYTHUSUD": 4, # Pyth Network
    "PERAUSD": 4, # Pera
    "JUPUSD": 4,  # Jupiter
    "FWOGUSD": 4, # Farmwars OG
    "XCNUSD": 8,  # Chain
    "TRUMPUSD": 4, # Trump
    "TRBOUSD": 3, # Tribund
    "TAOUSD": 4,  # TAO
    "MSKRUSD": 4, # Mitsukuri
    "FLRUSD": 4,  # Flare
    "MORPHOUSD": 4, # Morpho
    "UMAUSD": 4,  # UMA
    "RGTUSD": 4,  # Rogue Token
    "GRTXUSD": 4, # Gratix
    "RENDERUSD": 3,
    "SPXUSD": 4,
    "TAOUSD": 2,      # TAO needs 2 decimal precision
    "POPCATUSD": 4,   # POPCAT needs 4 decimal precision
    "default": 4
}

PAIR_MIN_VOLUMES = {
    "XCNUSD": 5000,  # XCN requires minimum 5000 units
    "PEPE": 100000,  # PEPE requires larger minimum
    "SHIB": 100000,  # SHIB requires larger minimum
    "default": 0  # Default minimum for other pairs
}

ticker_cache = {"data": None, "timestamp": None}
ohlc_cache = {}

#######################################################################
# In-memory dictionary to track the state of each row's buy→sell flow
# row_id => {
#   "pair": "BTCUSD",
#   "buy_open": True/False,
#   "buy_order_id": "...",
# In-memory dictionary to track the state of each row's buy→sell flow
# row_id => {
#   "pair": "BTCUSD",
#   "buy_open": True/False,
#   "buy_order_id": "...",
#   "buy_price": float,
#   "usd_amount": float,
#   "filled_qty": float,
#   "sell_open": True/False,
#   "sell_order_id": "...",
#   "completed": False,
#   "cancelled": False
# }
#######################################################################
order_states = {}

@app.route('/components/<path:filename>')
def serve_component(filename):
    return send_from_directory('components', filename)

#######################################################################
# REAL Kraken Private Request Helper
#######################################################################
@rate_limit_decorator(is_private=True)
def kraken_private_request(endpoint_path, data):
    """
    Enhanced version of kraken_private_request that includes rate limiting and improved nonce handling
    """
    try:
        # Get remaining requests before making the call
        remaining = limiter.get_remaining_requests(is_private=True)
        logging.info(f"Making private request. Remaining requests: {remaining}")
        
        # Original request code here...
        url = "https://api.kraken.com" + endpoint_path
        
        # Improved nonce generation - ensures monotonic increase with microsecond precision
        global last_nonce
        current_nonce = int(time.time() * 1000000)  # Use microseconds for higher precision
        
        # Ensure the nonce is always increasing
        if current_nonce <= last_nonce:
            current_nonce = last_nonce + 1
            
        last_nonce = current_nonce
        data["nonce"] = str(current_nonce)
        
        encoded_data = urllib.parse.urlencode(data)

        message = (str(current_nonce) + encoded_data).encode()
        sha256 = hashlib.sha256(message).digest()

        secret = base64.b64decode(KRAKEN_API_SECRET)
        base = endpoint_path.encode() + sha256
        signature = hmac.new(secret, base, hashlib.sha512).digest()
        api_sign = base64.b64encode(signature).decode()

        headers = {
            "API-Key": KRAKEN_API_KEY,
            "API-Sign": api_sign,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        resp = requests.post(url, data=encoded_data, headers=headers, timeout=10)
        result = resp.json()
            
        # Log rate limit error if encountered
        if result.get("error") and any("Rate limit" in err for err in result["error"]):
            logging.warning("Rate limit error encountered despite limiter!")
            
        return result
    
    except Exception as e:
        logging.error(f"Error in kraken_private_request: {str(e)}")
        raise

def retry_with_updated_nonce(func):
    """
    Decorator to retry API calls that fail due to nonce errors.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                
                # Check if there was a nonce error
                if isinstance(result, dict) and result.get("error"):
                    nonce_errors = [err for err in result["error"] if "nonce" in err.lower()]
                    if nonce_errors:
                        if attempt < max_retries - 1:
                            # Increment the global nonce substantially and retry
                            global last_nonce
                            last_nonce += 1000
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Log that all retries failed
                            logging.error(f"All nonce retries failed: {nonce_errors}")
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Retry {attempt+1}/{max_retries} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    raise
                    
        return func(*args, **kwargs)  # Final attempt if all retries fail
    return wrapper

# Modified exit handler to handle application restart
def handle_exit():
    if app.exit_code == 42:  # Our custom restart code
        print("Restarting application...")
        # With Batch files on Windows, we can just exit and the batch file will restart Python
        # The pause in the batch file will prevent immediate closure
        sys.exit(0)
    else:
        sys.exit(app.exit_code)

# Add a field to track exit code
app.exit_code = 0

# Register the exit handler
atexit.register(handle_exit)

def get_file_hash(filepath):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {str(e)}")
        return None

def get_local_version():
    """Get the local version information"""
    try:
        version_path = os.path.join(os.path.dirname(__file__), UPDATE_CONFIG["version_file"])
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                return json.load(f)
        return {"version": "1.0.0", "files": {}}
    except Exception as e:
        print(f"Error reading local version: {str(e)}")
        return {"version": "1.0.0", "files": {}}

def create_backup():
    """Create backup of current files"""
    try:
        backup_dir = os.path.join(os.path.dirname(__file__), UPDATE_CONFIG["backup_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
        
        # Create backup directory if it doesn't exist
        os.makedirs(backup_path, exist_ok=True)
        
        # Copy all app files to backup
        for filename in UPDATE_CONFIG["app_files"]:
            source = os.path.join(os.path.dirname(__file__), filename)
            if os.path.exists(source):
                # Create subdirectories if needed
                dest_file = os.path.join(backup_path, filename)
                os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                shutil.copy2(source, dest_file)
        
        # Also backup version file if it exists
        version_file = os.path.join(os.path.dirname(__file__), UPDATE_CONFIG["version_file"])
        if os.path.exists(version_file):
            shutil.copy2(version_file, os.path.join(backup_path, UPDATE_CONFIG["version_file"]))
            
        return True
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        return False

def restart_application():
    """Restart the application using the batch file"""
    try:
        # Kill the current Python process and let the batch file restart it
        print("Application will restart...")
        # This is a graceful shutdown
        os._exit(42)  # Using a specific exit code that could be checked by the batch file
    except Exception as e:
        print(f"Error during restart: {str(e)}")


@app.route("/api/check_updates", methods=["GET"])
def check_updates():
    """Check if updates are available from remote server"""
    print("DEBUG: /api/check_updates route hit!")  # Add debug print
    try:
        # Get local version
        local_version = get_local_version()
        local_ver = local_version.get("version", "1.0.0")
        
        # Get remote version file
        response = requests.get(f"{UPDATE_CONFIG['update_url']}/version.json", timeout=10)
        if not response.ok:
            return jsonify({"error": f"Failed to check for updates: {response.status_code}"}), 500
        
        remote_version = response.json()
        remote_ver = remote_version.get("version", "1.0.0")
        
        # Simple version comparison (assumes semantic versioning x.y.z format)
        local_parts = [int(x) for x in local_ver.split(".")]
        remote_parts = [int(x) for x in remote_ver.split(".")]
        
        update_available = remote_parts > local_parts
        
        return jsonify({
            "updateAvailable": update_available,
            "currentVersion": local_ver,
            "newVersion": remote_ver
        }), 200
        
    except Exception as e:
        print(f"Error checking for updates: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/install_updates", methods=["POST"])
def install_updates():
    """Download and install updates from remote server"""
    try:
        # Create backup first
        if not create_backup():
            return jsonify({"error": "Failed to create backup, update aborted"}), 500
        
        # Get remote version info
        response = requests.get(f"{UPDATE_CONFIG['update_url']}/version.json", timeout=10)
        if not response.ok:
            return jsonify({"error": f"Failed to get version info: {response.status_code}"}), 500
        
        remote_version = response.json()
        version = remote_version.get("version", "1.0.0")
        files_to_update = remote_version.get("files", [])  # List of files to update
        
        # Download and update each file
        updated_files = []
        for filename in files_to_update:
            # Download file
            file_url = f"{UPDATE_CONFIG['update_url']}/{filename}"
            print(f"Downloading {file_url}")
            file_response = requests.get(file_url, timeout=30)
            
            if not file_response.ok:
                print(f"Failed to download {filename}: {file_response.status_code}")
                continue
                
            # Ensure directory exists for the file
            file_path = os.path.join(os.path.dirname(__file__), filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
                
            updated_files.append(filename)
        
        # Update version file
        version_path = os.path.join(os.path.dirname(__file__), UPDATE_CONFIG["version_file"])
        with open(version_path, 'w') as f:
            json.dump(remote_version, f)
        
        # Return success response
        return jsonify({
            "success": True,
            "version": version,
            "updatedFiles": updated_files
        }), 200
        
    except Exception as e:
        print(f"Error installing updates: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

# UPDATE the convert_utc_to_pacific function for modern Python:
def convert_utc_to_pacific(utc_timestamp):
    """
    Convert a UTC timestamp to Pacific Time (Los Angeles)
    
    Args:
        utc_timestamp (float): UTC timestamp in seconds
        
    Returns:
        str: Formatted datetime string in Pacific Time
    """
    try:
        # Create a timezone-aware datetime object directly from timestamp
        from datetime import timezone
        dt_utc = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
        
        # Convert to Pacific Time
        pacific_tz = pytz.timezone('America/Los_Angeles')
        dt_pacific = dt_utc.astimezone(pacific_tz)
        
        # Format the datetime as a string
        return dt_pacific.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"ERROR converting timestamp {utc_timestamp} to Pacific time: {str(e)}")
        return f"ERROR: {str(e)}"

def calculate_trade_gains_with_fees(trades):
    """
    Calculates gain/loss percentages from Kraken trade history with proper fee handling.
    Matches buy and sell orders for the same asset using FIFO method.
    
    Args:
        trades (dict): Dictionary of trades from Kraken API
        
    Returns:
        list: List of completed trades with calculated gain/loss
    """
    # Group trades by asset
    assets = {}
    
    for trade_id, trade in trades.items():
        pair = trade["pair"]
        # Skip non-USD pairs
        if not pair.endswith("USD") and not pair.endswith("ZUSD"):
            continue
            
        # Clean asset name
        asset = pair.replace("USD", "").replace("ZUSD", "")
        if asset.startswith("X"):
            asset = asset[1:]  # Remove X prefix for some assets
            
        if asset not in assets:
            assets[asset] = []
            
        # Add trade to asset list
        assets[asset].append({
            "trade_id": trade_id,
            "time": float(trade["time"]),
            "type": trade["type"],  # buy or sell
            "price": float(trade["price"]),
            "volume": float(trade["vol"]),
            "cost": float(trade["cost"]),  # Total cost in USD
            "fee": float(trade["fee"]),    # Fee in USD
            "ordertype": trade.get("ordertype", "market")  # Get order type if available
        })
    
    # Calculate gain/loss for each asset
    completed_trades = []
    
    for asset, asset_trades in assets.items():
        # Sort trades by time
        asset_trades.sort(key=lambda x: x["time"])
        
        # Track buy orders
        buys = []
        
        for trade in asset_trades:
            if trade["type"] == "buy":
                # Add to buys
                buys.append(trade)
            elif trade["type"] == "sell" and buys:
                remaining_sell_volume = trade["volume"]
                sell_time = trade["time"]
                sell_price = trade["price"]
                sell_fee = trade["fee"]
                sell_ordertype = trade.get("ordertype", "market")
                sell_cost = trade["cost"]  # Total USD value of sell
                
                while remaining_sell_volume > 0.000001 and buys:  # Account for floating point errors
                    # Match with earliest buy
                    buy = buys[0]
                    
                    # Calculate volume to match
                    matched_volume = min(buy["volume"], remaining_sell_volume)
                    matched_ratio = matched_volume / buy["volume"] if buy["volume"] > 0 else 0
                    
                    # Get buy details
                    buy_price = buy["price"]
                    buy_fee = buy["fee"] * matched_ratio  # Proportional fee for matched volume
                    buy_ordertype = buy.get("ordertype", "market")
                    buy_cost = buy["cost"] * matched_ratio  # Proportional cost for matched volume
                    
                    # Determine fee types based on order types
                    buy_fee_type = 'maker' if buy_ordertype in ['limit', 'stop-limit'] else 'taker'
                    sell_fee_type = 'maker' if sell_ordertype in ['limit', 'stop-limit'] else 'taker'
                    
                    # Get fee percentages based on volume tier, default to highest tier
                    buy_fee_pct = KRAKEN_FEES['default'][buy_fee_type]
                    sell_fee_pct = KRAKEN_FEES['default'][sell_fee_type]
                    
                    # Calculate actual USD value of the matched portion
                    matched_sell_cost = sell_cost * (matched_volume / trade["volume"]) if trade["volume"] > 0 else 0
                    matched_sell_fee = sell_fee * (matched_volume / trade["volume"]) if trade["volume"] > 0 else 0
                    
                    # Calculate total fees in USD
                    total_fees_usd = buy_fee + matched_sell_fee
                    
                    # Calculate simple gain before fees
                    simple_gain_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    # Calculate net gain after fees
                    net_buy_cost = buy_cost + buy_fee
                    net_sell_proceeds = matched_sell_cost - matched_sell_fee

                    if net_buy_cost > 0:
                        # Calculate net gain percentage based on the actual USD amounts
                        net_gain_pct = ((net_sell_proceeds - net_buy_cost) / net_buy_cost) * 100
                    else:
                        net_gain_pct = 0
                        
                    # Ensure the percentage is properly rounded
                    net_gain_pct = round(net_gain_pct, 2)
                    print(f"Transaction details for {asset}:")
                    print(f"  Buy: ${buy_cost:.2f} + Fee: ${buy_fee:.2f} = ${net_buy_cost:.2f}")
                    print(f"  Sell: ${matched_sell_cost:.2f} - Fee: ${matched_sell_fee:.2f} = ${net_sell_proceeds:.2f}")
                    print(f"  Net gain: {((net_sell_proceeds - net_buy_cost) / net_buy_cost) * 100:.2f}%")
                    
                    # Convert UTC timestamp to Pacific Time
                    pacific_time = convert_utc_to_pacific(sell_time)
                    
                    # Add to completed trades
                    completed_trades.append({
                        "crypto": asset,
                        "date": pacific_time,  # Use Pacific Time
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "volume": matched_volume,
                        "percentage": round(net_gain_pct, 2),  # Use net gain percentage
                        "simple_percentage": round(simple_gain_pct, 2),  # Original gain without fees
                        "buy_fee_usd": buy_fee,
                        "sell_fee_usd": matched_sell_fee,
                        "total_fees_usd": total_fees_usd,
                        "buy_cost_usd": buy_cost,
                        "sell_proceeds_usd": matched_sell_cost
                    })
                    
                    # Update buy volume or remove if fully matched
                    buy["volume"] -= matched_volume
                    if buy["volume"] <= 0.000001:  # Account for floating point errors
                        buys.pop(0)
                    
                    # Update remaining sell volume
                    remaining_sell_volume -= matched_volume
    
    # Sort completed trades by date, newest first
    completed_trades.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    
    return completed_trades

@rate_limit_decorator(is_private=False)
def get_current_price_from_kraken(pair):
    """
    Enhanced version of get_current_price_from_kraken that includes rate limiting
    """
    try:
        remaining = limiter.get_remaining_requests(is_private=False)
        logging.info(f"Making public request. Remaining requests: {remaining}")
        
        url = "https://api.kraken.com/0/public/Ticker"
        resp = requests.get(url, params={"pair": pair}, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        if "error" in data and data["error"]:
            raise Exception(f"Kraken Ticker error: {data['error']}")

        result_key = pair
        if pair not in data["result"]:
            alternate_keys = [f"X{pair}", f"Z{pair}"]
            for key in alternate_keys:
                if key in data["result"]:
                    result_key = key
                    break
            else:
                raise Exception(f"No ticker data found for {pair}")

        result = data["result"][result_key]
        if not result or "c" not in result:
            raise Exception(f"Invalid ticker data structure for {pair}")

        last_price = float(result["c"][0])
        if last_price <= 0:
            raise Exception(f"Invalid price value: {last_price}")

        return last_price

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching price: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error fetching price: {str(e)}")
        raise

def calculate_stop_loss_from_take_profit(take_profit):
    """
    Calculates the stop-loss value based on a given take-profit value
    Maintains a 1:5.8 risk-reward ratio with minimum stop-loss of 0.2
    
    Args:
        take_profit (float): The take-profit value
        
    Returns:
        dict: A dictionary containing both stop-loss and take-profit values
    """
    # Calculate stop-loss based on the 5.8:1 ratio
    risk_reward_ratio = 5.8
    stop_loss = take_profit / risk_reward_ratio
    
    # Apply minimum stop-loss rule
    if stop_loss < 0.2:
        stop_loss = 0.2
        # Recalculate take-profit to maintain the ratio
        take_profit = stop_loss * risk_reward_ratio
    
    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

def calculate_risk_reward_values(calculated_stop_loss):
    """
    Calculates the adjusted stop-loss and take-profit values maintaining a 1:5.8 risk-reward ratio
    
    Args:
        calculated_stop_loss (float): The initially calculated stop-loss value
        
    Returns:
        dict: A dictionary containing the adjusted stop-loss and take-profit values
    """
    # Enforce minimum stop-loss of 0.2
    adjusted_stop_loss = max(0.2, calculated_stop_loss)
    
    # Calculate take-profit maintaining the 5.8:1 ratio (1.16:0.2)
    risk_reward_ratio = 5.8
    take_profit = adjusted_stop_loss * risk_reward_ratio
    
    return {
        "stop_loss": adjusted_stop_loss,
        "take_profit": take_profit
    }    

def get_buy_price_from_history(pair, volume):
    """Get the original buy price for a given pair and volume from trade history"""
    try:
        # Query recent trades, going back far enough to find the buy
        result = kraken_private_request("/0/private/TradesHistory", {
            "trades": True,
            "start": int(time.time() - 7*24*60*60)  # Last 7 days
        })
        
        if result.get("error"):
            raise Exception(f"Failed to fetch trade history: {result['error']}")
            
        trades = result["result"]["trades"]
        
        # Look for the most recent buy order for this pair with matching volume
        for trade_id, trade in trades.items():
            if (trade["pair"] == pair and 
                trade["type"] == "buy" and 
                abs(float(trade["vol"]) - volume) < 0.0001):  # Compare with small tolerance
                return float(trade["price"])
                
        return None
        
    except Exception as e:
        print(f"Error fetching buy price from history: {str(e)}")
        return None
    
def get_high_value_from_ohlc(ohlc_data, default_high=0.2):
    """
    Extracts the highest percentage change from OHLC data
    
    Args:
        ohlc_data (list): List of OHLC candles
        default_high (float): Default high value if calculation fails
        
    Returns:
        float: The highest percentage change value
    """
    try:
        if not ohlc_data or len(ohlc_data) < 6:
            return default_high
        
        highest_change = 0.0
        for candle in ohlc_data[-24:]:  # Look at last 24 candles (2 hours for 5-min candles)
            try:
                open_price = float(candle[1])
                high_price = float(candle[2])
                if open_price <= 0:
                        continue  # Skip invalid prices
                high_change_percent = ((high_price - open_price) / open_price) * 100
                if high_change_percent > highest_change:
                    highest_change = high_change_percent
            except (IndexError, ValueError, TypeError, ZeroDivisionError):
                # Skip candles with invalid data
                continue
                
            return round(max(highest_change, default_high), 1)  # Return at least the default value
    except Exception as e:
        print(f"Error calculating high value: {str(e)}")
        return default_high  # Return default on any error  # Round to 1 decimal place

def check_price_conditions_for_renewal(current_price, buy_price, last_renewal_price, stop_loss_pct, is_initial_stop_loss):
    """
    # Sell orders only
    Determine if we should renew the stop-loss sell order based on price movement
    """
    # Calculate gains
    total_gain_pct = ((current_price - buy_price) / buy_price) * 100
    gain_since_last_renewal = ((current_price - last_renewal_price) / last_renewal_price) * 100
    
    if is_initial_stop_loss:
        # For initial 0.2% stop loss, renew when we reach 0.2% gain
        return total_gain_pct >= 0.2
    else:
        # For subsequent renewals, check if price increased by half the stop loss percentage
        renewal_threshold = stop_loss_pct / 2
        return gain_since_last_renewal >= renewal_threshold

def record_completed_trade(state, sell_price, buy_price, crypto_name):
    """Enhanced version that uses both systems and handles Pacific Time"""
    if not crypto_name or not isinstance(crypto_name, str):
            print(f"Warning: Invalid crypto name: {crypto_name}")
            return False
        
    # Calculate percentage gain/loss
    percent_gain = ((sell_price - buy_price) / buy_price) * 100
    timestamp = time.time()
    
    # Get fee percentages (default tier)
    buy_fee_pct = KRAKEN_FEES['default']['taker']  # Assume taker fee for market orders
    sell_fee_pct = KRAKEN_FEES['default']['taker']
    
    # Calculate effective prices after fees
    effective_buy_price = buy_price * (1 + (buy_fee_pct / 100))
    effective_sell_price = sell_price * (1 - (sell_fee_pct / 100))
    
    # Calculate gain/loss percentage accounting for fees
    percent_gain_with_fees = ((effective_sell_price - effective_buy_price) / effective_buy_price) * 100
    
    # Initialize recorder
    recorder = TransactionRecorder()
    
    # First attempt to record with fees included
    success, message = recorder.record_transaction(
            crypto=crypto_name.strip(),
            percentage=round(percent_gain_with_fees, 2),
            timestamp=timestamp  # The recorder will convert to Pacific
        )
        
    if not success:
        print(f"Initial recording failed: {message}")
        # Try recovery
        recovered = recorder.recover_pending_transactions()
        if recovered > 0:
            print(f"Recovered {recovered} pending transactions")
            success, message = recorder.record_transaction(
                crypto=crypto_name.strip(),
                percentage=round(percent_gain_with_fees, 2),
                timestamp=timestamp
            )

    # Verify recording
    if success:
        verified = recorder.verify_transaction(
            crypto=crypto_name.strip(),
            percentage=round(percent_gain_with_fees, 2),
            timestamp=timestamp
        )
        if verified:
            print(f"Transaction verified for {crypto_name}: {percent_gain_with_fees:.2f}%")
            state["status"] = "completed"  # Update state after verification
            return True
        else:
            print(f"Warning: Transaction verification failed for {crypto_name}")
            return False
        
    print(f"Failed to record transaction: {message}")
    return False

def rate_limit_decorator(is_private=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get immediate permission
            can_request, wait_time = limiter.check_rate_limit(is_private)
            
            if not can_request:
                # If can't request immediately, queue the request
                return limiter.queue_request(func, is_private, *args, **kwargs)
            
            try:
                # Perform the request
                result = func(*args, **kwargs)
                
                # Additional error checking for Kraken-specific errors
                if isinstance(result, dict) and result.get("error"):
                    # Check for rate limit or other API-specific errors
                    if any("Rate limit" in err for err in result.get("error", [])):
                        logging.warning("Rate limit error detected in response")
                        limiter.handle_rate_limit_error(is_private)
                        # Requeue the request
                        return limiter.queue_request(func, is_private, *args, **kwargs)
                
                return result
            
            except Exception as e:
                # Handle any unexpected errors
                logging.error(f"Error in rate-limited request: {str(e)}")
                
                # Check for rate limit errors
                if "Rate limit" in str(e):
                    limiter.handle_rate_limit_error(is_private)
                    # Requeue the request
                    return limiter.queue_request(func, is_private, *args, **kwargs)
                
                raise
        
        return wrapper
    return decorator
    
def check_private_rate_limit():
    """Check remaining private API calls"""
    can_request, wait_time = limiter.check_rate_limit(is_private=True)
    remaining = limiter.get_remaining_requests(is_private=True)
    return {
        'can_request': can_request,
        'wait_time': wait_time,
        'remaining_requests': remaining
    }

def check_public_rate_limit():
    """Check remaining public API calls"""
    can_request, wait_time = limiter.check_rate_limit(is_private=False)
    remaining = limiter.get_remaining_requests(is_private=False)
    return {
        'can_request': can_request,
        'wait_time': wait_time,
        'remaining_requests': remaining
    }

def get_order_details_from_result(result):
    """
    Safely extracts order details from Kraken API response
    Returns tuple: (price, descr)
    """
    try:
        if isinstance(result, dict) and "result" in result:
            if "descr" in result["result"]:
                # Try to get price from descr first
                price = result["result"]["descr"].get("price")
                if price:
                    return float(price), result["result"]["descr"]
            
            # If no price in descr, check orders array
            if "orders" in result["result"]:
                for order in result["result"]["orders"]:
                    if "price" in order:
                        return float(order["price"]), order
            
            # If still no price, use provided limit price
            if "descr" in result["result"] and "price" in result["result"]["descr"]:
                return float(result["result"]["descr"]["price"]), result["result"]["descr"]
                
    except (KeyError, ValueError, TypeError) as e:
        print(f"DEBUG: Error parsing order details: {str(e)}")
    
    return None, None    
    
def place_trailing_stop_sell(pair, volume, trailing_offset_percent, current_price=None):
    try:
        if current_price is None:
            current_price = get_current_price_from_kraken(pair)
        
        price_precision = PAIR_PRICE_PRECISION.get(pair, PAIR_PRICE_PRECISION["default"])
        
        # Start the trailing stop slightly below current price
        trigger_price = current_price * (1 - trailing_offset_percent/100)
        
        # For trailing stops, Kraken expects:
        # - 'price' as the initial trigger price
        # - 'price2' as the trail amount in absolute terms
        trail_amount = current_price * (trailing_offset_percent/100)
        
        order_data = {
            "ordertype": "trailing-stop",
            "type": "sell",
            "pair": pair,
            "volume": f"{volume:.8f}",
            "price": f"{trigger_price:.{price_precision}f}",  # Initial trigger price
            "price2": f"{trail_amount:.{price_precision}f}"   # Trail amount in absolute terms
        }
        
        print(f"DEBUG: place_trailing_stop_sell - Order data: {order_data}")
        print(f"DEBUG: Current price: {current_price}, Trigger: {trigger_price}, Trail amount: {trail_amount}")
        
        # Validate
        order_data["validate"] = True
        validation = kraken_private_request("/0/private/AddOrder", order_data)
        print(f"DEBUG: Validation response: {validation}")
        
        if validation.get("error"):
            raise ValueError(f"Sell order validation failed: {validation['error']}")
        
        # Place order
        order_data["validate"] = False
        result = kraken_private_request("/0/private/AddOrder", order_data)
        if result.get("error"):
            raise ValueError(f"Failed to place sell order: {result['error']}")
            
        return result["result"]["txid"][0], current_price
        
    except Exception as e:
        print(f"DEBUG: place_trailing_stop_sell - Exception: {str(e)}")
        raise


#######################################################################
# For the EVAL logic
#######################################################################

def eval_highest_roc(crypto_pair):
    params = {"pair": crypto_pair, "interval": 60}
    resp = requests.get(OHLC_API_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data["error"]:
        raise Exception(f"Kraken API error: {data['error']}")

    if crypto_pair not in data["result"]:
        found_key = next(iter(data["result"]), "")
        ohlc_data = data["result"].get(found_key, [])
    else:
        ohlc_data = data["result"][crypto_pair]

    if len(ohlc_data) < 6:
        return 0.2, 0.2  # Minimum values rounded to 0.2 (changed from 0.3)

    highest_rate_of_change = 0.0
    for candle in ohlc_data[-6:]:
        open_price = float(candle[1])
        close_price = float(candle[4])
        roc = abs(close_price - open_price) / open_price
        if roc > highest_rate_of_change:
            highest_rate_of_change = roc

    # Calculate dynamic stop loss based on volatility
    # First convert rate of change to percentage
    roc_percent = highest_rate_of_change * 100
    # Then take 30% of that value for stop loss, minimum 0.2%
    stop_loss_percent = max(0.2, round(roc_percent * 0.3, 1))

    return stop_loss_percent, stop_loss_percent

def adjust_stop_loss_for_fees(stop_loss_percent, fee_tier='default'):
    """
    Adjust the stop-loss percentage to account for fees
    The actual stop-loss will be set lower than the user specified to ensure
    net loss matches the specified percentage after fees.
    
    Args:
        stop_loss_percent (float): User-specified stop-loss percentage
        fee_tier (str): User's fee tier
        
    Returns:
        float: Adjusted stop-loss percentage
    """
    # Get fee rates for the provided tier
    fee_rates = KRAKEN_FEES.get(fee_tier, KRAKEN_FEES['default'])
    
    # Use taker fee rate for stop-loss (usually executed as market orders)
    taker_fee_percent = fee_rates['taker']
    
    # Subtract the fee to get the desired net stop-loss
    adjusted_stop_loss = max(0.1, stop_loss_percent - taker_fee_percent)
    
    print(f"Fee-adjusted stop-loss: Original {stop_loss_percent}% → Adjusted {adjusted_stop_loss:.2f}% (Fee: {taker_fee_percent}%)")
    
    return adjusted_stop_loss

def adjust_take_profit_for_fees(take_profit_percent, fee_tier='default'):
    """
    Adjust the take-profit percentage to account for fees
    The actual take-profit will be set higher than the user specified to ensure
    net gain matches the specified percentage after fees.
    
    Args:
        take_profit_percent (float): User-specified take-profit percentage
        fee_tier (str): User's fee tier
        
    Returns:
        float: Adjusted take-profit percentage
    """
    # Get fee rates for the provided tier
    fee_rates = KRAKEN_FEES.get(fee_tier, KRAKEN_FEES['default'])
    
    # Use taker fee rate for take-profit (usually executed as market orders)
    taker_fee_percent = fee_rates['taker']
    
    # Add the fee to get the desired net take-profit
    adjusted_take_profit = take_profit_percent + taker_fee_percent
    
    print(f"Fee-adjusted take-profit: Original {take_profit_percent}% → Adjusted {adjusted_take_profit:.2f}% (Fee: {taker_fee_percent}%)")
    
    return adjusted_take_profit

@app.route("/api/get_recent_trade", methods=["POST"])
def get_recent_trade():
    try:
        data = request.get_json()
        pair = data.get("pair")
        volume = data.get("volume")
        
        if not pair or volume is None:
            return jsonify({"error": "Missing required parameters"}), 400
            
        # Normalize volume for comparison (handle minor differences due to precision)
        target_volume = float(volume)
        
        # Query recent trades from Kraken
        result = kraken_private_request("/0/private/TradesHistory", {
            "trades": True,
            "start": int(time.time() - 60 * 60)  # Last hour
        })
        
        if result.get("error"):
            raise Exception(f"Failed to fetch trade history: {result['error']}")
            
        trades = result["result"]["trades"]
        
        # Look for the most recent matching trade
        matching_trade = None
        for trade_id, trade in trades.items():
            if trade["pair"] == pair and trade["type"] == "sell":
                trade_volume = float(trade["vol"])
                # Allow for small differences in volume due to precision/rounding
                if abs(trade_volume - target_volume) < 0.001 * target_volume:
                    if matching_trade is None or float(trade["time"]) > float(matching_trade["time"]):
                        matching_trade = trade
        
        if not matching_trade:
            # If no exact match, look for any recent sell for this pair
            for trade_id, trade in trades.items():
                if trade["pair"] == pair and trade["type"] == "sell":
                    if matching_trade is None or float(trade["time"]) > float(matching_trade["time"]):
                        matching_trade = trade
        
        if not matching_trade:
            return jsonify({"error": "No matching trade found"}), 404
            
        # Find the corresponding buy trade
        buy_trade = None
        for trade_id, trade in trades.items():
            if trade["pair"] == pair and trade["type"] == "buy":
                if buy_trade is None or float(trade["time"]) > float(buy_trade["time"]):
                    buy_trade = trade
        
        if not buy_trade:
            return jsonify({"error": "No matching buy trade found"}), 404
            
        # Create mini trades dict to calculate gain
        mini_trades = {
            "buy": buy_trade,
            "sell": matching_trade
        }
        
        # Calculate fees
        buy_price = float(buy_trade["price"])
        sell_price = float(matching_trade["price"])
        buy_fee = float(buy_trade["fee"])
        sell_fee = float(matching_trade["fee"])
        buy_cost = float(buy_trade["cost"])
        sell_cost = float(matching_trade["cost"])
        
        # Get maker/taker fee types
        buy_order_type = buy_trade.get("ordertype", "market")
        sell_order_type = matching_trade.get("ordertype", "market")
        
        buy_fee_type = 'maker' if buy_order_type in ['limit', 'stop-limit'] else 'taker'
        sell_fee_type = 'maker' if sell_order_type in ['limit', 'stop-limit'] else 'taker'
        
        # Get fee percentages
        buy_fee_pct = KRAKEN_FEES['default'][buy_fee_type]
        sell_fee_pct = KRAKEN_FEES['default'][sell_fee_type]
        
        # Calculate net gain/loss percentage
        total_fees_usd = buy_fee + sell_fee
        net_buy_cost = buy_cost + buy_fee
        net_sell_proceeds = sell_cost - sell_fee
        
        # Calculate net percentage using actual costs with fees
        if net_buy_cost > 0:
            net_gain_pct = ((net_sell_proceeds - net_buy_cost) / net_buy_cost) * 100
        else:
            net_gain_pct = 0
        
        # Calculate raw percentage (without fees) for comparison
        raw_gain_pct = ((sell_price - buy_price) / buy_price) * 100
        
        # Create response with detailed fee information
        response = {
            "netPercentage": round(net_gain_pct, 2),  # This includes fees
            "rawPercentage": round(raw_gain_pct, 2),  # This is without fees
            "details": {
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_fee": buy_fee,
                "sell_fee": sell_fee,
                "total_fees": total_fees_usd,
                "buy_cost": buy_cost,
                "sell_cost": sell_cost,
                "net_buy_cost": net_buy_cost,
                "net_sell_proceeds": net_sell_proceeds
            }
        }
        
        # Log the calculation for debugging
        fee_logger.info(f"Trade calculation: Raw: {raw_gain_pct:.2f}%, With fees: {net_gain_pct:.2f}%, Total fees: ${total_fees_usd:.2f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error getting recent trade: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/user_fee_tier", methods=["GET"])
def get_user_fee_tier():
    """
    Get the user's current fee tier based on 30-day trading volume
    
    Returns:
        str: Fee tier name ('default', '50k+', etc.)
    """
    try:
        # Query user's 30-day trading volume from Kraken
        result = kraken_private_request("/0/private/TradeVolume", {})
        if result.get("error"):
            print(f"Failed to get trade volume: {result['error']}")
            # Return as properly formatted JSON string
            return jsonify("default"), 200
            
        # Get the 30-day USD volume
        volume = float(result["result"].get("volume", "0"))
        
        # Determine fee tier based on volume
        tier = 'default'
        if volume >= 10000000:
            tier = '10m+'
        elif volume >= 5000000:
            tier = '5m+'
        elif volume >= 2500000:
            tier = '2.5m+'
        elif volume >= 1000000:
            tier = '1m+'
        elif volume >= 500000:
            tier = '500k+'
        elif volume >= 250000:
            tier = '250k+'
        elif volume >= 100000:
            tier = '100k+'
        elif volume >= 50000:
            tier = '50k+'
        
        # Return as properly formatted JSON string
        return jsonify(tier), 200
        
    except Exception as e:
        print(f"Error getting fee tier: {str(e)}")
        # Return as properly formatted JSON string
        return jsonify("default"), 200

@app.route("/api/check_account", methods=["GET"])
@retry_with_updated_nonce
def check_account():
    """Get account balances and open orders from Kraken"""
    try:
        # Get balances
        balance_result = kraken_private_request("/0/private/Balance", {})
        if balance_result.get("error"):
            return jsonify({"error": f"Failed to fetch balances: {balance_result['error']}"}), 500
            
        balances = []
        for asset, amount in balance_result["result"].items():
            # Skip if amount is 0
            if float(amount) <= 0:
                continue
                
            # Clean up asset name
            clean_name = asset.replace("X", "").replace("Z", "")
            if clean_name == "USD":
                balances.insert(0, {
                    "name": "USD",
                    "quantity": float(amount),
                    "usd_amount": float(amount)
                })
            else:
                # Get current price for non-USD assets
                try:
                    price = get_current_price_from_kraken(f"{clean_name}USD")
                    usd_amount = float(amount) * price
                    # Only add if USD amount is greater than 0.01
                    if usd_amount > 0.01:
                        balances.append({
                            "name": clean_name,
                            "quantity": float(amount),
                            "usd_amount": usd_amount
                        })
                except:
                    # Skip if we can't get price
                    continue

        # Get open orders
        orders_result = kraken_private_request("/0/private/OpenOrders", {})
        if orders_result.get("error"):
            return jsonify({"error": f"Failed to fetch orders: {orders_result['error']}"}), 500
            
        orders = []
        for order_id, order in orders_result["result"].get("open", {}).items():
            descr = order["descr"]
            pair = descr["pair"].replace("XBT", "BTC")
            base = pair.replace("USD", "")
            
            # Calculate USD amount for order
            price = float(order["descr"].get("price", 0)) if "price" in order["descr"] else 0
            volume = float(order["vol"])
            
            if price == 0:  # For market orders or if price isn't available
                try:
                    current_price = get_current_price_from_kraken(pair)
                    usd_amount = current_price * volume
                except:
                    usd_amount = 0  # Fallback if we can't get current price
            else:
                usd_amount = price * volume
            
            orders.append({
                "id": order_id,
                "name": base,
                "order": f"{descr['type'].capitalize()} ({descr['ordertype']})",
                "status": order["status"],
                "quantity": volume,
                "usd_amount": round(usd_amount, 2)  # Round to 2 decimal places
            })

        return jsonify({
            "balances": balances,
            "orders": orders
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cancel_order/<order_id>", methods=["POST"])
def cancel_kraken_order(order_id):
    """Cancel a specific order on Kraken"""
    try:
        result = kraken_private_request("/0/private/CancelOrder", {"txid": order_id})
        if result.get("error"):
            return jsonify({"error": f"Failed to cancel order: {result['error']}"}), 500
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/eval_roc", methods=["GET"])
def api_eval_roc():
    crypto_pair = request.args.get("pair")
    if not crypto_pair:
        return jsonify({"error": "Missing 'pair' parameter"}), 400
        
    # Validate the pair - don't try to process standalone "USD"
    if crypto_pair == "USD":
        return jsonify({
            "high_value": 0.0,
            "take_profit_percent": 0.2,
            "stop_loss_percent": 0.2,
            "min_volume": 0,
            "min_usd": 0
        }), 200  # Return defaults instead of error
        
    try:
        # Get OHLC data for this pair
        params = {"pair": crypto_pair, "interval": 5}  # 5-minute candles
        resp = requests.get(OHLC_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("error") and data["error"]:
            print(f"Kraken API error for {crypto_pair}: {data['error']}")
            # Return defaults instead of error
            return jsonify({
                "high_value": 0.0,
                "take_profit_percent": 0.2,
                "stop_loss_percent": 0.2, 
                "min_volume": 0,
                "min_usd": 0
            }), 200

        # Try different key formats if direct match not found
        ohlc_data = []
        if crypto_pair in data["result"]:
            ohlc_data = data["result"][crypto_pair]
        else:
            # Try common Kraken prefixes
            possible_keys = [
                crypto_pair,
                f"X{crypto_pair}",
                f"Z{crypto_pair}",
                crypto_pair.replace("XBT", "BTC"),
                crypto_pair.replace("BTC", "XBT")
            ]
            
            # Also try keys from the result
            for key in data["result"].keys():
                if crypto_pair in key:
                    possible_keys.append(key)
            
            # Try all possible keys
            for key in possible_keys:
                if key in data["result"]:
                    ohlc_data = data["result"][key]
                    break
            
            # If still no data, use the first key if available
            if not ohlc_data and data["result"]:
                first_key = next(iter(data["result"]))
                ohlc_data = data["result"][first_key]
            
        # Calculate high value from OHLC data
        high_value = get_high_value_from_ohlc(ohlc_data, 0.2)  # Provide default value
        
        # Take-profit is half of the high value
        take_profit_percent = high_value / 2
        
        # Calculate stop-loss based on take-profit to maintain the 1:5.8 ratio
        values = calculate_stop_loss_from_take_profit(take_profit_percent)
        stop_loss_percent = values["stop_loss"]
        
        # Make sure take_profit_percent matches the value in values dict
        take_profit_percent = values["take_profit"]

        # Calculate minimum USD amount if applicable
        min_volume = PAIR_MIN_VOLUMES.get(crypto_pair, PAIR_MIN_VOLUMES["default"])
        min_usd = 0
        if min_volume > 0:
            try:
                # Get current price
                current_price = get_current_price_from_kraken(crypto_pair)
                min_usd = min_volume * current_price
            except Exception as price_error:
                print(f"Error getting price for {crypto_pair}: {price_error}")
                # Don't fail the whole function for this error

        return jsonify({
            "high_value": round(high_value, 1),
            "take_profit_percent": round(take_profit_percent, 1),
            "stop_loss_percent": round(stop_loss_percent, 1),
            "min_volume": min_volume if min_volume > 0 else None,
            "min_usd": round(min_usd, 2) if min_usd > 0 else None
        }), 200
    except Exception as e:
        print(f"Error in eval_roc for {crypto_pair}: {str(e)}")
        # Return sensible defaults instead of error
        return jsonify({
            "high_value": 0.2,
            "take_profit_percent": 1.16,
            "stop_loss_percent": 0.2,
            "min_volume": 0,
            "min_usd": 0
        }), 200 

#######################################################################
# Fetch Real USD Balance from /0/private/Balance
#######################################################################
@retry_with_updated_nonce
def get_kraken_usd_balance():
    """
    Retrieves actual USD balance from Kraken by calling /0/private/Balance.
    We parse 'ZUSD' typically for USD. If it doesn't exist, fallback to 0.
    """
    result = kraken_private_request("/0/private/Balance", {})
    if result.get("error"):
        raise Exception(f"Kraken Balance error: {result['error']}")

    # Typically the 'result' dict has something like {"ZUSD":"123.45", "XXBT":"0.002" ...}
    balances = result.get("result", {})
    usd_str = balances.get("ZUSD", "0.0")  # Some accounts might store 'ZUSD'
    return float(usd_str)

@app.route("/order/limit_sell", methods=["POST"])
def limit_sell():
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        target_percent = data.get("target_percent")
        
        print(f"DEBUG: limit_sell route called with row_id: {row_id}")
        print(f"DEBUG: Current order_states keys: {list(order_states.keys())}")
        
        if not row_id:
            print("DEBUG: No row_id provided")
            return jsonify({"error": "No row_id provided"}), 404

        # Check if the row_id exists in order_states, ignoring case sensitivity
        matching_keys = [k for k in order_states.keys() if k.lower() == row_id.lower()]
        
        if not matching_keys:
            print(f"DEBUG: No matching order found for row_id: {row_id}")
            return jsonify({"error": "Order not found"}), 404

        # Use the first matching key
        row_id = matching_keys[0]
        state = order_states[row_id]

        print(f"DEBUG: Found state for {row_id}: {state}")
        
        if not state.get("filled_volume") or not state.get("buy_fill_price"):
            print("DEBUG: No filled buy order found")
            return jsonify({"error": "No filled buy order found"}), 400

        # Retrieve the previous sell percentage from the last sell_open history entry
        previous_sell_percent = None
        for entry in reversed(state.get("history", [])):
            if entry.get("status") == "sell_open":
                previous_sell_percent = entry.get("stop_loss")
                break

        if previous_sell_percent is None:
            previous_sell_percent = state.get("sell_percent", target_percent)

        # Get current price
        current_price = get_current_price_from_kraken(state["pair"])

        # Cancel existing sell order first if one exists
        if state.get("sell_order_id"):
            cancel_result = kraken_private_request(
                "/0/private/CancelOrder", {"txid": state["sell_order_id"]}
            )
            
            if cancel_result.get("error"):
                if not any("Unknown order" in err for err in cancel_result["error"]):
                    print(f"DEBUG: Failed to cancel existing sell order: {cancel_result['error']}")
                    return jsonify({"error": f"Failed to cancel existing sell order: {cancel_result['error']}"}), 500
            
            # Wait for cancellation to be confirmed
            retries = 5
            while retries > 0:
                time.sleep(5)  # Wait 5 seconds before checking status again
                check_status = kraken_private_request(
                    "/0/private/OpenOrders", {}
                )
                
                # If order no longer appears in open orders, proceed
                if state["sell_order_id"] not in check_status.get("result", {}).get("open", {}):
                    break
                
                retries -= 1

            if retries == 0:
                print("DEBUG: Sell order cancellation was not confirmed")
                return jsonify({"error": "Sell order cancellation was not confirmed"}), 500

        # Get price precision for the pair
        price_precision = PAIR_PRICE_PRECISION.get(state["pair"], PAIR_PRICE_PRECISION["default"])
        
        # Calculate limit price based on buy price and target percentage
        buy_price = state["buy_fill_price"]
        limit_price = buy_price * (1 + target_percent / 100)
        limit_price_str = f"{limit_price:.{price_precision}f}"
        
        # Calculate current price difference from limit price
        current_price_diff = ((current_price - limit_price) / limit_price) * 100
        
        # Place limit sell order
        order_data = {
            "ordertype": "limit",
            "type": "sell",
            "pair": state["pair"],
            "volume": f"{state['filled_volume']:.8f}",
            "price": limit_price_str
        }
        
        # Validate first
        validation = kraken_private_request("/0/private/AddOrder", {"validate": True, **order_data})
        if validation.get("error"):
            print(f"DEBUG: Limit sell validation failed: {validation['error']}")
            return jsonify({"error": f"Limit sell validation failed: {validation['error']}"}), 400
            
        # Place actual order
        result = kraken_private_request("/0/private/AddOrder", order_data)
        if result.get("error"):
            print(f"DEBUG: Failed to place limit sell: {result['error']}")
            return jsonify({"error": f"Failed to place limit sell: {result['error']}"}), 500
            
        txid = result["result"]["txid"][0]
        price_str = result.get("result", {}).get("descr", {}).get("price")
        limit_price = float(price_str) if price_str else None

        current_price = get_current_price_from_kraken(state["pair"])
        price_diff = ((current_price - limit_price) / limit_price) * 100 if limit_price else 0
        
        # Update state
        state["sell_order_id"] = txid
        state["history"].append({
            "timestamp": time.time(),
            "status": "limit_sell_placed",
            "price": limit_price,
            "order_id": txid,
            "target_percent": target_percent,
            "previous_sell_percent": previous_sell_percent,
            "current_price_diff": price_diff
        })
        
        return jsonify({
            "status": "success",
            "order_id": txid,
            "limit_price": f"{limit_price:.{price_precision}f}",
            "sell_status": f"Selling at {previous_sell_percent}% [Current diff: {current_price_diff:.2f}%]",
            "price_diff": price_diff
        }), 200
        
    except Exception as e:
        print(f"Error in limit_sell: {str(e)}")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        return jsonify({"error": str(e)}), 500

# Add to app.py

@app.route("/api/active_orders")
def get_active_orders():
    try:
        # Query all open orders from Kraken
        result = kraken_private_request("/0/private/OpenOrders", {})
        if result.get("error"):
            raise Exception(f"Kraken API error: {result['error']}")
            
        active_orders = []
        open_orders = result.get("result", {}).get("open", {})
        
        for order_id, order_info in open_orders.items():
            descr = order_info.get("descr", {})
            pair = descr.get("pair", "").replace("XBT", "BTC")
            order_type = descr.get("type")  # buy or sell
            price = float(order_info.get("price", 0))
            volume = float(order_info.get("vol", 0))
            
            # Get current price for percentage calculation
            current_price = get_current_price_from_kraken(pair)
            
            # Calculate percentage difference
            price_diff = ((current_price - price) / price) * 100 if price > 0 else 0
            
            active_orders.append({
                "order_id": order_id,
                "pair": pair,
                "type": order_type,
                "price": price,
                "volume": volume,
                "price_diff": price_diff,
                "current_price": current_price
            })
            
        return jsonify({"orders": active_orders}), 200
        
    except Exception as e:
        print(f"Error fetching active orders: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/account_balance", methods=["GET"])
def get_account_balance():
    """
    Now returns the REAL USD balance from Kraken, instead of the old 10000.0 placeholder.
    """
    try:
        usd_balance = get_kraken_usd_balance()
        return jsonify({"balance": usd_balance}), 200
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return jsonify({"error": "Could not fetch balance"}), 500

def check_price_stall(price_history, buy_price, min_gain=0.5, stall_threshold=0.2, stall_minutes=10):
    """
    Check if price has increased by min_gain% but then stalled for stall_minutes.
    
    Args:
        price_history: List of tuples (timestamp, price)
        buy_price: Original purchase price
        min_gain: Minimum gain percentage required before checking for stall (default 0.3%)
        stall_threshold: Maximum price movement during stall period (default 0.2%)
        stall_minutes: How long price must stall before triggering sell (default 10 min)
        
    Returns:
        tuple: (bool: should_sell, float: current_gain_percent)
    """
    if len(price_history) < 2:
        return False, 0.0
        
    current_price = price_history[-1][1]
    current_gain = ((current_price - buy_price) / buy_price) * 100
    
    # First check if we've achieved minimum gain
    if current_gain < min_gain:
        return False, current_gain
        
    # Get prices from last stall_minutes
    cutoff_time = time.time() - (stall_minutes * 60)
    recent_prices = [p[1] for p in price_history if p[0] >= cutoff_time]
    
    if len(recent_prices) < 2:  # Need at least 2 prices to check stall
        return False, current_gain
        
    # Calculate max price movement during stall period
    max_price = max(recent_prices)
    min_price = min(recent_prices)
    price_range_percent = ((max_price - min_price) / min_price) * 100
    
    # If price movement during stall period is less than threshold, trigger sell
    should_sell = price_range_percent <= stall_threshold
    
    return should_sell, current_gain

def execute_take_profit(state, current_price):
    """
    Execute take-profit order with comprehensive logging and error handling
    """
    try:
        print(f"\nDEBUG: Take-profit execution started")
        print(f"DEBUG: Current price: {current_price}")
        print(f"DEBUG: Buy price: {state['buy_fill_price']}")
        
        current_gain = ((current_price - state['buy_fill_price']) / state['buy_fill_price']) * 100
        print(f"DEBUG: Current gain: {current_gain:.2f}%")
        print(f"DEBUG: Take-profit target: {state['take_profit']}%")
        
        # 1. Cancel existing stop-loss first
        if state.get('sell_order_id'):
            print(f"DEBUG: Cancelling stop-loss order: {state['sell_order_id']}")
            cancel_result = kraken_private_request(
                "/0/private/CancelOrder",
                {"txid": state['sell_order_id']}
            )
            
            if cancel_result.get('error'):
                if not any("Unknown order" in err for err in cancel_result['error']):
                    raise ValueError(f"Failed to cancel stop-loss: {cancel_result['error']}")
            
            # Wait briefly for cancellation to process
            time.sleep(1)
        
        # 2. Place market sell order
        print(f"DEBUG: Placing market sell order for {state['filled_volume']} {state['pair']}")
        market_result = kraken_private_request(
            "/0/private/AddOrder",
            {
                "ordertype": "market",
                "type": "sell",
                "pair": state['pair'],
                "volume": str(state['filled_volume'])
            }
        )
        
        if market_result.get('error'):
            raise ValueError(f"Failed to place market sell: {market_result['error']}")
        
        # Get the order ID
        txid = market_result['result']['txid'][0]
        print(f"DEBUG: Market sell order placed: {txid}")
        
        # 3. Monitor order execution
        for _ in range(10):  # Try for up to 10 seconds
            check_result = kraken_private_request("/0/private/QueryOrders", {"txid": txid})
            if not check_result.get('error'):
                order_info = check_result['result'].get(txid)
                if order_info and order_info['status'] == 'closed':
                    final_price = float(order_info['price'])
                    final_gain = ((final_price - state['buy_fill_price']) / state['buy_fill_price']) * 100
                    
                    print(f"DEBUG: Take-profit executed successfully")
                    print(f"DEBUG: Final execution price: {final_price}")
                    print(f"DEBUG: Final gain: {final_gain:.2f}%")
                    
                    # 4. Update state with new recorder
                    crypto_name = state['pair'].replace('USD', '')
                    recorder = TransactionRecorder()
                    
                    success, message = recorder.record_transaction(
                        crypto=crypto_name,
                        percentage=round(final_gain, 2)
                    )
                    
                    if success:
                        state.update({
                            'status': 'completed',
                            'percent_gain': final_gain,
                            'history': state.get('history', []) + [{
                                'timestamp': time.time(),
                                'status': 'take_profit_executed',
                                'price': final_price,
                                'gain': final_gain,
                                'order_id': txid
                            }]
                        })
                    
                        # Verify recording
                        if recorder.verify_transaction(
                            crypto=crypto_name,
                            percentage=round(final_gain, 2),
                            timestamp=time.time()
                        ):
                            print(f"Transaction verified for {crypto_name}: {final_gain:.2f}%")
                            return True, final_gain
                        else:
                            print(f"Warning: Transaction verification failed for {crypto_name}")
                    else:
                        print(f"Failed to record transaction: {message}")
                    
            time.sleep(1)
            
        raise ValueError("Take-profit order not confirmed after 10 seconds")
        
    except Exception as e:
        print(f"ERROR: Take-profit execution failed: {str(e)}")
        return False, None

def check_take_profit_condition(state, current_price):
    """
    Check if take-profit conditions are met with enhanced logging
    """
    try:
        if not state.get('take_profit') or not state.get('buy_fill_price'):
            return False
            
        take_profit = float(state['take_profit'])
        current_gain = ((current_price - state['buy_fill_price']) / state['buy_fill_price']) * 100
        
        print(f"\nDEBUG: Checking take-profit condition")
        print(f"DEBUG: Current price: {current_price}")
        print(f"DEBUG: Buy price: {state['buy_fill_price']}")
        print(f"DEBUG: Current gain: {current_gain:.2f}%")
        print(f"DEBUG: Take-profit target: {take_profit}%")
        
        # Check if we're in the 'monitoring' status (No stop-loss mode)
        if state.get('status') == 'monitoring':
            print(f"DEBUG: In monitoring mode (No stop-loss)")
            if current_gain >= take_profit:
                print(f"DEBUG: Take-profit triggered in monitoring mode: {current_gain:.2f}% >= {take_profit}%")
                return True
            return False
            
        # Regular take-profit check for normal mode
        return current_gain >= take_profit
        
    except Exception as e:
        print(f"ERROR: Take-profit check failed: {str(e)}")
        return False

def update_order_status(state, order_info, kraken_status):
    try:
        current_price = get_current_price_from_kraken(state["pair"])
        current_time = time.time()
        
        # First check take-profit condition
        if (state.get('status') == 'sell_open' and 
            check_take_profit_condition(state, current_price)):
            
            success, final_gain = execute_take_profit(state, current_price)
            if success:
                return state
            
        # Update current price difference for sell orders
        if state.get("status") == "sell_open" and state.get("buy_fill_price"):
            price_diff = ((current_price - state["buy_fill_price"]) / state["buy_fill_price"]) * 100
            state["current_price_diff"] = price_diff
            print(f"DEBUG: Updated price difference: {price_diff:.2f}%")

        # Prevent duplicate processing of completed trades
        if state.get("status") == "completed":
            print(f"DEBUG: Order {state.get('sell_order_id')} is already completed. Skipping duplicate update.")
            return state

        # Update current price and calculate price difference
        state["current_price"] = current_price
        
        # Calculate price difference for both buy and sell orders
        if state.get("status") == "buy_open" and state.get("buy_price"):
            price_diff_pct = ((current_price - state["buy_price"]) / state["buy_price"]) * 100
            state["buy_price_diff"] = price_diff_pct
        elif state.get("status") == "sell_open" and state.get("buy_fill_price"):
            price_diff_pct = ((current_price - state["buy_fill_price"]) / state["buy_fill_price"]) * 100
            state["current_price_diff"] = price_diff_pct

        if "pair" not in state:
            return state
        
        if "price_history" not in state:
            state["price_history"] = []
        
        # Maintain price history
        state["price_history"].append((current_time, current_price))
        cutoff_time = current_time - 3600
        state["price_history"] = [(t, p) for t, p in state["price_history"] if t >= cutoff_time]

        # Handle completed orders
        if kraken_status == "closed":
            vol_exec = float(order_info.get("vol_exec", 0))
            if vol_exec > 0:
                fill_price = float(order_info["price"])

                # Stop-loss order was executed
                if "stopprice" in order_info and float(order_info["stopprice"]) > 0:
                    print(f"DEBUG: Stop-loss order {state['sell_order_id']} executed at {fill_price}")

                    buy_price = state["buy_fill_price"]
                    percent_gain = ((fill_price - buy_price) / buy_price) * 100

                    # Record transaction and mark trade as completed
                    crypto_name = state["pair"].replace("USD", "")
                    if record_completed_trade(state, fill_price, buy_price, crypto_name):
                        print(f"Trade completed: {crypto_name} sold at {fill_price}, Gain/Loss: {((fill_price - buy_price) / buy_price) * 100:.2f}%")

                    # Prevent reattempting a stop-loss
                    state["sell_order_id"] = None
                    state["status"] = "completed"
                    state["percent_gain"] = round(percent_gain, 2)

                    state["history"].append({
                        "timestamp": current_time,
                        "status": "stop_loss_completed",
                        "sell_price": fill_price,
                        "percent_gain": percent_gain
                    })

                    print(f"Trade completed: {crypto_name} sold at {fill_price}, Gain/Loss: {percent_gain:.2f}%")
                    return state  # Exit early to prevent placing another stop-loss order

                # Normal Buy Order Fulfillment
                state["status"] = "buy_filled"
                state["buy_fill_price"] = fill_price
                state["filled_volume"] = vol_exec
                state["history"].append({
                    "timestamp": current_time,
                    "status": "buy_filled",
                    "price": fill_price,
                    "volume": vol_exec
                })

                # Place Stop-Loss Sell Order
                try:
                    sell_txid = place_stop_loss_sell(
                        state["pair"], vol_exec, state["sell_percent"], fill_price
                    )
                    state["status"] = "sell_open"
                    state["sell_order_id"] = sell_txid
                    state["current_price_diff"] = 0  # Initialize price difference
                    state["last_renewal_price"] = fill_price  # Initialize last renewal price
                    state["history"].append({
                        "timestamp": current_time,
                        "status": "sell_open",
                        "order_id": sell_txid,
                        "stop_loss": state["sell_percent"]
                    })
                except Exception as e:
                    print(f"Error placing stop loss: {str(e)}")
                    state["status"] = "error"
                    state["error"] = f"Failed to place stop loss: {str(e)}"

        # Handle open orders (Trailing Stop Logic)
        if kraken_status == "open" and state.get("status") == "sell_open" and "buy_fill_price" in state:
            # Always update current price difference for sell orders
            if state.get("buy_fill_price"):
                price_diff = ((current_price - state["buy_fill_price"]) / state["buy_fill_price"]) * 100
                state["current_price_diff"] = price_diff
                print(f"DEBUG: Updated price difference in status check: {price_diff:.2f}%")

            # Get last renewal price or use buy price if not set
            last_renewal_price = state.get("last_renewal_price", state["buy_fill_price"])
            price_increase = ((current_price - last_renewal_price) / last_renewal_price) * 100
            
            # Check if price has increased enough to warrant stop-loss renewal
            if price_increase >= (state["sell_percent"] / 2):
                print(f"DEBUG: Price increased by {price_increase:.2f}% since last renewal, updating stop-loss")
                
                try:
                    # Cancel existing stop-loss
                    cancel_result = kraken_private_request(
                        "/0/private/CancelOrder",
                        {"txid": state["sell_order_id"]}
                    )

                    if cancel_result.get("error"):
                        raise ValueError(f"Failed to cancel order: {cancel_result['error']}")

                    # Wait for cancellation confirmation
                    retries = 5
                    while retries > 0:
                        time.sleep(1)
                        check_status = kraken_private_request("/0/private/OpenOrders", {})
                        if state["sell_order_id"] not in check_status.get("result", {}).get("open", {}):
                            break
                        retries -= 1
                        
                    if retries == 0:
                        raise ValueError("Cancellation not confirmed after retries")

                    # Place new stop-loss at higher price
                    new_txid = place_stop_loss_sell(
                        state["pair"],
                        float(order_info["vol"]),
                        state["sell_percent"],  # Use original stop-loss percentage
                        current_price
                    )

                    state["sell_order_id"] = new_txid
                    state["last_renewal_price"] = current_price
                    state["current_price_diff"] = ((current_price - state["buy_fill_price"]) / state["buy_fill_price"]) * 100
                    state["history"].append({
                        "timestamp": current_time,
                        "status": "stop_loss_renewed",
                        "price": current_price,
                        "price_increase": price_increase,
                        "stop_loss": state["sell_percent"]
                    })
                    print(f"DEBUG: Renewed stop-loss at {state['sell_percent']}% based on price {current_price}")

                except Exception as e:
                    print(f"DEBUG: Error renewing stop loss: {str(e)}")

            # Check for stall conditions
            should_sell, current_gain = check_price_stall(
                state["price_history"], state["buy_fill_price"], min_gain=0.7, stall_threshold=0.3, stall_minutes=10
            )

            if should_sell:
                try:
                    cancel_result = kraken_private_request(
                        "/0/private/CancelOrder", {"txid": state["sell_order_id"]}
                    )

                    if not cancel_result.get("error"):
                        # Place market sell order
                        market_result = kraken_private_request(
                            "/0/private/AddOrder",
                            {
                                "ordertype": "market",
                                "type": "sell",
                                "pair": state["pair"],
                                "volume": str(state["filled_volume"])
                            }
                        )

                        if not market_result.get("error"):
                            state["status"] = "completed"
                            state["history"].append({
                                "timestamp": time.time(),
                                "status": "market_sell_stall",
                                "gain": current_gain
                            })
                except Exception as e:
                    print(f"DEBUG: Error executing stall-triggered sell: {str(e)}")

        return state

    except Exception as e:
        print(f"Error in update_order_status: {str(e)}")
        return state
    
#######################################################################
# Place trailing stop BUY order (Real)
#######################################################################
def get_pair_precision(pair, error_message=None):
    """
    Extract precision information from error message or use predefined values.
    """
    # First check if we have a predefined precision
    if pair in PAIR_PRICE_PRECISION:
        return PAIR_PRICE_PRECISION[pair]
        
    # Try to extract precision from error message
    if error_message and "decimals" in error_message.lower():
        try:
            # Extract number from "can only be specified up to X decimals"
            import re
            match = re.search(r"up to (\d+) decimals", error_message.lower())
            if match:
                precision = int(match.group(1))
                # Cache this for future use
                PAIR_PRICE_PRECISION[pair] = precision
                return precision
        except Exception:
            pass
            
    return 4  # Default to 4 if we couldn't determine

@app.route("/api/fee_adjusted_percentages", methods=["POST"])
def calculate_fee_adjusted_percentages():
    """
    Calculate fee-adjusted stop-loss and take-profit percentages
    
    Expected request body:
    {
        "stop_loss": 1.0,  // Original stop-loss percentage (optional)
        "take_profit": 2.0 // Original take-profit percentage (optional)
    }
    
    Returns:
    {
        "original_stop_loss": 1.0,
        "adjusted_stop_loss": 0.84,
        "original_take_profit": 2.0,
        "adjusted_take_profit": 2.16,
        "fee_tier": "default",
        "maker_fee": 0.16,
        "taker_fee": 0.26
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        stop_loss = data.get("stop_loss")
        take_profit = data.get("take_profit")
        
        # Get user's fee tier
        fee_tier = "default"
        fee_rates = KRAKEN_FEES.get(fee_tier, KRAKEN_FEES['default'])
        
        result = {
            "fee_tier": fee_tier,
            "maker_fee": fee_rates['maker'],
            "taker_fee": fee_rates['taker'],
        }
        
        # Calculate fee-adjusted percentages if provided
        if stop_loss is not None and stop_loss > 0:
            adjusted_stop_loss = adjust_stop_loss_for_fees(float(stop_loss), fee_tier)
            result.update({
                "original_stop_loss": float(stop_loss),
                "adjusted_stop_loss": adjusted_stop_loss
            })
            
        if take_profit is not None and take_profit > 0:
            adjusted_take_profit = adjust_take_profit_for_fees(float(take_profit), fee_tier)
            result.update({
                "original_take_profit": float(take_profit),
                "adjusted_take_profit": adjusted_take_profit
            })
            
        fee_logger.info(f"Fee adjustments: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        fee_logger.error(f"Error calculating fee-adjusted percentages: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/calculate_net_percentage", methods=["POST"])
def calculate_net_percentage():
    """
    Calculate the actual net gain/loss percentage after fees
    
    Expected request body:
    {
        "buy_price": 50000.0,   // Price at which asset was bought
        "sell_price": 51000.0,  // Price at which asset was sold
        "buy_order_type": "limit",  // Type of buy order (limit or market)
        "sell_order_type": "market" // Type of sell order (limit or market)
    }
    
    Returns:
    {
        "gross_percentage": 2.0,    // Percentage without considering fees
        "net_percentage": 1.68,     // Percentage after accounting for fees
        "fee_tier": "default",
        "buy_fee_percent": 0.16,
        "sell_fee_percent": 0.26
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Get required parameters
        buy_price = float(data.get("buy_price", 0))
        sell_price = float(data.get("sell_price", 0))
        buy_order_type = data.get("buy_order_type", "market")  # Default to market
        sell_order_type = data.get("sell_order_type", "market")  # Default to market
        
        if buy_price <= 0 or sell_price <= 0:
            return jsonify({"error": "Invalid prices provided"}), 400
            
        # Get user's fee tier
        fee_tier = "default"
        fee_rates = KRAKEN_FEES.get(fee_tier, KRAKEN_FEES['default'])
        
        # Determine fee percentages based on order types
        buy_fee_percent = fee_rates['maker'] if buy_order_type == "limit" else fee_rates['taker']
        sell_fee_percent = fee_rates['maker'] if sell_order_type == "limit" else fee_rates['taker']
        
        # Calculate gross percentage gain/loss
        gross_percentage = ((sell_price - buy_price) / buy_price) * 100
        
        # Calculate effective prices after fees
        effective_buy_price = buy_price * (1 + buy_fee_percent / 100)
        effective_sell_price = sell_price * (1 - sell_fee_percent / 100)
        
        # Calculate net percentage gain/loss
        net_percentage = ((effective_sell_price - effective_buy_price) / effective_buy_price) * 100
        
        result = {
            "gross_percentage": round(gross_percentage, 2),
            "net_percentage": round(net_percentage, 2),
            "fee_tier": fee_tier,
            "buy_fee_percent": buy_fee_percent,
            "sell_fee_percent": sell_fee_percent
        }
        
        fee_logger.info(f"Net percentage calculation: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        fee_logger.error(f"Error calculating net percentage: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/average_fees", methods=["GET"])
def get_average_fee_impact():
    """
    Calculate the average fee impact from recent transactions
    
    Expected query parameters:
    days: Number of days to analyze (default: 30)
    
    Returns:
    {
        "average_fee_impact_percent": 0.32,     // Average fee impact as percentage
        "total_fee_amount": 123.45,             // Total fees paid (in USD)
        "total_trade_volume": 12345.67,         // Total trade volume (in USD)
        "trade_count": 25                       // Number of trades analyzed
    }
    """
    try:
        # Get days parameter
        days = request.args.get("days", "30")
        try:
            days = int(days)
        except ValueError:
            days = 30
            
        # Fetch recent trades
        start_time = int(time.time() - days * 24 * 60 * 60)
        trades = get_kraken_trading_history(start_time)
        
        if not trades:
            return jsonify({
                "average_fee_impact_percent": 0,
                "total_fee_amount": 0,
                "total_trade_volume": 0,
                "trade_count": 0
            }), 200
            
        # Calculate totals
        total_fee_amount = 0
        total_trade_volume = 0
        trade_count = 0
        
        for trade_id, trade in trades.items():
            # Only consider USD trades
            pair = trade["pair"]
            if not pair.endswith("USD") and not pair.endswith("ZUSD"):
                continue
                
            fee = float(trade["fee"])
            cost = float(trade["cost"])
            
            total_fee_amount += fee
            total_trade_volume += cost
            trade_count += 1
            
        # Calculate average fee impact percentage
        average_fee_impact_percent = (total_fee_amount / total_trade_volume * 100) if total_trade_volume > 0 else 0
        
        result = {
            "average_fee_impact_percent": round(average_fee_impact_percent, 2),
            "total_fee_amount": round(total_fee_amount, 2),
            "total_trade_volume": round(total_trade_volume, 2),
            "trade_count": trade_count
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        fee_logger.error(f"Error calculating average fee impact: {str(e)}")
        return jsonify({"error": str(e)}), 500

def place_stop_loss_sell(pair, volume, stop_loss_percent, reference_price, max_retries=3):
    """Modified to include fee adjustment logging"""
    fee_logger.info(f"Placing stop-loss for {pair} at {stop_loss_percent}% (reference price: {reference_price})")
    current_precision = PAIR_PRICE_PRECISION.get(pair, 4)
    
    for attempt in range(max_retries):
        try:
            # Use exactly the stop_loss_percent provided, it should already be fee-adjusted
            stop_price = reference_price * (1 - stop_loss_percent/100)
            print(f"DEBUG: Calculated stop price: {stop_price} from reference price: {reference_price}")
            price_str = f"{stop_price:.{current_precision}f}"
            volume_str = f"{float(volume):.8f}"
            
            fee_logger.info(f"Stop price calculated: {stop_price} ({stop_loss_percent}% below {reference_price})")

            order_data = {
                "ordertype": "stop-loss",
                "type": "sell",
                "pair": pair,
                "volume": volume_str,
                "price": price_str
            }
            
            # Validate first
            validation = kraken_private_request("/0/private/AddOrder", {"validate": True, **order_data})
            
            if validation.get("error"):
                error_msg = str(validation["error"][0]) if validation["error"] else ""
                print(f"DEBUG: Validation error: {error_msg}")
                new_precision = extract_precision_from_error(error_msg)
                
                if new_precision is not None:
                    current_precision = new_precision
                    PAIR_PRICE_PRECISION[pair] = new_precision
                    print(f"DEBUG: Updated precision to {new_precision}")
                    continue
                    
                raise ValueError(f"Stop-loss validation failed: {validation['error']}")
            
            # Place actual order
            order_data["validate"] = False
            result = kraken_private_request("/0/private/AddOrder", order_data)
            
            if result.get("error"):
                raise ValueError(f"Failed to place stop-loss order: {result['error']}")
                
            print(f"DEBUG: Successfully placed stop-loss order at {price_str}")
            return result["result"]["txid"][0]
            
        except Exception as e:
            print(f"DEBUG: Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
            
    raise ValueError("Failed to place stop loss after all retries")

# List of assets restricted in US:CA
US_CA_RESTRICTED_ASSETS = {
    "RAY", "PERP", "ATLAS", "POLIS", "STEP", "MEDIA", "COPE", "TULIP", "SLND",
    "PORT", "FIDA", "MER", "CRV", "SNX", "UNI", "AAVE", "YFI", "BAL", "COMP",
    "AUD", "EUR", "GBP", "CHF", "CAD", "JPY",
}

def is_trading_restricted(pair):
    """
    Check if an asset is restricted for trading.
    
    Args:
        pair (str): Trading pair (e.g., 'BTCUSD', 'RAYUSD', 'AUDUSD')
    
    Returns:
        tuple: (bool, str) - (is_restricted, reason)
    """
    # Extract base asset from pair
    base_asset = pair.replace('USD', '')
    
    # Check US:CA restrictions
    if base_asset in US_CA_RESTRICTED_ASSETS:
        if base_asset in ["AUD", "EUR", "GBP", "CHF", "CAD", "JPY"]:
            return True, f"{base_asset} forex trading is restricted in your region (US:CA)"
        return True, f"{base_asset} trading is restricted in your region (US:CA)"
    
    return False, None

# Track recent orders to prevent duplicates
recent_orders = {}
ORDER_EXPIRY = 60  # seconds

def prevent_duplicate_orders(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = request.headers.get('X-Request-ID')
        
        if not request_id:
            return jsonify({"error": "Missing request ID"}), 400

        # Clean up expired orders
        current_time = time.time()
        expired = [k for k, v in recent_orders.items() if current_time - v > ORDER_EXPIRY]
        for k in expired:
            recent_orders.pop(k, None)

        # Check if this is a duplicate request
        if request_id in recent_orders:
            return jsonify({"error": "Duplicate order", "status": "duplicate"}), 400

        # Mark this request as processed
        recent_orders[request_id] = current_time

        return f(*args, **kwargs)
    return decorated_function

def try_place_order(pair, volume, price_precision=None):
    """
    Attempt to place a limit order with proper price precision and volume minimums
    """
    last_error = None
    max_retries = 3
    post_only_retries = 0  # Keep tracking post-only retries
    current_precision = price_precision or PAIR_PRICE_PRECISION.get(pair, 4)

    min_volume = PAIR_MIN_VOLUMES.get(pair, PAIR_MIN_VOLUMES["default"])
    if volume < min_volume:
        volume = min_volume
        print(f"DEBUG: Adjusting volume to meet minimum requirement: {volume}")

    # More granular price factors for maker-only orders
    price_factors = [
        0.9995, 0.9990, 0.9985, 0.9980, 0.9975, 
        0.9970, 0.9965, 0.9960, 0.9955, 0.9950
    ]  # Try more price levels to ensure maker order

    # First try all maker order attempts
    for factor in price_factors:
        try:
            if post_only_retries > 3:  # Keep the post-only retry limit
                break  # Break out to try market order instead of raising error

            current_price = get_current_price_from_kraken(pair)
            limit_price = round(current_price * factor, current_precision)
            limit_price_str = f"{limit_price:.{current_precision}f}"
            volume_str = f"{volume:.8f}"

            # Always use post-only orders
            order_data = {
                "ordertype": "limit",
                "type": "buy",
                "pair": pair,
                "volume": volume_str,
                "price": limit_price_str,
                "oflags": "post"  # Always enforce maker orders
            }

            print(f"DEBUG: Attempting maker order for {pair} at factor {factor} - Price: {limit_price_str}")

            # Validate first
            validation_data = {"validate": True, **order_data}
            validation = kraken_private_request("/0/private/AddOrder", validation_data)

            if validation.get("error"):
                error_msg = str(validation["error"][0]) if validation["error"] else ""
                print(f"DEBUG: Validation error for {pair}: {error_msg}")

                if "Post only" in error_msg:
                    post_only_retries += 1
                    print(f"DEBUG: Post-only validation failed at factor {factor}, trying lower price")
                    continue  # Try next lower price factor

                if "decimals" in error_msg.lower():
                    new_precision = extract_precision_from_error(error_msg)
                    if new_precision is not None:
                        print(f"DEBUG: Updating precision for {pair} from {current_precision} to {new_precision}")
                        current_precision = new_precision
                        PAIR_PRICE_PRECISION[pair] = new_precision
                        continue

                raise ValueError(f"Order validation failed: {validation['error']}")

            # Use the same validated price
            order_data["price"] = validation_data["price"]

            # Place actual order
            order_data["validate"] = False
            result = kraken_private_request("/0/private/AddOrder", order_data)

            if result.get("error"):
                error_msg = str(result["error"][0]) if result["error"] else ""
                print(f"DEBUG: Order error for {pair}: {error_msg}")
                
                if "Post only" in error_msg:
                    post_only_retries += 1
                    print(f"DEBUG: Post-only order failed at factor {factor}, trying lower price")
                    continue  # Try next lower price factor
                    
                raise ValueError(f"Failed to place maker order: {result['error']}")

            # Check immediate order status
            order_id = result["result"]["txid"][0]
            status_result = kraken_private_request("/0/private/QueryOrders", {"txid": order_id})

            if "error" in status_result and status_result["error"]:
                print(f"DEBUG: Error querying order status: {status_result['error']}")

            order_status = status_result.get("result", {}).get(order_id, {}).get("status", "")

            if order_status == "canceled" and "Post only order" in status_result["result"][order_id].get("reason", ""):
                post_only_retries += 1
                print(f"DEBUG: Order {order_id} canceled - Retrying with next factor")
                continue

            print(f"DEBUG: Successfully placed maker order at factor {factor}")
            return result

        except Exception as e:
            last_error = str(e)
            print(f"DEBUG: Error at attempt with factor {factor} for {pair}: {last_error}")
            if factor == price_factors[-1]:  # Don't raise error, try market order
                break

    # If we get here, all maker attempts failed - try market order
    try:
        print(f"DEBUG: All maker orders failed for {pair}, attempting market order")
        
        market_order_data = {
            "ordertype": "market",
            "type": "buy",
            "pair": pair,
            "volume": f"{volume:.8f}"
        }

        # Validate market order first
        validation = kraken_private_request("/0/private/AddOrder", {"validate": True, **market_order_data})
        if validation.get("error"):
            raise ValueError(f"Market order validation failed: {validation['error']}")

        # Place actual market order
        result = kraken_private_request("/0/private/AddOrder", market_order_data)
        if result.get("error"):
            raise ValueError(f"Failed to place market order: {result['error']}")

        print(f"DEBUG: Successfully placed market order for {pair}")
        return result

    except Exception as e:
        last_error = str(e)
        print(f"DEBUG: Market order failed for {pair}: {last_error}")
        raise ValueError(f"Failed to place any order after all attempts. Last error: {last_error}")


def extract_precision_from_error(error_message):
    """
    Extract precision requirement from Kraken error messages
    """
    try:
        if "decimals" in error_message.lower():
            match = re.search(r"up to (\d+) decimals", error_message.lower())
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error extracting precision: {str(e)}")
    return None

@app.route("/order/update_stop_loss", methods=["POST"])
def update_stop_loss():
    """Update an existing stop-loss order to trigger immediately"""
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        order_id = data.get("order_id")
        
        if not row_id or not order_id:
            return jsonify({"error": "Missing row_id or order_id"}), 400
            
        state = order_states.get(row_id)
        if not state:
            return jsonify({"error": "Order state not found"}), 404
            
        # Get current order details first
        try:
            order_info = kraken_private_request("/0/private/QueryOrders", {
                "txid": order_id
            })
            
            if order_info.get("error"):
                return jsonify({"error": f"Failed to query order: {order_info['error']}"}), 500
                
            order = order_info["result"].get(order_id)
            if not order:
                return jsonify({"error": "Order not found"}), 404
        
        # First try block for order info retrieval
        except Exception as e:
            print(f"Error retrieving order info: {str(e)}")
            return jsonify({"error": f"Failed to retrieve order info: {str(e)}"}), 500
        
        # Get current price
        current_price = get_current_price_from_kraken(state["pair"])
        
        # Attempt to modify the existing order
        try:
            modify_result = kraken_private_request("/0/private/EditOrder", {
                "txid": order_id,
                "price": f"{current_price:.{PAIR_PRICE_PRECISION.get(state['pair'], 4)}f}",
                "volume": order["vol"]
            })
            
            if modify_result.get("error"):
                # If direct edit fails, we'll try canceling and creating a new order
                print(f"Direct order edit failed: {modify_result['error']}")
                
                # Cancel existing order
                cancel_result = kraken_private_request("/0/private/CancelOrder", {"txid": order_id})
                if cancel_result.get("error"):
                    return jsonify({"error": f"Failed to cancel order: {cancel_result['error']}"}), 500
                
                # Place new order
                new_result = kraken_private_request("/0/private/AddOrder", {
                    "ordertype": "stop-loss",
                    "type": "sell",
                    "pair": state["pair"],
                    "volume": order["vol"],
                    "price": f"{current_price:.{PAIR_PRICE_PRECISION.get(state['pair'], 4)}f}"
                })
                
                if new_result.get("error"):
                    return jsonify({"error": f"Failed to place new order: {new_result['error']}"}), 500
                
                new_txid = new_result["result"]["txid"][0]
                state["sell_order_id"] = new_txid
            else:
                # Order successfully modified
                new_txid = order_id
        
        # Second try block for order modification or replacement
        except Exception as e:
            print(f"Error modifying/replacing order: {str(e)}")
            return jsonify({"error": f"Failed to update order: {str(e)}"}), 500
        
        # Update state
        state["history"].append({
            "timestamp": time.time(),
            "status": "stop_loss_updated_to_zero",
            "old_order": order_id,
            "new_order": new_txid,
            "price": current_price
        })
        
        return jsonify({
            "status": "success",
            "order_id": new_txid,
            "price": current_price
        }), 200
        
    # Final catch-all for any unexpected errors
    except Exception as e:
        print(f"Unexpected error updating stop-loss: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/order/update_buy_order", methods=["POST"])
def update_buy_order():
    """Update an existing buy order to be a market order"""
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        order_id = data.get("order_id")
        
        if not row_id or not order_id:
            return jsonify({"error": "Missing row_id or order_id"}), 400
            
        state = order_states.get(row_id)
        if not state:
            return jsonify({"error": "Order state not found"}), 404

        # Get initial balance for verification
        pair = state["pair"]
        crypto_symbol = pair.replace("USD", "")
        kraken_symbol = f"X{crypto_symbol}" if crypto_symbol in ["XBT", "XMR", "XRP"] else crypto_symbol
        
        try:
            balance_result = kraken_private_request("/0/private/Balance", {})
            initial_balance = float(balance_result.get("result", {}).get(kraken_symbol, 0))
            print(f"DEBUG: Initial {crypto_symbol} balance: {initial_balance}")
        except Exception as e:
            print(f"DEBUG: Error getting initial balance: {str(e)}")
            initial_balance = 0
        
        # Get current order details
        order_info = kraken_private_request("/0/private/QueryOrders", {
            "txid": order_id
        })
        
        if order_info.get("error"):
            return jsonify({"error": f"Failed to query order: {order_info['error']}"}), 500
        
        if not order_info.get("result", {}).get(order_id):
            return jsonify({"error": "Order not found"}), 404
            
        order = order_info["result"][order_id]
        pair = order["descr"]["pair"]
        volume = order["vol"]
        
        # Cancel existing order first
        result = kraken_private_request("/0/private/CancelOrder", {
            "txid": order_id
        })
        
        if result.get("error"):
            if not any("Unknown order" in err for err in result["error"]):
                return jsonify({"error": f"Failed to cancel order: {result['error']}"}), 500
        
        # Wait for cancellation to be confirmed
        retries = 5
        while retries > 0:
            time.sleep(1)
            check_status = kraken_private_request("/0/private/OpenOrders", {})
            if order_id not in check_status.get("result", {}).get("open", {}):
                break
            retries -= 1
            
        if retries == 0:
            return jsonify({"error": "Cancellation not confirmed"}), 500

        # Place new market buy order
        order_data = {
            "ordertype": "market",
            "type": "buy",
            "pair": pair,
            "volume": volume,
        }
        
        # Place actual order
        new_result = kraken_private_request("/0/private/AddOrder", order_data)
        if new_result.get("error"):
            return jsonify({"error": f"Failed to place market order: {new_result['error']}"}), 500
            
        new_txid = new_result["result"]["txid"][0]
        print(f"DEBUG: Market buy order placed: {new_txid}")

        # Monitor order status with retries
        max_status_checks = 10
        order_filled = False
        final_volume = 0
        final_price = 0

        for check in range(max_status_checks):
            time.sleep(2)  # Wait between checks
            
            # Check order status
            status_result = kraken_private_request("/0/private/QueryOrders", {"txid": new_txid})
            
            if not status_result.get("error") and status_result["result"].get(new_txid):
                order_info = status_result["result"][new_txid]
                
                if order_info["status"] == "closed":
                    vol_exec = float(order_info.get("vol_exec", 0))
                    if vol_exec > 0:
                        final_volume = vol_exec
                        final_price = float(order_info["price"])
                        order_filled = True
                        break

            # Also check balance
            try:
                balance_check = kraken_private_request("/0/private/Balance", {})
                if not balance_check.get("error"):
                    new_balance = float(balance_check["result"].get(kraken_symbol, 0))
                    if new_balance > initial_balance:
                        final_volume = new_balance - initial_balance
                        final_price = float(state.get("initial_price", 0))
                        order_filled = True
                        break
            except Exception as e:
                print(f"DEBUG: Error checking balance: {str(e)}")

        if not order_filled:
            # One final balance check
            try:
                final_check = kraken_private_request("/0/private/Balance", {})
                if not final_check.get("error"):
                    final_balance = float(final_check["result"].get(kraken_symbol, 0))
                    if final_balance > initial_balance:
                        final_volume = final_balance - initial_balance
                        final_price = float(state.get("initial_price", 0))
                        order_filled = True
            except Exception as e:
                print(f"DEBUG: Error in final balance check: {str(e)}")

        if order_filled:
            print(f"DEBUG: Order filled - Volume: {final_volume}, Price: {final_price}")

            # Update state with buy fill information first
            state.update({
                "status": "buy_filled",
                "buy_fill_price": final_price,
                "filled_volume": final_volume,
                "buy_order_id": new_txid,
                "history": state.get("history", []) + [{
                    "timestamp": time.time(),
                    "status": "market_buy_filled",
                    "price": final_price,
                    "volume": final_volume,
                    "old_order": order_id,
                    "new_order": new_txid
                }]
            })

            # Get current price for initial difference calculation
            try:
                current_price = get_current_price_from_kraken(pair)
                initial_price_diff = ((current_price - final_price) / final_price) * 100
            except Exception as e:
                print(f"DEBUG: Error getting initial price difference: {str(e)}")
                initial_price_diff = 0

            # Immediately place stop-loss
            try:
                sell_percent = state.get("sell_percent", 1.0)
                sell_txid = place_stop_loss_sell(pair, final_volume, sell_percent, final_price)
                
                # Update state for sell order immediately
                state.update({
                    "status": "sell_open",
                    "sell_order_id": sell_txid,
                        "current_price_diff": initial_price_diff,  # Set initial price difference
                        "buy_fill_price": final_price,  # Ensure this is set again
                        "last_renewal_price": final_price,  # Add initial renewal price
                    "history": state["history"] + [{
                        "timestamp": time.time(),
                        "status": "sell_open",
                        "order_id": sell_txid,
                            "stop_loss": sell_percent,
                            "initial_price_diff": initial_price_diff
                    }]
                })
                print(f"DEBUG: Initial price difference: {initial_price_diff:.2f}%")
                print(f"DEBUG: Stop-loss placed successfully: {sell_txid}")
                
                return jsonify({
                    "status": "success",
                    "filled": True,
                    "order_id": sell_txid,
                    "price": final_price,
                    "volume": final_volume
                }), 200

            except Exception as e:
                print(f"DEBUG: Error placing stop-loss: {str(e)}")
                return jsonify({"error": f"Buy completed but failed to place stop-loss: {str(e)}"}), 500

        # Skip any cancelled state updates
        return jsonify({
            "status": "success",
            "order_id": new_txid,
            "monitoring": True
        }), 200
        
    except Exception as e:
        print(f"Error updating buy order: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/order/update_take_profit", methods=["POST"])
def update_take_profit():
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        take_profit = data.get("take_profit")

        if not row_id or take_profit is None:
            return jsonify({"error": "Missing required fields"}), 400

        # Ensure order state exists; create default if missing
        if row_id not in order_states:
            order_states[row_id] = {
                "status": "buy_open",
                "pair": None,
                "buy_open": False,
                "usd_amount": 0.0,
                "sell_percent": 0.0,
                "take_profit": None,
                "history": []
            }

        # Preserve existing pair unless a new one is provided
        if order_states[row_id].get("pair") and not data.get("pair"):
            pass  # Do not overwrite existing pair if new pair is missing
        else:
            order_states[row_id]["pair"] = data.get("pair")

        # If trade is completed, reset state to allow a new trade
        if order_states[row_id].get("status") == "completed":
            print(f"DEBUG: Trade for {row_id} completed. Resetting for a new trade.")

            order_states[row_id]["status"] = "buy_open"
            order_states[row_id]["history"] = []
            order_states[row_id]["take_profit"] = take_profit
            return jsonify({"message": "New trade opened successfully"}), 200

        # Update take profit normally for active trades
        order_states[row_id]["take_profit"] = take_profit

        print(f"DEBUG: Updated take_profit for {row_id} => {take_profit}")
        print(f"DEBUG: Current order_states => {order_states}")

        return jsonify({"message": "Take Profit updated successfully"}), 200

    except Exception as e:
        print(f"Error updating take profit: {str(e)}")
        return jsonify({"error": str(e)}), 500


# In the trailing_stop_buy route, modify the order placement logic:

@app.route("/order/trailing_stop_buy", methods=["POST"])
@prevent_duplicate_orders
def trailing_stop_buy():
    try:
        data = request.get_json()
        print("DEBUG: Received buy order request =>", data)

        row_id = data.get("row_id")
        pair = data.get("pair", "").upper()
        usd_amount = float(data.get("usd_amount", 0))
        sell_percent = float(data.get("sell_percent", 0))
        take_profit = data.get("take_profit")
        stop_loss_mode = data.get("stop_loss_mode", "recommended")

        if not row_id or not pair:
            return jsonify({"error": "Invalid input"}), 400

        # Get user's fee tier
        fee_tier = "default"
        
        # Apply fee adjustments here
        original_sell_percent = sell_percent
        original_take_profit = take_profit
        
        # Only adjust if values are valid
        if sell_percent > 0:
            sell_percent = adjust_stop_loss_for_fees(sell_percent, fee_tier)
            
        if take_profit is not None and float(take_profit) > 0:
            take_profit = adjust_take_profit_for_fees(float(take_profit), fee_tier)
            
        print(f"Fee-adjusted values - Stop-loss: {original_sell_percent}% → {sell_percent}%, Take-profit: {original_take_profit}% → {take_profit}%")

        # Ensure state is properly reset if the order was previously cancelled
        if row_id in order_states and order_states[row_id].get("status") == "cancelled":
            print(f"DEBUG: Resetting state for row {row_id} as it was previously cancelled.")
            order_states[row_id] = {}

        # Prevent duplicate buy orders
        if (row_id in order_states and order_states[row_id].get("status") == OrderStatus.BUY_OPEN
            and order_states[row_id].get("pair") and order_states[row_id].get("usd_amount", 0) > 0):
            return jsonify({"error": "Order already exists"}), 400

        # Get current price from Kraken
        current_price = get_current_price_from_kraken(pair)
        if current_price <= 0:
            raise ValueError(f"Invalid current price fetched for {pair}: {current_price}")

        volume = usd_amount / current_price

        # Enforce minimum volume for XCNUSD
        if pair == "XCNUSD" and volume < 1000.0:
            volume = 1000.0

        # Initialize order state
        order_states[row_id] = {
            "status": OrderStatus.PLACING_BUY,
            "pair": pair,
            "buy_open": True,
            "initial_price": current_price,
            "usd_amount": usd_amount,
            "sell_percent": sell_percent,           # Store fee-adjusted value
            "original_sell_percent": original_sell_percent,  # Store original for UI display
            "take_profit": take_profit,             # Store fee-adjusted value
            "original_take_profit": original_take_profit,    # Store original for UI display
            "stop_loss_mode": stop_loss_mode,
            "fee_tier": "default",                   # Store the fee tier used
            "history": []
        }

        print(f"DEBUG: Added row_id to order_states: {row_id}, pair: {pair}")

        # Calculate stop-loss percent dynamically
        stop_loss_percent, _ = eval_highest_roc(pair)
        print(f"DEBUG: Calculated stop loss percent for {pair}: {stop_loss_percent}%")

        # Ensure all parameters are valid
        if not all([row_id, pair, usd_amount > 0, sell_percent > 0]):
            return jsonify({"error": "Missing or invalid parameters"}), 400

        pair = pair.replace("BTC", "XBT")  # Normalize BTC symbol for Kraken

        try:
            order_states[row_id]["status"] = OrderStatus.PLACING_BUY
            order_states[row_id]["buy_open"] = False

            # Try placing the order
            result = try_place_order(pair, volume)
            if result.get("error"):
                raise ValueError(f"Failed to place order: {result['error']}")

            txid = result["result"]["txid"][0]
            
            # Get the order price safely
            buy_price, order_descr = get_order_details_from_result(result)
            if not buy_price:
                buy_price = current_price * 0.9995  # Use default offset if unknown
            
            # Update state with buy order details
            order_states[row_id].update({
                "status": OrderStatus.BUY_OPEN,
                "buy_order_id": txid,
                "buy_price": buy_price,
                "filled_qty": 0.0,
                "sell_open": False,
                "sell_order_id": None,
                "completed": False,
                "cancelled": False,
                "history": order_states[row_id]["history"] + [{
                    "timestamp": time.time(),
                    "status": "limit_buy_placed",
                    "price": buy_price,
                    "order_id": txid,
                    "stop_loss_percent": stop_loss_percent,
                    "take_profit": take_profit,
                    "stop_loss_mode": stop_loss_mode
                }]
            })
            
            print(f"DEBUG: Successfully placed buy order for {pair} at {buy_price}, txid={txid}")
            
            return jsonify({
                "status": "buy_open",
                "details": {
                    "pair": pair,
                    "price": buy_price,
                    "volume": volume,
                    "order_id": txid
                }
            }), 200

        except Exception as e:
            print(f"Error placing limit buy: {e}")
            # If order fails, reset the state properly
            if row_id in order_states:
                order_states[row_id]["status"] = OrderStatus.ERROR
                order_states[row_id]["buy_open"] = False
                order_states[row_id]["buy_order_id"] = None  # Ensure ID is cleared
                return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"Error in trailing_stop_buy: {e}")
        if row_id in order_states:
            order_states[row_id]["status"] = OrderStatus.ERROR
            order_states[row_id]["buy_open"] = False
            order_states[row_id]["buy_order_id"] = None  # Ensure state reset
        return jsonify({"error": str(e)}), 500


#######################################################################
# Cancel whichever order is open (buy or sell) [Still simulation logic]
#######################################################################
@app.route("/api/kraken/order/<order_id>")
def get_kraken_order_status(order_id):
    try:
        result = kraken_private_request("/0/private/QueryOrders", {"txid": order_id})
        if result.get("error"):
            # Check if error indicates order doesn't exist
            error_str = str(result.get("error"))
            if "Unknown order" in error_str:
                return jsonify({
                    "status": "cancelled",
                    "reason": "order_not_found"
                }), 200
            return jsonify({"error": error_str}), 400
            
        order_info = result["result"].get(order_id)
        if not order_info:
            return jsonify({
                "status": "cancelled",
                "reason": "order_not_found"
            }), 200
            
        return jsonify({
            "status": order_info["status"],
            "price": order_info.get("price"),
            "volume": order_info.get("vol"),
            "cost": order_info.get("cost")
        })
    except Exception as e:
        print(f"Error fetching Kraken order status: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Replace the existing transaction history endpoint
# REPLACE the get_transaction_history function with this version:
@app.route("/api/transaction_history", methods=["GET"])
def get_transaction_history():
    """Get Kraken trading history with calculated gain/loss percentages in Pacific Time"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Get start time from query parameters, default to 90 days
            days = request.args.get("days", "90")
            try:
                days = int(days)
            except ValueError:
                days = 90
            
            # Force a fresh fetch for testing - bypass cache
            print("Forcing fresh fetch of transaction history")
            start_time = int(time.time() - days * 24 * 60 * 60)
            trades = get_kraken_trading_history(start_time)
            completed_trades = calculate_trade_gains_with_fees(trades)
            
            # Debug the timestamps for the first few trades
            print(f"DEBUG: Number of completed trades: {len(completed_trades)}")
            for i, trade in enumerate(completed_trades[:3]):
                print(f"DEBUG: Trade {i+1} - Date: {trade['date']}, Crypto: {trade['crypto']}")
                # Add fee information debugging
                print(f"DEBUG: Fee info - Buy fee: ${trade['buy_fee_usd']}, Sell fee: ${trade['sell_fee_usd']}, Total: ${trade['total_fees_usd']}")
            
            # Return all completed trades in the response
            return jsonify(completed_trades), 200
            
        except Exception as e:
            error_str = str(e)
            if ('Invalid nonce' in error_str) and (attempt < max_retries - 1):
                print(f"Nonce error (attempt {attempt+1}): {error_str}. Retrying...")
                time.sleep(retry_delay)  # Add a delay before retrying
                continue
            print(f"Error getting transaction history: {error_str}")
            return jsonify({"error": error_str}), 500   

@app.route("/api/test_specific_timestamp", methods=["GET"])
def test_specific_timestamp():
    """Test a specific timestamp conversion for debugging"""
    try:
        from datetime import datetime
        import pytz
        
        # March 3, 2024 at 19:22:00 UTC (7:22 PM UTC)
        # Should convert to 11:22 AM Pacific on March 3
        test_timestamp = 1709493720
        
        # Convert using our function
        pacific_time = convert_utc_to_pacific(test_timestamp)
        
        # Manual check for verification
        utc_dt = datetime.utcfromtimestamp(test_timestamp)
        utc_aware = pytz.UTC.localize(utc_dt)
        pacific_tz = pytz.timezone('America/Los_Angeles')
        pacific_dt = utc_aware.astimezone(pacific_tz)
        manual_pacific = pacific_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Time that crosses day boundary
        test_timestamp2 = 1709438400  # March 3, 2024 at 04:00:00 UTC
        pacific_time2 = convert_utc_to_pacific(test_timestamp2)
        
        return jsonify({
            "test_timestamp": test_timestamp,
            "utc_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "pacific_time": pacific_time,
            "manual_pacific": manual_pacific,
            "boundary_test_timestamp": test_timestamp2,
            "boundary_pacific_time": pacific_time2
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500 

@app.route("/api/test_timestamp_conversion", methods=["GET"])
def test_timestamp_conversion():
    """Test endpoint to verify timestamp conversion is working correctly"""
    try:
        # Get current time in UTC
        now_utc = time.time()
        
        # Convert to Pacific time
        pacific_time = convert_utc_to_pacific(now_utc)
        
        # Also get UTC time directly for comparison
        utc_datetime = datetime.utcfromtimestamp(now_utc)
        utc_str = utc_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get 24 hours ago for additional testing
        yesterday_utc = now_utc - (24 * 60 * 60)
        yesterday_pacific = convert_utc_to_pacific(yesterday_utc)
        
        return jsonify({
            "utc_timestamp": now_utc,
            "utc_formatted": utc_str,
            "pacific_formatted": pacific_time,
            "yesterday_utc_timestamp": yesterday_utc,
            "yesterday_pacific": yesterday_pacific
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/order/cancel", methods=["POST"])
def cancel_order():
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        preserve_state = True  # Always preserve state

        if not row_id or row_id not in order_states:
            return jsonify({"error": "Order not found"}), 404

        state = order_states[row_id]
        print(f"DEBUG: Cancelling orders for state: {state}")

        pair = state.get("pair")
        if not pair:
            return jsonify({"error": "No trading pair found in state"}), 400

        # Cancel ALL orders for this pair
        cancelled = ensure_all_orders_cancelled(pair)
        if not cancelled:
            return jsonify({"error": "Failed to confirm all orders cancelled"}), 500

        print(f"DEBUG: All orders for {pair} successfully cancelled.")

        # Get current price for final state update
        try:
            current_price = get_current_price_from_kraken(pair)
            if current_price and state.get("buy_fill_price"):
                percent_gain = ((current_price - state["buy_fill_price"]) / state["buy_fill_price"]) * 100
            else:
                percent_gain = None
        except Exception as e:
            print(f"DEBUG: Error getting final price: {str(e)}")
            current_price = None
            percent_gain = None

        # Preserve essential state with additional info
        preserved_state = {
            "status": "cancelled",
            "pair": pair,
            "buy_fill_price": state.get("buy_fill_price"),
            "filled_volume": state.get("filled_volume"),
            "sell_percent": state.get("sell_percent", state.get("sell_percent")),
            "take_profit": state.get("take_profit"),
            "buy_price_diff": state.get("buy_price_diff"),
            "current_price_diff": state.get("current_price_diff"),
            "final_price": current_price,
            "percent_gain": percent_gain,
            "history": state.get("history", []) + [{
                "timestamp": time.time(),
                "status": "all_orders_cancelled",
                "pair": pair,
                "final_price": current_price,
                "percent_gain": percent_gain,
                "reason": "User cancelled"
            }]
        }
        
        # Always preserve the state
        order_states[row_id] = preserved_state
        print(f"DEBUG: Preserved state for row {row_id}: {preserved_state}")

        # Check if we should record a completed trade
        if (state.get("status") == "sell_open" and 
            state.get("buy_fill_price") and 
            current_price and 
            percent_gain is not None):
            try:
                crypto_name = pair.replace("USD", "")
                record_completed_trade(
                    preserved_state,  # Pass full state
                    current_price,    # Final price
                    state["buy_fill_price"],
                    crypto_name
                )
            except Exception as e:
                print(f"DEBUG: Error recording completed trade: {str(e)}")

        return jsonify({
            "status": "cancelled", 
            "preserve_state": True,
            "percent_gain": percent_gain
        }), 200

    except Exception as e:
        print(f"Error in cancel_order: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def ensure_all_orders_cancelled(pair):
    """
    Ensures all orders for a specific crypto pair are cancelled and confirmed cancelled.
    Returns True only when all orders are confirmed to be cancelled or no orders exist.
    
    Args:
        pair (str): Trading pair (e.g., 'BTCUSD')
    Returns:
        bool: True if all orders confirmed cancelled or no orders exist
    """
    max_retries = 20
    retry_delay = 3
    
    try:
        while max_retries > 0:
            # Get all open orders
            result = kraken_private_request("/0/private/OpenOrders", {})
            if result.get("error"):
                print(f"Error checking open orders: {result['error']}")
                time.sleep(retry_delay)
                max_retries -= 1
                continue
                
            # Filter orders for this pair
            open_orders = []
            for order_id, order in result.get("result", {}).get("open", {}).items():
                if order["descr"]["pair"] == pair:
                    open_orders.append(order_id)
            
            # If no open orders found, verify with a second check
            if not open_orders:
                # Double check after a short delay
                time.sleep(2)
                verify_result = kraken_private_request("/0/private/OpenOrders", {})
                if not verify_result.get("error"):
                    verify_orders = [
                        order_id for order_id, order in verify_result.get("result", {}).get("open", {}).items()
                        if order["descr"]["pair"] == pair
                    ]
                    if not verify_orders:
                        print(f"Confirmed no open orders for {pair}")
                        return True
            
            # Cancel any found orders
            for order_id in open_orders:
                cancel_result = kraken_private_request("/0/private/CancelOrder", {"txid": order_id})
                if cancel_result.get("error"):
                    if not any("Unknown order" in err for err in cancel_result["error"]):
                        print(f"Error cancelling order {order_id}: {cancel_result['error']}")
            
            # Wait before checking again
            time.sleep(retry_delay)
            max_retries -= 1
            
        print(f"Failed to confirm all orders cancelled for {pair} after maximum retries")
        return False
        
    except Exception as e:
        print(f"Error in ensure_all_orders_cancelled: {str(e)}")
        return False
    
def get_kraken_trading_history(start_time=None, end_time=None):
    """
    Fetches trading history from Kraken API for a specified time period.
    
    Args:
        start_time (int, optional): Start time as unix timestamp
        end_time (int, optional): End time as unix timestamp
    
    Returns:
        dict: Dictionary containing trades history
    """
    # Default to last 90 days if no time specified
    if not start_time:
        start_time = int(time.time() - 90 * 24 * 60 * 60)
    
    data = {
        "start": start_time
    }
    
    if end_time:
        data["end"] = end_time
    
    # Use existing kraken_private_request function with rate limiting
    result = kraken_private_request("/0/private/TradesHistory", data)
    
    if result.get("error"):
        raise Exception(f"Failed to fetch trade history: {result['error']}")
        
    return result["result"]["trades"]

def calculate_trade_gains(trades):
    """
    Calculates gain/loss percentages from Kraken trade history.
    Matches buy and sell orders for the same asset using FIFO method.
    
    Args:
        trades (dict): Dictionary of trades from Kraken API
        
    Returns:
        list: List of completed trades with calculated gain/loss
    """
    # Group trades by asset
    assets = {}
    
    for trade_id, trade in trades.items():
        pair = trade["pair"]
        # Skip non-USD pairs
        if not pair.endswith("USD") and not pair.endswith("ZUSD"):
            continue
            
        # Clean asset name
        asset = pair.replace("USD", "").replace("ZUSD", "")
        if asset.startswith("X"):
            asset = asset[1:]  # Remove X prefix for some assets
            
        if asset not in assets:
            assets[asset] = []
            
        # Add trade to asset list
        assets[asset].append({
            "trade_id": trade_id,
            "time": float(trade["time"]),
            "type": trade["type"],  # buy or sell
            "price": float(trade["price"]),
            "volume": float(trade["vol"]),
            "cost": float(trade["cost"]),
            "fee": float(trade["fee"])
        })
    
    # Calculate gain/loss for each asset
    completed_trades = []
    
    for asset, asset_trades in assets.items():
        # Sort trades by time
        asset_trades.sort(key=lambda x: x["time"])
        
        # Track buy orders
        buys = []
        
        for trade in asset_trades:
            if trade["type"] == "buy":
                # Add to buys
                buys.append(trade)
            elif trade["type"] == "sell" and buys:
                remaining_sell_volume = trade["volume"]
                sell_time = trade["time"]
                sell_price = trade["price"]
                
                while remaining_sell_volume > 0.000001 and buys:  # Account for floating point errors
                    # Match with earliest buy
                    buy = buys[0]
                    
                    # Calculate volume to match
                    matched_volume = min(buy["volume"], remaining_sell_volume)
                    
                    # Calculate gain/loss percentage
                    buy_price = buy["price"]
                    percent_gain = ((sell_price - buy_price) / buy_price) * 100
                    
                    # Add to completed trades
                    completed_trades.append({
                        "crypto": asset,
                        "date": datetime.fromtimestamp(sell_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "volume": matched_volume,
                        "percentage": round(percent_gain, 2)
                    })
                    
                    # Update buy volume or remove if fully matched
                    buy["volume"] -= matched_volume
                    if buy["volume"] <= 0.000001:  # Account for floating point errors
                        buys.pop(0)
                    
                    # Update remaining sell volume
                    remaining_sell_volume -= matched_volume
    
    # Sort completed trades by date, newest first
    completed_trades.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    
    return completed_trades

# Add caching to reduce API calls
HISTORY_CACHE = {"data": None, "timestamp": None, "duration": 300}  # Cache for 5 minutes

def get_cached_history(days=90):
    """Get cached history or fetch new data if cache expired"""
    # We'll force a refresh of the cache every time for testing
    now = time.time()
    
    # Cache expired, fetch new data
    start_time = int(now - days * 24 * 60 * 60)
    trades = get_kraken_trading_history(start_time)
    completed_trades = calculate_trade_gains_with_fees(trades)
    
    # Debug output
    print(f"DEBUG: Fetched {len(completed_trades)} completed trades")
    for trade in completed_trades[:3]:  # Show first 3 trades
        print(f"DEBUG: Sample trade - Date: {trade['date']}, Crypto: {trade['crypto']}, Net %: {trade['percentage']}")
    
    # Update cache
    HISTORY_CACHE["data"] = completed_trades
    HISTORY_CACHE["timestamp"] = now
    
    return HISTORY_CACHE["data"]

@app.route("/order/market_buy", methods=["POST"])
def market_buy():
    try:
        data = request.get_json()
        row_id = data.get("row_id")

        if not row_id:
            print(f"DEBUG: No row_id provided for market buy")
            return jsonify({"error": "No row_id provided"}), 400

        state = order_states.get(row_id)
        if not state:
            if not data.get("pair"):
                return jsonify({"error": "Missing required parameter: pair"}), 400
            # Instead of returning 404, retrieve state from previous cancellation
            state = {
                "pair": data.get("pair"),
                "usd_amount": data.get("usd_amount", 0),
                "sell_percent": data.get("sell_percent", 0),
                "take_profit": data.get("take_profit"),
                "stop_loss_mode": data.get("stop_loss_mode", "recommended"),
                "status": "buy_open",
                "history": []
            }
            
        pair = state["pair"]
        
        # First ensure all existing orders are cancelled
        if not ensure_all_orders_cancelled(pair):
            return jsonify({"error": "Failed to cancel existing orders"}), 500

        order_states[row_id] = state
        
        # First check balance to see if we already own the crypto
        crypto_symbol = state["pair"].replace("USD", "")
        kraken_symbol = f"X{crypto_symbol}" if crypto_symbol in ["XBT", "XMR", "XRP"] else crypto_symbol
        initial_balance = 0
        
        try:
            balance_result = kraken_private_request("/0/private/Balance", {})
            if not balance_result.get("error"):
                initial_balance = float(balance_result.get("result", {}).get(kraken_symbol, 0))
                print(f"DEBUG: Initial balance for {kraken_symbol}: {initial_balance}")
        except Exception as e:
            print(f"DEBUG: Error checking initial balance: {str(e)}")

        # Check if the buy order was already filled
        if state.get("status") == "buy_filled":
            print(f"DEBUG: Buy order already filled for {row_id}, verifying balance")
            
            # Get account balances to verify the purchase
            balance_result = kraken_private_request("/0/private/Balance", {})
            if balance_result.get("error"):
                raise ValueError(f"Failed to fetch balances: {balance_result['error']}")
                
            balances = balance_result.get("result", {})
            crypto_balance = float(balances.get(kraken_symbol, 0))
            if crypto_balance >= state.get("filled_volume", 0):
                print(f"DEBUG: Verified balance for {crypto_symbol}: {crypto_balance}")
                
                # Place stop-loss if needed
                if not state.get("sell_order_id"):
                    current_price = get_current_price_from_kraken(state["pair"])
                    sell_percent = state.get("sell_percent")
                    try:
                        sell_txid = place_stop_loss_sell(
                            state["pair"],
                            crypto_balance,
                            sell_percent,
                            current_price
                        )
                        
                        state.update({
                            "status": "sell_open",
                            "sell_order_id": sell_txid,
                            "history": state["history"] + [{
                                "timestamp": time.time(),
                                "status": "sell_open",
                                "order_id": sell_txid,
                                "stop_loss": sell_percent
                            }]
                        })
                    except Exception as e:
                        print(f"DEBUG: Error placing stop-loss: {str(e)}")
                
                return jsonify({
                    "status": "already_filled",
                    "message": "Buy order was already filled and balance verified"
                }), 200

        # Get current price for volume calculation
        current_price = get_current_price_from_kraken(pair)
        volume = state["usd_amount"] / current_price
        
        # Place market buy order
        order_data = {
            "ordertype": "market",
            "type": "buy",
            "pair": pair,
            "volume": f"{volume:.8f}"
        }

        # Validate first
        validation = kraken_private_request("/0/private/AddOrder", {"validate": True, **order_data})
        if validation.get("error"):
            print(f"DEBUG: Market buy validation failed: {validation['error']}")
            return jsonify({"error": f"Market buy validation failed: {validation['error']}"}), 400

        # Place actual order
        result = kraken_private_request("/0/private/AddOrder", order_data)
        if result.get("error"):
            print(f"DEBUG: Market buy order placement failed: {result['error']}")
            return jsonify({"error": f"Failed to place market buy: {result['error']}"}), 500

        txid = result["result"]["txid"][0]
        print(f"DEBUG: Market buy order placed with txid: {txid}")

        # Wait briefly for order to execute
        time.sleep(2)

        # Monitor order until filled
        max_retries = 10
        order_filled = False
        vol_exec = 0
        fill_price = 0

        print(f"DEBUG: Monitoring market buy order {txid} for fill status")

        for attempt in range(max_retries):
            try:
                # Check order status
                status_result = kraken_private_request("/0/private/QueryOrders", {"txid": txid})
                
                if status_result.get("error"):
                    print(f"DEBUG: Error checking order status: {status_result['error']}")
                    time.sleep(2)
                    continue
                    
                order_info = status_result["result"].get(txid)
                if not order_info:
                    print("DEBUG: Order info not found")
                    time.sleep(2)
                    continue
                    
                print(f"DEBUG: Order status: {order_info}")
                
                if order_info["status"] == "closed":
                    vol_exec = float(order_info["vol_exec"])
                    if vol_exec > 0:
                        fill_price = float(order_info["price"])
                        order_filled = True
                        print(f"DEBUG: Market buy filled - Price: {fill_price}, Volume: {vol_exec}")
                        break
                        
                elif order_info["status"] == "canceled":
                    print("DEBUG: Order was cancelled")
                    break
                    
                time.sleep(2)
            except Exception as e:
                print(f"DEBUG: Error checking order status: {str(e)}")
                time.sleep(2)

        # If order status unclear, verify through balance
        if not order_filled:
            try:
                print("DEBUG: Verifying order through balance check")
                balance_check = kraken_private_request("/0/private/Balance", {})
                if not balance_check.get("error"):
                    new_balance = float(balance_check["result"].get(kraken_symbol, 0))
                    if new_balance > initial_balance:
                        vol_exec = new_balance - initial_balance
                        fill_price = current_price  # Use market price as approximate fill price
                        order_filled = True
                        print(f"DEBUG: Order confirmed through balance check - Volume: {vol_exec}, Price: {fill_price}")
            except Exception as e:
                print(f"DEBUG: Error in balance verification: {str(e)}")

        if not order_filled:
            print("DEBUG: Market buy order not confirmed as filled")
            return jsonify({"error": "Market order not filled or verified"}), 500

        print(f"DEBUG: Proceeding to place stop-loss with volume: {vol_exec}, price: {fill_price}")

        # Update state with buy fill information
        state.update({
            "status": "buy_filled",
            "buy_fill_price": fill_price,
            "filled_volume": vol_exec,
            "buy_order_id": txid,
            "history": state.get("history", []) + [{
                "timestamp": time.time(),
                "status": "market_buy_filled",
                "price": fill_price,
                "volume": vol_exec,
                "txid": txid
            }]
        })

        # Get current price for initial difference calculation
        try:
            current_price = get_current_price_from_kraken(state["pair"])
            initial_price_diff = ((current_price - fill_price) / fill_price) * 100
        except Exception as e:
            print(f"DEBUG: Error getting initial price difference: {str(e)}")
            initial_price_diff = 0

        # Place stop-loss sell order only if stop_loss_mode is not "none"
        if state.get("stop_loss_mode") != "none":
            sell_txid = None
            stop_loss_retries = 3
            for attempt in range(stop_loss_retries):
                try:
                    print(f"DEBUG: Placing stop-loss sell order for {vol_exec} {state['pair']}")
                    sell_txid = place_stop_loss_sell(
                        state["pair"],
                        vol_exec,
                        state["sell_percent"],
                        fill_price
                    )
                    break
                except Exception as e:
                    print(f"DEBUG: Stop-loss placement failed (attempt {attempt+1}): {str(e)}")
                    if attempt < stop_loss_retries - 1:
                        time.sleep(2)

                if not sell_txid:
                    return jsonify({"error": "Failed to place stop-loss after retries"}), 500

            # Update state for sell order
            state.update({
                "status": "sell_open",
                "sell_order_id": sell_txid,
                "last_renewal_price": fill_price,  # Add initial renewal price
                "current_price_diff": initial_price_diff,  # Set initial price difference
                "history": state["history"] + [{
                    "timestamp": time.time(),
                    "status": "sell_open",
                    "order_id": sell_txid,
                    "stop_loss": state["sell_percent"],
                    "initial_price_diff": initial_price_diff
                }]
            })

            print(f"DEBUG: Stop-loss placed successfully: {sell_txid}")
        else:
            # For "No stop-loss" mode, update state without placing stop-loss
            state.update({
                "status": "monitoring",  # New status for take-profit only
                "current_price_diff": initial_price_diff,
                "history": state["history"] + [{
                    "timestamp": time.time(),
                    "status": "monitoring_resumed",
                    "initial_price_diff": initial_price_diff,
                    "take_profit": state.get("take_profit")
                }]
            })
            print(f"DEBUG: No stop-loss mode active, monitoring for take-profit only: {state.get('take_profit')}%")

        return jsonify({
            "status": "success",
            "order_id": txid,
            "fill_price": fill_price,
            "volume": vol_exec,
            "sell_status": "open" if state.get("sell_order_id") else "monitoring",
            "sell_order_id": state.get("sell_order_id")
        }), 200

    except Exception as e:
        print(f"DEBUG: Unexpected error in market_buy: {str(e)}")
        return jsonify({"error": str(e)}), 500

# In app.py
@app.route("/order/market_sell", methods=["POST"])
def market_sell():
    """
    Cancels any existing stop-loss order (if applicable) and places a new market sell order.
    """
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        volume = data.get("volume")
        
        if not row_id or not volume:
            return jsonify({"error": "Missing row_id or volume"}), 400
            
        if row_id not in order_states:
            return jsonify({"error": "Order not found"}), 404
            
        state = order_states[row_id]
        
        # Allow Market Sell if status is 'cancelled' but volume exists
        if not state.get("filled_volume"):
            return jsonify({"error": "No position to sell"}), 400
            
        pair = state["pair"]
        
        # Ensure all previous orders are cancelled before placing a market sell
        if not ensure_all_orders_cancelled(pair):
            return jsonify({"error": "Failed to cancel existing orders"}), 500
            
        # Use the input field amount from UI
        volume = float(volume)
        
        # Place Market Sell Order with retries
        max_order_retries = 5
        txid = None
        
        for attempt in range(max_order_retries):
            try:
                # Place actual market sell order
                order_data = {
                    "ordertype": "market",
                    "type": "sell",
                    "pair": pair,
                    "volume": f"{volume:.8f}"
                }
                
                result = kraken_private_request("/0/private/AddOrder", order_data)
                if result.get("error"):
                    if attempt == max_order_retries - 1:
                        return jsonify({"error": f"Failed to place market sell: {result['error']}"}), 500
                    time.sleep(3)
                    continue
                    
                txid = result["result"]["txid"][0]
                break
                
            except Exception as e:
                if attempt == max_order_retries - 1:
                    return jsonify({"error": str(e)}), 500
                time.sleep(3)
                
        # Update state with new order info
        state.update({
            "status": "market_sell_executing",
            "sell_order_id": txid,
            "history": state.get("history", []) + [{
                "timestamp": time.time(),
                "status": "market_sell_placed",
                "order_id": txid,
                "volume": volume
            }]
        })
        
        # Monitor order execution to get final price and calculate percentage
        # [Existing monitoring code]
        
        return jsonify({
            "status": "success",
            "order_id": txid,
            "percent_gain": state.get("percent_gain")
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "status": "unexpected_error"
        }), 500

@app.route("/order/status/<row_id>", methods=["GET"])
def get_order_status(row_id):
    try:
        if row_id not in order_states:
            return jsonify({"status": "cancelled", "reason": "order_not_found"}), 200

        state = order_states[row_id]
        current_status = state.get("status", "unknown")
        print(f"DEBUG: Checking status for order {row_id}, current status: {current_status}")

        if isinstance(state.get("fee_tier"), tuple):
            state["fee_tier"] = "default"

        # Return immediately for completed/cancelled orders
        if current_status in ["completed", "cancelled", "error"]:
            # Clean up state after returning final status
            if row_id in order_states:
                del order_states[row_id]
            return jsonify(state), 200

        # Update current price and calculate differences
        try:
            current_price = get_current_price_from_kraken(state["pair"])
            state["current_price"] = current_price

            # Calculate price difference for sell orders
            if current_status == "sell_open" and state.get("buy_fill_price"):
                price_diff = ((current_price - state["buy_fill_price"]) / state["buy_fill_price"]) * 100
                state["current_price_diff"] = price_diff
                print(f"DEBUG: Updated sell price difference: {price_diff:.2f}%")

            # Calculate price difference for buy orders
            elif current_status == "buy_open" and state.get("buy_price"):
                price_diff = ((current_price - state["buy_price"]) / state["buy_price"]) * 100
                state["buy_price_diff"] = price_diff
                print(f"DEBUG: Updated buy price difference: {price_diff:.2f}%")

        except Exception as e:
            print(f"DEBUG: Error updating price differences: {str(e)}")

        # Handle buy_filled status explicitly
        if current_status == "buy_filled":
            # Check if we've already tried and failed to place stop loss
            if state.get("stop_loss_failed"):
                return jsonify({
                    "status": "error",
                    "error": "Failed to place stop loss order", 
                    "details": state.get("stop_loss_error")
                }), 500

            try:
                # Get filled details
                vol_exec = state.get("filled_volume", 0)
                fill_price = state.get("buy_fill_price", 0)
                
                print(f"DEBUG: Placing stop loss for filled buy - Volume: {vol_exec}, Price: {fill_price}")
                
                # Place stop loss order for the filled amount
                target_stop_loss = state.get("sell_percent")
                if not target_stop_loss:
                    raise ValueError("No stop loss percentage set")
                    
                try:
                    sell_txid = place_stop_loss_sell(
                        state["pair"],
                        vol_exec, 
                        target_stop_loss,
                        fill_price
                    )
                    
                    # Update state on success
                    state["status"] = "sell_open"
                    state["sell_order_id"] = sell_txid
                    state["current_stop_loss"] = target_stop_loss
                    state["current_price_diff"] = 0  # Initialize price difference
                    state["last_renewal_price"] = fill_price  # Initialize renewal price
                    state["history"].append({
                        "timestamp": time.time(),
                        "status": "sell_open", 
                        "order_id": sell_txid,
                        "stop_loss": target_stop_loss
                    })
                    
                    return jsonify(state), 200
                    
                except Exception as e:
                    print(f"DEBUG: Error placing stop loss: {str(e)}")
                    state["stop_loss_failed"] = True
                    state["stop_loss_error"] = str(e)
                    state["status"] = "error"
                    state["history"].append({
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e)
                    })
                    return jsonify(state), 200
                    
            except Exception as e:
                print(f"DEBUG: Error handling buy_filled state: {str(e)}")
                return jsonify({
                    "status": "error",
                    "error": f"Failed to handle buy_filled state: {str(e)}"
                }), 500

        # Check active orders with Kraken
        if current_status in ["buy_open", "sell_open"]:
            order_id = state.get("sell_order_id") or state.get("buy_order_id")
            if order_id:
                try:
                    query = kraken_private_request("/0/private/QueryOrders", {"txid": order_id})
                    print(f"DEBUG: Full state during order check: {state}")
                    print(f"DEBUG: Kraken order status response: {query}")
                    
                    if not query.get("error"):
                        order_info = query["result"][order_id]
                        kraken_status = order_info["status"]
                        print(f"DEBUG: Kraken status for {order_id}: {kraken_status}")
                        
                        # Handle renewal of stop loss for sell orders
                        if current_status == "sell_open" and kraken_status == "open":
                            last_renewal_price = state.get("last_renewal_price", state["buy_fill_price"])
                            price_increase = ((current_price - last_renewal_price) / last_renewal_price) * 100
                            
                            if price_increase >= (state["sell_percent"] / 2):
                                print(f"DEBUG: Price increased by {price_increase:.2f}% since last renewal")
                                
                                try:
                                    # Cancel existing stop-loss
                                    cancel_result = kraken_private_request(
                                        "/0/private/CancelOrder",
                                        {"txid": state["sell_order_id"]}
                                    )

                                    if not cancel_result.get("error"):
                                        # Place new stop-loss at higher price
                                        sell_txid = place_stop_loss_sell(
                                            state["pair"],
                                            state["filled_volume"],
                                            state["sell_percent"],
                                            current_price
                                        )
                                        
                                        state["sell_order_id"] = sell_txid
                                        state["last_renewal_price"] = current_price
                                        state["history"].append({
                                            "timestamp": time.time(),
                                            "status": "stop_loss_renewed",
                                            "price": current_price,
                                            "price_increase": price_increase
                                        })
                                        print(f"DEBUG: Renewed stop-loss at {state['sell_percent']}%")
                                
                                except Exception as e:
                                    print(f"DEBUG: Error renewing stop loss: {str(e)}")

                        # Update state based on order status
                        state = update_order_status(state, order_info, kraken_status)
                        return jsonify(state), 200

                except Exception as e:
                    print(f"DEBUG: Exception in order status check: {str(e)}")
                    return jsonify({
                        "status": "error",
                        "error": f"Failed to check order status: {str(e)}"
                    }), 500

            return jsonify(state), 200

        return jsonify(state), 200

    except Exception as e:
        print(f"Error in get_order_status: {str(e)}")
        error_response = {"error": str(e)}
        if state:
            error_response["state"] = state
        return jsonify(error_response), 500

#################################################
#  TICKER / OHLC / UTILS
#################################################
def get_ticker_data():
    response = requests.get(KRAKEN_API_URL, timeout=10)
    data = response.json().get("result", {})
    ticker_cache["data"] = data
    now = time.time()
    ticker_cache["timestamp"] = now
    return data

def get_ohlc_data(pair, interval=5):
    now = time.time()
    key = (pair, interval)
    if key in ohlc_cache:
        cached = ohlc_cache[key]
        if now - cached["timestamp"] < OHLC_CACHE_DURATION:
            return cached["data"]

    response = requests.get(OHLC_API_URL, params={"pair": pair, "interval": interval}, timeout=10)
    if response.status_code != 200:
        print(f"Error fetching OHLC data for {pair}.")
        return []
    ohlc = response.json().get("result", {}).get(pair, [])
    ohlc_cache[key] = {"data": ohlc, "timestamp": now}
    return ohlc

def fetch_kraken_cryptos():
    """
    Fetch tickers from Kraken, but skip:
      - Non-USD pairs
      - pairs containing 'USDT'
      - base assets that start with 'C', 'X', or 'Z', except defined exceptions
      - volume < 500_000
      - price < $0.001
      - restricted assets (from US_CA_RESTRICTED_ASSETS)
    """
    exceptions = {
        "XRP","XTZ","XLM","XMR","XCN","ZETA","ZEC","ZRO","ZEUS","ZK","ZRX",
        "COTI","CRV","CPOOL","COMP","CVC","CELR","CXT","CHZ","CTSI","CHR",
        "CQT","CVX"
    }

    data = get_ticker_data()
    high_volume = {}
    
    for pair, details in data.items():
        # Only pairs that end with "USD" and do not contain "USDT"
        if pair.endswith("USD") and "USDT" not in pair:
            # Derive the base asset by removing "USD"
            base = pair.replace("USD", "")

            if base == "FART":
                continue

            if base == "FARTCOIN":
                continue

            # Skip if asset is restricted
            if base in US_CA_RESTRICTED_ASSETS:
                print(f"DEBUG: Skipping {pair} - restricted in US:CA")
                continue

            # If base starts with C, X, or Z, skip unless in the exceptions set
            if base[0] in ("C", "X", "Z") and base not in exceptions:
                continue

            try:
                # Get current price and check minimum threshold
                price = float(details["c"][0])  # Current price
                if price < 0.001:  # Skip if price is too low
                    print(f"DEBUG: Skipping {pair} - price too low: {price}")
                    continue

                volume_in_base_asset = float(details["v"][1])
                volume_in_usd = volume_in_base_asset * price

                if volume_in_usd > 1_500_000:
                    high_volume[pair] = {
                        "volume": volume_in_usd,
                        "price": price
                    }
                    
            except (KeyError, ValueError) as e:
                print(f"DEBUG: Error processing {pair}: {str(e)}")
                continue

    return high_volume

def calculate_probability(kraken_pair):
    ohlc_data = get_ohlc_data(kraken_pair, 5)
    if len(ohlc_data) < 6:
        return False
    start_price = float(ohlc_data[0][4])
    end_price = float(ohlc_data[-1][4])
    change = ((end_price - start_price) / start_price) * 100
    return change >= 1.5

def calculate_probability_tier1(kraken_pair):
    """First tier: 1.5% over 6 periods"""
    ohlc_data = get_ohlc_data(kraken_pair, 5)
    if len(ohlc_data) < 6:
        return False
    start_price = float(ohlc_data[0][4])
    end_price = float(ohlc_data[-1][4])
    change = ((end_price - start_price) / start_price) * 100
    return change >= 1.5

def calculate_probability_tier2(kraken_pair):
    """Second tier: 1% over 8 periods"""
    ohlc_data = get_ohlc_data(kraken_pair, 5)
    if len(ohlc_data) < 8:
        return False
    start_price = float(ohlc_data[0][4])
    end_price = float(ohlc_data[-1][4])
    change = ((end_price - start_price) / start_price) * 100
    return change >= 1.0

def find_top_cryptos():
    kraken_cryptos = fetch_kraken_cryptos()
    tier1_gainers = []
    tier2_gainers = []
    
    # Get all Tier 1 cryptos (1.5% over 6 periods)
    for kraken_pair in kraken_cryptos.keys():
        base_asset = kraken_pair.replace("USD", "")
        if calculate_probability_tier1(kraken_pair):
            tier1_gainers.append(base_asset)
    
    # Get Tier 2 cryptos (1% over 8 periods) from remaining pairs
    for kraken_pair in kraken_cryptos.keys():
        base_asset = kraken_pair.replace("USD", "")
        if base_asset not in tier1_gainers and calculate_probability_tier2(kraken_pair):
            tier2_gainers.append(base_asset)
    
    return tier1_gainers[:3], tier2_gainers[:3]  # Return top 3 from each tier

def fetch_top_volume_cryptos():
    data = fetch_kraken_cryptos()  # already filtered out pairs starting with 'C','X','Z'
    return data

def calculate_roc(pair, ohlc_data):
    if len(ohlc_data) < 24:
        return 0
    peaks = []
    for i in range(1, 13):
        try:
            current_close = float(ohlc_data[-i][4])
            previous_close = float(ohlc_data[-(i + 1)][4])
            roc = ((current_close - previous_close) / previous_close) * 100
            peaks.append(roc)
        except:
            continue
    return sum(peaks) / len(peaks) if peaks else 0

def calculate_one_percent_change(pair, ohlc_data):
    if len(ohlc_data) < 48:  # 4 hours = 48 five-minute candles
        return 0
    count = 0
    in_positive_gain = False
    for i in range(1, 48):
        try:
            current_close = float(ohlc_data[-i][4])
            previous_close = float(ohlc_data[-(i + 1)][4])
            roc = ((current_close - previous_close) / previous_close) * 100
            if roc > 1:
                if not in_positive_gain:
                    count += 1
                    in_positive_gain = True
            elif roc < -1:
                in_positive_gain = False
        except:
            continue
    return count

def calculate_high_low(pair, ohlc_data):
    if len(ohlc_data) < 36:  # 3 hours = 36 five-minute candles
        return "0.00% | 0.00%"
    
    # Get only the last 3 hours of data (36 five-minute candles)
    recent_data = ohlc_data[-36:]
    
    highs = []
    lows = []
    
    # We need to iterate up to len-1 since we're comparing with previous candle
    for i in range(len(recent_data)-1):
        try:
            current_close = float(recent_data[i+1][4])  # Current candle close
            previous_close = float(recent_data[i][4])   # Previous candle close
            
            roc = ((current_close - previous_close) / previous_close) * 100
            if roc > 0:
                highs.append(roc)
            elif roc < 0:
                lows.append(abs(roc))  # Store absolute value for easier comparison
        except (IndexError, ValueError) as e:
            print(f"Error processing high/low for {pair}: {str(e)}")
            continue
            
    # Handle cases where we might not have any highs or lows
    high = max(highs) if highs else 0.00
    low = max(lows) if lows else 0.00  # Using max since we stored absolute values
    
    return f"{high:.2f}% | -{low:.2f}%"

def calculate_roc_activity(pair, ohlc_data):
    if len(ohlc_data) < 24:
        return 0
    
    activity = 0
    
    # We need to iterate up to len-1 since we're comparing with previous candle
    for i in range(len(ohlc_data)-1):
        try:
            current_close = float(ohlc_data[i+1][4])  # Current candle close
            previous_close = float(ohlc_data[i][4])   # Previous candle close
            
            roc = abs(((current_close - previous_close) / previous_close) * 100)
            activity += roc
            
        except (IndexError, ValueError) as e:
            print(f"Error calculating RoC activity for {pair}: {str(e)}")
            continue
    
    # Return average activity
    return round(activity / (len(ohlc_data)-1), 2) if len(ohlc_data) > 1 else 0

def calculate_current_trend(pair, ohlc_data):
    if len(ohlc_data) < 3:  # Use last 3 candles for better trend confirmation
        return "-"
        
    last_candles = ohlc_data[-3:]
    trend_count = {"up": 0, "down": 0}
    
    for candle in last_candles:
        open_price = float(candle[1])
        close_price = float(candle[4])
        price_change_pct = ((close_price - open_price) / open_price) * 100
        
        if price_change_pct > 0.1:
            trend_count["up"] += 1
        elif price_change_pct < -0.1:
            trend_count["down"] += 1
            
    # Determine trend based on majority direction
    if trend_count["up"] >= 2:  # At least 2 out of 3 candles show upward movement
        return "up"
    elif trend_count["down"] >= 2:  # At least 2 out of 3 candles show downward movement
        return "down"
    return "-"  # No clear trend

def check_roc_spikes(ohlc_data, hours=5):
    """
    Check if the Rate of Change has spiked above 2% at least 3 times in the specified hours
    Args:
        ohlc_data: List of OHLC candles (5-minute intervals)
        hours: Number of hours to look back
    Returns:
        bool: True if condition is met, False otherwise
    """
    try:
        # Convert hours to number of 5-minute candles
        num_candles = hours * 12  # 12 five-minute candles per hour
        relevant_candles = ohlc_data[-num_candles:]
        if len(relevant_candles) < num_candles:
            return False
            
        spike_count = 0
        last_spike_idx = None
        
        # Calculate RoC for each candle
        for i in range(1, len(relevant_candles)):
            current_close = float(relevant_candles[i][4])
            previous_close = float(relevant_candles[i-1][4])
            roc = ((current_close - previous_close) / previous_close) * 100
            
        # In check_roc_spikes function
        if roc > 2.0:
            spike_count += 1
            # Removed adjacent check completely
                
        return spike_count >= 2  # Changed from 3 to 2
        
    except Exception as e:
        print(f"Error in check_roc_spikes: {str(e)}")
        return False  

def calculate_recent_weight(ohlc_data):
    """
    Calculate weight giving more importance to recent activity
    """
    # Last hour (12 candles)
    recent_candles = ohlc_data[-12:]
    recent_roc_spikes = sum(1 for i in range(1, len(recent_candles))
        if abs(float(recent_candles[i][4]) - float(recent_candles[i-1][4])) 
        / float(recent_candles[i-1][4]) * 100 > 1.0)
    
    # Weight recent spikes more heavily
    return recent_roc_spikes * 2  # Double weight for recent hour

def has_favorable_conditions(pair, ohlc_data, hours=3):
    try:
        num_candles = hours * 12
        if len(ohlc_data) < num_candles:
            return {'roc_spikes': False, 'volume_trend': False, 'above_avg_volume': False, 
                   'increasing_volatility': False, 'recent_moves': False}, False

        # Track which conditions are met
        conditions = {
            'roc_spikes': check_roc_spikes(ohlc_data, hours),
            'volume_trend': check_volume_trend(ohlc_data),
            'above_avg_volume': check_above_avg_volume(ohlc_data),
            'increasing_volatility': check_increasing_volatility(ohlc_data),
            'recent_moves': count_recent_moves(ohlc_data) >= 3
        }
        
        # Print debug info
        print(f"\nAnalyzing {pair}:")
        for condition, result in conditions.items():
            print(f"{condition}: {result}")
            
        conditions_met = sum(1 for condition in conditions.values() if condition)
        print(f"Total conditions met: {conditions_met}")

        has_trend = any(conditions.values())
        print(f"Has trend: {has_trend}")

        return conditions, has_trend
        #return conditions, conditions_met >= 2  # Return both conditions dict and boolean

    except Exception as e:
        print(f"Error checking conditions for {pair}: {str(e)}")
        return {'roc_spikes': False, 'volume_trend': False, 'above_avg_volume': False, 
                'increasing_volatility': False, 'recent_moves': False}, False

def check_volume_trend(ohlc_data):
    """
    Check if volume is trending up
    """
    recent_volumes = [float(candle[6]) for candle in ohlc_data[-12:]]  # Last hour
    first_half_avg = sum(recent_volumes[:6]) / 6
    second_half_avg = sum(recent_volumes[6:]) / 6
    
    # Return True if volume is increasing
    return second_half_avg > first_half_avg * 1.1  # 10% increase

def check_above_avg_volume(ohlc_data):
    """
    Check if recent volume is above 24h average
    """
    day_volumes = [float(candle[6]) for candle in ohlc_data[-288:]]  # Last 24 hours
    recent_volumes = [float(candle[6]) for candle in ohlc_data[-3:]]  # Last 15 minutes
    
    day_avg = sum(day_volumes) / len(day_volumes)
    recent_avg = sum(recent_volumes) / len(recent_volumes)
    
    return recent_avg > day_avg * 1.2  # 20% above daily average

def check_increasing_volatility(ohlc_data):
    """
    Check if candle ranges are widening
    """
    recent_candles = ohlc_data[-6:]  # Last 30 minutes
    ranges = []
    
    for candle in recent_candles:
        high = float(candle[2])
        low = float(candle[3])
        candle_range = ((high - low) / low) * 100
        ranges.append(candle_range)
    
    # Check if ranges are increasing
    return ranges[-1] > sum(ranges[:-1]) / (len(ranges) - 1)

def count_recent_moves(ohlc_data):
    """
    Count number of >1% moves in recent history
    """
    count = 0
    recent_candles = ohlc_data[-24:]  # Last 2 hours
    
    for i in range(1, len(recent_candles)):
        curr_close = float(recent_candles[i][4])
        prev_close = float(recent_candles[i-1][4])
        move = abs((curr_close - prev_close) / prev_close * 100)
        if move > 1.0:
            count += 1
            
    return count

def filter_and_sort_cryptos(cryptos):
    filtered_cryptos = []
    seen_pairs = set()  # Track unique pairs
    min_conditions = 2  # Start with requiring 2 conditions
    
    def add_crypto(pair, details, hours, conditions_met, conditions):
        base = pair.replace("USD", "")
        if base in seen_pairs:  # Skip if we've already added this pair
            return
            
        seen_pairs.add(base)  # Add to seen pairs set
        
        ohlc_data = get_ohlc_data(pair, 5)
        # Calculate crypto metrics
        avg_roc = calculate_roc(pair, ohlc_data)
        roc_activity = calculate_roc_activity(pair, ohlc_data)
        one_percent_gain = calculate_one_percent_change(pair, ohlc_data)
        high_low = calculate_high_low(pair, ohlc_data)
        current_trend = calculate_current_trend(pair, ohlc_data)
        current_price = float(ohlc_data[-1][4])
        strength_score = roc_activity * 100

        filtered_cryptos.append({
            "pair": base,
            "avg_roc": round(avg_roc, 3),
            "roc_activity": roc_activity,
            "volume": round(details["volume"], 1),
            "one_percent_gain": one_percent_gain,
            "high_low": high_low,
            "current_trend": current_trend,
            "price": current_price,
            "score": strength_score,
            "time_window": f"{hours}h",
            "conditions": conditions,
            "has_trend": True
        })
    
    # First pass - look for cryptos with 2+ conditions
    for hours in [3, 5, 8]:
        for pair, details in cryptos.items():
            if pair.replace("USD", "") in seen_pairs:
                continue

            ohlc_data = get_ohlc_data(pair, 5)
            conditions, has_trend = has_favorable_conditions(pair, ohlc_data, hours)
            conditions_met = sum(conditions.values())

            if conditions_met >= min_conditions:
                add_crypto(pair, details, hours, conditions_met, conditions)

        # If we found enough cryptos with 2+ conditions, break
        if len(filtered_cryptos) >= 5:
            break
            
    # Second pass - look for cryptos with 1 condition if we need more
    if len(filtered_cryptos) < 5:
        for hours in [3, 5, 8]:
            for pair, details in cryptos.items():
                if pair.replace("USD", "") in seen_pairs:
                    continue
                    
                ohlc_data = get_ohlc_data(pair, 5)
                conditions, _ = has_favorable_conditions(pair, ohlc_data, hours)
                conditions_met = sum(conditions.values())

                if conditions_met >= 1:  # Reduced threshold
                    add_crypto(pair, details, hours, conditions_met, conditions)

            if len(filtered_cryptos) >= 5:
                break

    return sorted(filtered_cryptos, key=lambda x: x["score"], reverse=True)[:20]

# @app.route('/launch-trading-window', methods=['POST'])
# def launch_trading_window():
#     try:
#         script_path = os.path.join(os.path.dirname(__file__), "kb.py")
#         subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return jsonify({"status": "success", "message": "Trading window launched successfully."}), 200
#     except Exception as e:
#         print(f"Error launching trading window: {e}")
#         return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/update")
def update():
    try:
        cryptos = fetch_top_volume_cryptos()
        sorted_cryptos = filter_and_sort_cryptos(cryptos)
        return jsonify({"cryptos": sorted_cryptos})
    except Exception as e:
        print(f"Error in /update route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/test_connection", methods=["GET"])
def test_connection():
    try:
        # Test public API
        public_resp = requests.get(KRAKEN_API_URL, timeout=10)
        public_status = "OK" if public_resp.ok else f"Error: {public_resp.status_code}"

        # Test private API
        private_resp = kraken_private_request("/0/private/Balance", {})
        private_status = "OK" if not private_resp.get("error") else f"Error: {private_resp['error']}"

        # Test order book
        orderbook = requests.get("https://api.kraken.com/0/public/Depth", 
                               params={"pair": "BTCUSD"}, timeout=10)
        orderbook_status = "OK" if orderbook.ok else f"Error: {orderbook.status_code}"

        return jsonify({
            "public_api": public_status,
            "private_api": private_status,
            "orderbook": orderbook_status,
            "api_key_exists": bool(KRAKEN_API_KEY),
            "api_secret_exists": bool(KRAKEN_API_SECRET)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    tier1_cryptos, tier2_cryptos = find_top_cryptos()
    cryptos = fetch_top_volume_cryptos()
    sorted_cryptos = filter_and_sort_cryptos(cryptos)
    return render_template(
        "index.html",
        tier1_cryptos=tier1_cryptos,
        tier2_cryptos=tier2_cryptos,
        cryptos_script2=sorted_cryptos
    )

def open_chrome():
    time.sleep(2)
    url = 'http://127.0.0.1:5000'
    
    # Detect operating system
    platform = sys.platform
    
    if platform == "darwin":  # macOS
        chrome_paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
        ]
        
        # Try to open with Chrome or Edge
        for browser in chrome_paths:
            if os.path.exists(browser):
                try:
                    subprocess.Popen([browser, '--new-window', url])
                    return
                except Exception as e:
                    print(f"Error opening browser: {e}")
                    continue
                    
    elif platform == "win32":  # Windows
        chrome_paths = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
        ]
        
        # Try to open with Chrome or Edge
        for browser in chrome_paths:
            if os.path.exists(browser):
                try:
                    subprocess.Popen([browser, '--new-window', url])
                    return
                except Exception as e:
                    print(f"Error opening browser: {e}")
                    continue
    
    # Fallback to default browser if Chrome/Edge not found
    print("Chrome/Edge not found, using default browser")
    webbrowser.open(url, new=2)

if __name__ == "__main__":
    # Ensure application is accessible from all local network interfaces
    threading.Thread(target=open_chrome).start()
    app.run(host='0.0.0.0', port=5000, debug=False)

@app.route("/order/stop_loss_sell", methods=["POST"])
def create_stop_loss_sell():
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        sell_percent = float(data.get("sell_percent", 0))  # Get the exact percentage from input
        
        print(f"DEBUG: Received stop-loss sell request for row {row_id} with {sell_percent}%")
        
        state = order_states.get(row_id)
        if not state:
            return jsonify({"error": "Order state not found"}), 404
            
        if state.get("status") != "buy_filled":
            return jsonify({"error": "Order not in correct state for stop-loss"}), 400
            
        if sell_percent <= 0:
            return jsonify({"error": "Invalid stop-loss percentage"}), 400
            
        try:
            # Use the exact sell_percent provided by the user
            sell_txid = place_stop_loss_sell(
                state["pair"],
                state["filled_volume"],
                sell_percent,  # Use exactly what the user provided
                state["buy_fill_price"]
            )
            
            # Update state
            state.update({
                "status": "sell_open",
                "sell_order_id": sell_txid,
                "current_price_diff": 0,
                "history": state.get("history", []) + [{
                    "timestamp": time.time(),
                    "status": "sell_open",
                    "order_id": sell_txid,
                    "stop_loss": sell_percent
                }]
            })
            
            return jsonify({
                "status": "success",
                "order_id": sell_txid
            }), 200
            
        except Exception as e:
            print(f"Error placing stop-loss: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"Error in stop_loss_sell: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/order/limit_sell_now", methods=["POST"])
def limit_sell_now():
    try:
        data = request.get_json()
        row_id = data.get("row_id")
        state = order_states.get(row_id)
        
        if not state or state["status"] != OrderStatus.SELL_OPEN:
            return jsonify({"error": "No active sell order found"}), 400
            
        # Cancel existing stop-loss sell order first
        if state.get("sell_order_id"):
            cancel_result = kraken_private_request(
                "/0/private/CancelOrder",
                {"txid": state["sell_order_id"]}
            )
            if cancel_result.get("error"):
                raise ValueError(f"Failed to cancel existing sell order: {cancel_result['error']}")

        # Get current price for the pair
        current_price = get_current_price_from_kraken(state["pair"])
        if not current_price:
            raise ValueError("Failed to get current price")

        # Calculate the gain percentage
        buy_price = state.get("buy_fill_price")
        if not buy_price:
            raise ValueError("Buy price not found in order state")
            
        gain_percent = ((current_price - buy_price) / buy_price) * 100
        
        # Place limit sell order at current price
        price_precision = PAIR_PRICE_PRECISION.get(state["pair"], PAIR_PRICE_PRECISION["default"])
        
        order_data = {
            "ordertype": "limit",
            "type": "sell",
            "pair": state["pair"],
            "volume": f"{state['filled_volume']:.8f}",
            "price": f"{current_price:.{price_precision}f}"
        }

        # Validate first
        validation = kraken_private_request("/0/private/AddOrder", {"validate": True, **order_data})
        if validation.get("error"):
            raise ValueError(f"Limit sell validation failed: {validation['error']}")

        # Place actual order
        result = kraken_private_request("/0/private/AddOrder", order_data)
        if result.get("error"):
            raise ValueError(f"Failed to place limit sell: {result['error']}")

        txid = result["result"]["txid"][0]

        # Update order state
        state["sell_order_id"] = txid
        state["history"].append({
            "timestamp": time.time(),
            "status": "limit_sell_placed",
            "price": current_price,
            "gain_percent": gain_percent,
            "order_id": txid
        })

        return jsonify({
            "status": "success",
            "order_id": txid,
            "price": current_price,
            "gain_percent": gain_percent
        }), 200

    except Exception as e:
        print(f"Error in limit_sell_now: {str(e)}")
        
def place_stop_loss_sell_with_monitoring(state, stop_loss_percent):
    """Place a stop-loss sell order and set up monitoring for price increases"""
    try:
        current_price = get_current_price_from_kraken(state["pair"])
        stop_price = current_price * (1 - stop_loss_percent/100)
        
        # Place stop-loss order
        order_data = {
            "ordertype": "stop-loss",
            "type": "sell",
            "pair": state["pair"],
            "volume": f"{state['filled_volume']:.8f}",
            "price": f"{stop_price:.{PAIR_PRICE_PRECISION.get(state['pair'], 4)}f}"
        }
        
        # Validate and place order
        validation = kraken_private_request("/0/private/AddOrder", {"validate": True, **order_data})
        if validation.get("error"):
            raise ValueError(f"Stop-loss validation failed: {validation['error']}")
            
        result = kraken_private_request("/0/private/AddOrder", order_data)
        if result.get("error"):
            raise ValueError(f"Failed to place stop-loss: {result['error']}")
            
        txid = result["result"]["txid"][0]
        
        # Update state
        state["sell_order_id"] = txid
        state["status"] = OrderStatus.SELL_OPEN
        state["stop_loss_percent"] = stop_loss_percent
        state["last_renewal_price"] = current_price
        state["history"].append({
            "timestamp": time.time(),
            "status": "stop_loss_placed",
            "price": stop_price,
            "stop_loss_percent": stop_loss_percent
        })
        
        return True, txid
        
    except Exception as e:
        print(f"Error placing stop-loss: {str(e)}")
        return False, None

def check_and_renew_stop_loss(state):
    """Check if stop-loss should be renewed based on price increase"""
    try:
        if not state.get("sell_order_id") or not state.get("stop_loss_percent"):
            return False
            
        current_price = get_current_price_from_kraken(state["pair"])
        last_renewal_price = state.get("last_renewal_price", state["buy_fill_price"])
        price_increase = ((current_price - last_renewal_price) / last_renewal_price) * 100
        
        # If price has increased by half the stop-loss percentage
        if price_increase >= state["stop_loss_percent"] / 2:
            # Cancel existing stop-loss
            cancel_result = kraken_private_request(
                "/0/private/CancelOrder",
                {"txid": state["sell_order_id"]}
            )
            
            if not cancel_result.get("error"):
                # Place new stop-loss at higher price
                success, new_txid = place_stop_loss_sell_with_monitoring(state, state["stop_loss_percent"])
                if success:
                    state["last_renewal_price"] = current_price
                    state["history"].append({
                        "timestamp": time.time(),
                        "status": "stop_loss_renewed",
                        "price": current_price,
                        "price_increase": price_increase
                    })
                    return True
                    
        return False
        
    except Exception as e:
        print(f"Error checking/renewing stop-loss: {str(e)}")
        return False