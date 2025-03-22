from collections import deque
import time
from functools import wraps
import threading
import logging
import queue
import random

# Create a singleton rate limiter instance
_GLOBAL_RATE_LIMITER = None
_GLOBAL_LOCK = threading.RLock()

def get_global_rate_limiter():
    """Get or create the global rate limiter instance"""
    global _GLOBAL_RATE_LIMITER
    with _GLOBAL_LOCK:
        if _GLOBAL_RATE_LIMITER is None:
            _GLOBAL_RATE_LIMITER = KrakenRateLimiter()
        return _GLOBAL_RATE_LIMITER

class KrakenRateLimiter:
    """
    Rate limiter for Kraken API requests with more conservative limits:
    - Private endpoints: 15 requests per 10 seconds (reduced from 20)
    - Public endpoints: 15 requests per 5 seconds (reduced from 20)
    
    Now includes request queueing and backoff mechanisms.
    """
    def __init__(self):
        # More conservative limits for better reliability
        self.private_max_requests = 10  # Reduced from 20
        self.public_max_requests = 10   # Reduced from 20
        
        self.private_requests = deque(maxlen=self.private_max_requests)
        self.public_requests = deque(maxlen=self.public_max_requests)
        
        self.private_window = 15  # 10 seconds window for private API
        self.public_window = 10    # 5 seconds window for public API
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Request queues for each type
        self.private_queue = queue.Queue()
        self.public_queue = queue.Queue()
        
        # Start worker threads for processing queued requests
        self.private_worker = threading.Thread(target=self._process_queue, 
                                              args=(True, self.private_queue), 
                                              daemon=True)
        self.public_worker = threading.Thread(target=self._process_queue, 
                                             args=(False, self.public_queue), 
                                             daemon=True)
        self.private_worker.start()
        self.public_worker.start()
        
        # Track consecutive errors for circuit breaker
        self.consecutive_errors = 0
        self.error_threshold = 5  # After this many errors, slow down dramatically
        self.backoff_multiplier = 1.0
        self.last_error_time = 0
        
        self.logger.info("Initialized Kraken rate limiter with more conservative limits")

    def _cleanup_old_requests(self, requests_queue, window_seconds):
        """Remove requests older than the time window"""
        current_time = time.time()
        while requests_queue and current_time - requests_queue[0] > window_seconds:
            requests_queue.popleft()

    def check_rate_limit(self, is_private=True):
        """
        Check if making a new request would exceed rate limits
        Returns: (bool, float) - (can_make_request, seconds_until_available)
        """
        with self.lock:
            current_time = time.time()
            
            # Apply circuit breaker pattern - slow down if too many errors
            if self.consecutive_errors >= self.error_threshold:
                elapsed_since_error = current_time - self.last_error_time
                if elapsed_since_error < 60:  # Within a minute of last error
                    # Add jitter to avoid thundering herd problem
                    wait_time = 1.0 + (self.backoff_multiplier * random.uniform(0.5, 1.5))
                    self.logger.warning(f"Circuit breaker active: waiting {wait_time:.2f}s")
                    return False, wait_time
                else:
                    # Reset after a minute of success
                    self.consecutive_errors = max(0, self.consecutive_errors - 1)
                    self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.8)
            
            queue = self.private_requests if is_private else self.public_requests
            window = self.private_window if is_private else self.public_window
            max_requests = self.private_max_requests if is_private else self.public_max_requests

            self._cleanup_old_requests(queue, window)

            if len(queue) < max_requests:
                return True, 0

            # Calculate when the oldest request will expire
            oldest_request = queue[0]
            time_until_available = (oldest_request + window) - current_time
            
            # Add a small buffer to be safe
            time_until_available += 0.1
                
            return False, max(0.1, time_until_available)

    def add_request(self, is_private=True):
        """Record a new API request"""
        with self.lock:
            queue = self.private_requests if is_private else self.public_requests
            queue.append(time.time())

    def get_remaining_requests(self, is_private=True):
        """Get number of remaining requests in the current window"""
        with self.lock:
            queue = self.private_requests if is_private else self.public_requests
            window = self.private_window if is_private else self.public_window
            max_requests = self.private_max_requests if is_private else self.public_max_requests
            
            self._cleanup_old_requests(queue, window)
            return max_requests - len(queue)
    
    def handle_rate_limit_error(self, is_private=True):
        """Handle a rate limit error by increasing backoff"""
        with self.lock:
            self.consecutive_errors += 1
            self.last_error_time = time.time()
            
            # Exponential backoff
            self.backoff_multiplier = min(30.0, self.backoff_multiplier * 2.0)
            
            self.logger.warning(
                f"Rate limit error encountered! Consecutive errors: {self.consecutive_errors}, "
                f"Backoff multiplier: {self.backoff_multiplier:.2f}x"
            )
    
    def handle_success(self):
        """Record a successful request to gradually reset circuit breaker"""
        with self.lock:
            if self.consecutive_errors > 0:
                self.consecutive_errors -= 1
                if self.consecutive_errors == 0:
                    self.backoff_multiplier = 1.0
                    self.logger.info("Circuit breaker reset after successful requests")
    
    def queue_request(self, func, is_private, *args, **kwargs):
        """Queue a request for later execution"""
        request_queue = self.private_queue if is_private else self.public_queue
        
        # Create a future to store the result
        result_future = queue.Queue(maxsize=1)
        
        # Add the request to the queue
        request_queue.put((func, args, kwargs, result_future))
        
        # Wait for the result
        try:
            result = result_future.get(timeout=60)  # Wait up to 60 seconds
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            raise TimeoutError("Request timed out waiting in rate limit queue")
    
    def _process_queue(self, is_private, request_queue):
        """Worker thread to process queued requests"""
        endpoint_type = "private" if is_private else "public"
        self.logger.info(f"Started {endpoint_type} API request worker thread")
        
        while True:
            try:
                # Get the next request from the queue
                func, args, kwargs, result_future = request_queue.get(block=True)
                
                # Check if we can make a request
                can_request, wait_time = self.check_rate_limit(is_private)
                
                if not can_request:
                    # If we need to wait, sleep and then retry
                    self.logger.debug(f"{endpoint_type} API: Waiting {wait_time:.2f}s before processing queued request")
                    time.sleep(wait_time)
                
                # Execute the request
                try:
                    result = func(*args, **kwargs)
                    self.add_request(is_private)
                    self.handle_success()
                    
                    # Add extra delay after successful requests for stability
                    time.sleep(0.1 + (random.random() * 0.1))  # 100-200ms
                    
                    # Return the result
                    result_future.put(result)
                except Exception as e:
                    # Check if this is a rate limit error
                    if "Rate limit" in str(e):
                        self.handle_rate_limit_error(is_private)
                        
                        # Re-queue the request with a delay
                        self.logger.info(f"{endpoint_type} API: Re-queueing failed request after rate limit error")
                        time.sleep(1.0 + random.random())  # Add jitter
                        request_queue.put((func, args, kwargs, result_future))
                    else:
                        # For other errors, return the exception
                        result_future.put(e)
                
                # Mark the task as done
                request_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in {endpoint_type} request worker: {str(e)}")
                # Sleep briefly to avoid tight loops in case of errors
                time.sleep(1.0)

def rate_limit_decorator(is_private=False):
    """
    Decorator to handle rate limiting for API calls with improved error handling
    and automatic queueing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the global rate limiter
            limiter = get_global_rate_limiter()
            
            # Get the endpoint name for logging
            endpoint = kwargs.get('endpoint_path', func.__name__)
            endpoint_type = "private" if is_private else "public"
            
            # Check if we can make a request
            can_request, wait_time = limiter.check_rate_limit(is_private)
            
            if not can_request:
                # Queue the request and wait for the result
                logging.info(f"Queueing {endpoint_type} API request to {endpoint} (wait: {wait_time:.2f}s)")
                return limiter.queue_request(func, is_private, *args, **kwargs)
            
            # Make the request
            try:
                result = func(*args, **kwargs)
                limiter.add_request(is_private)
                limiter.handle_success()
                
                # Check for rate limit errors in the response
                if isinstance(result, dict) and result.get("error"):
                    error_messages = result.get("error", [])
                    if any("Rate limit" in str(err) for err in error_messages):
                        limiter.handle_rate_limit_error(is_private)
                        
                        # Queue the request for retry
                        logging.warning(f"Rate limit in response for {endpoint_type} API request, retrying...")
                        return limiter.queue_request(func, is_private, *args, **kwargs)
                
                return result
                
            except Exception as e:
                if "Rate limit" in str(e):
                    limiter.handle_rate_limit_error(is_private)
                    
                    # Queue the request for retry
                    logging.warning(f"Rate limit error for {endpoint_type} API request, retrying...")
                    return limiter.queue_request(func, is_private, *args, **kwargs)
                
                # Re-raise other exceptions
                raise
                
        return wrapper
    return decorator