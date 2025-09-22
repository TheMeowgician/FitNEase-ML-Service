"""
Base Service for Cross-Service Communication
===========================================

Base class for communication with other Laravel services
"""

import requests
import logging
import os
from typing import Dict, Optional, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class BaseService:
    """Base service class for Laravel service communication"""

    def __init__(self, service_name: str, base_url: str):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.timeout = 30
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create configured session with retry strategy"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'FitNEase-ML-Service/1.0'
        })

        return session

    def get(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make GET request to Laravel service"""
        try:
            url = f"{self.base_url}{endpoint}"
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)

            logger.debug(f"GET request to {self.service_name}: {url}")

            response = self.session.get(
                url,
                params=params,
                headers=request_headers,
                timeout=self.timeout
            )

            return self._handle_response(response)

        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed to {self.service_name} service at {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to {self.service_name} service")
            return None
        except Exception as e:
            logger.error(f"Error communicating with {self.service_name} service: {e}")
            return None

    def post(self, endpoint: str, data: Dict = None, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make POST request to Laravel service"""
        try:
            url = f"{self.base_url}{endpoint}"
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)

            logger.debug(f"POST request to {self.service_name}: {url}")

            response = self.session.post(
                url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=self.timeout
            )

            return self._handle_response(response)

        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed to {self.service_name} service at {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to {self.service_name} service")
            return None
        except Exception as e:
            logger.error(f"Error communicating with {self.service_name} service: {e}")
            return None

    def put(self, endpoint: str, data: Dict = None, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make PUT request to Laravel service"""
        try:
            url = f"{self.base_url}{endpoint}"
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)

            logger.debug(f"PUT request to {self.service_name}: {url}")

            response = self.session.put(
                url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=self.timeout
            )

            return self._handle_response(response)

        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed to {self.service_name} service at {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to {self.service_name} service")
            return None
        except Exception as e:
            logger.error(f"Error communicating with {self.service_name} service: {e}")
            return None

    def delete(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make DELETE request to Laravel service"""
        try:
            url = f"{self.base_url}{endpoint}"
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)

            logger.debug(f"DELETE request to {self.service_name}: {url}")

            response = self.session.delete(
                url,
                params=params,
                headers=request_headers,
                timeout=self.timeout
            )

            return self._handle_response(response)

        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed to {self.service_name} service at {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to {self.service_name} service")
            return None
        except Exception as e:
            logger.error(f"Error communicating with {self.service_name} service: {e}")
            return None

    def _handle_response(self, response: requests.Response) -> Optional[Dict]:
        """Handle HTTP response"""
        try:
            # Log response details
            logger.debug(f"Response from {self.service_name}: {response.status_code}")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 201:
                return response.json()
            elif response.status_code == 204:
                return {'success': True}
            elif response.status_code == 404:
                logger.warning(f"Resource not found in {self.service_name} service")
                return None
            elif response.status_code >= 500:
                logger.error(f"Server error in {self.service_name} service: {response.status_code}")
                return None
            else:
                logger.warning(f"Unexpected response from {self.service_name}: {response.status_code}")
                try:
                    return response.json()
                except:
                    return None

        except requests.exceptions.JSONDecodeError:
            logger.error(f"Invalid JSON response from {self.service_name} service")
            return None
        except Exception as e:
            logger.error(f"Error handling response from {self.service_name}: {e}")
            return None

    def validate_response(self, response: Dict, required_fields: list = None) -> bool:
        """Validate response structure"""
        if not response:
            return False

        if required_fields:
            for field in required_fields:
                if field not in response:
                    logger.warning(f"Missing required field '{field}' in {self.service_name} response")
                    return False

        return True

    def is_service_available(self) -> bool:
        """Check if service is available"""
        try:
            response = self.get('/health')
            return response is not None and response.get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Service availability check failed for {self.service_name}: {e}")
            return False

    def get_service_info(self) -> Dict:
        """Get service information"""
        return {
            'service_name': self.service_name,
            'base_url': self.base_url,
            'available': self.is_service_available(),
            'timeout': self.timeout
        }