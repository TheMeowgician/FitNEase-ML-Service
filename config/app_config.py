"""
Application Configuration
========================

Configuration settings for FitNEase ML Service
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""

    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # Database Configuration
    ML_DB_HOST = os.environ.get('ML_DB_HOST', 'localhost')
    ML_DB_PORT = int(os.environ.get('ML_DB_PORT', 3306))
    ML_DB_NAME = os.environ.get('ML_DB_NAME', 'fitnease_ml_db')
    ML_DB_USER = os.environ.get('ML_DB_USER', 'root')
    ML_DB_PASSWORD = os.environ.get('ML_DB_PASSWORD', 'rootpassword')

    # Service URLs
    LARAVEL_AUTH_URL = os.environ.get('LARAVEL_AUTH_URL', 'http://fitneaseauth:8001')
    LARAVEL_CONTENT_URL = os.environ.get('LARAVEL_CONTENT_URL', 'http://fitneasecontent:8002')
    LARAVEL_TRACKING_URL = os.environ.get('LARAVEL_TRACKING_URL', 'http://fitneasetracking:8007')
    LARAVEL_PLANNING_URL = os.environ.get('LARAVEL_PLANNING_URL', 'http://fitneaseplanning:8005')
    LARAVEL_ENGAGEMENT_URL = os.environ.get('LARAVEL_ENGAGEMENT_URL', 'http://fitneaseengagement:8008')

    # Model Configuration
    MODEL_RETRAIN_INTERVAL = int(os.environ.get('MODEL_RETRAIN_INTERVAL', 24))  # hours
    MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE_THRESHOLD', 0.7))
    HYBRID_CONTENT_WEIGHT = float(os.environ.get('HYBRID_CONTENT_WEIGHT', 0.6))
    HYBRID_COLLABORATIVE_WEIGHT = float(os.environ.get('HYBRID_COLLABORATIVE_WEIGHT', 0.4))

    # Request Configuration
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 30))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))

    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Model File Paths
    MODELS_DIRECTORY = os.environ.get('MODELS_DIRECTORY', 'models_pkl')
    CONTENT_BASED_MODEL_FILE = 'fitnease_content_based_model.pkl'
    COLLABORATIVE_MODEL_FILE = 'proper_cf_model.pkl'
    HYBRID_MODEL_FILE = 'fitnease_hybrid_complete.pkl'
    RANDOM_FOREST_MODEL_FILE = 'fitnease_rf_single.pkl'

    # Performance Configuration
    RECOMMENDATION_CACHE_TTL = int(os.environ.get('RECOMMENDATION_CACHE_TTL', 300))  # 5 minutes
    BATCH_SIZE_LIMIT = int(os.environ.get('BATCH_SIZE_LIMIT', 100))
    MAX_RECOMMENDATIONS = int(os.environ.get('MAX_RECOMMENDATIONS', 50))

    # Security Configuration
    CORS_ENABLED = os.environ.get('CORS_ENABLED', 'True').lower() == 'true'
    ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')

    # Health Check Configuration
    HEALTH_CHECK_TIMEOUT = int(os.environ.get('HEALTH_CHECK_TIMEOUT', 5))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    ML_DB_NAME = 'fitnease_ml_test_db'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}