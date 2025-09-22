"""
FitNEase ML Service - Main Flask Application
============================================

Purpose: Machine Learning & Intelligence Service
Technology: Python Flask + scikit-learn/pandas/numpy
Responsibility: Hybrid recommender system, Random Forest classifier,
               behavioral pattern analysis, intelligent recommendation scoring

Author: FitNEase Development Team
Date: 2025-09-22
"""

from flask import Flask
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import custom classes for pickle loading
import sys
from ml_models.custom_classes import (
    FitNeaseFeatureEngineer, ProperCollaborativeFiltering, FinalHybridRecommender,
    ProperCFConfig, ContentBasedConfig, ContentBasedRecommenderModel, HybridRecommenderModel,
    FitNeaseContentBasedRecommender
)

# Add classes to main module for pickle compatibility
sys.modules['__main__'].FitNeaseFeatureEngineer = FitNeaseFeatureEngineer
sys.modules['__main__'].ProperCollaborativeFiltering = ProperCollaborativeFiltering
sys.modules['__main__'].FinalHybridRecommender = FinalHybridRecommender
sys.modules['__main__'].ProperCFConfig = ProperCFConfig
sys.modules['__main__'].ContentBasedConfig = ContentBasedConfig
sys.modules['__main__'].ContentBasedRecommenderModel = ContentBasedRecommenderModel
sys.modules['__main__'].HybridRecommenderModel = HybridRecommenderModel
sys.modules['__main__'].FitNeaseContentBasedRecommender = FitNeaseContentBasedRecommender

# Import routes
from routes.api_routes import api_bp
from routes.health_routes import health_bp

# Import core modules
from config.app_config import Config
from models.database_models import init_db
from services.model_manager import ModelManager

# Load environment variables
load_dotenv()

def create_app():
    """Flask application factory"""
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(Config)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("Initializing FitNEase ML Service...")

    # Initialize database
    init_db()

    # Initialize model manager
    app.model_manager = ModelManager()
    app.model_manager.load_models()

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix='/api/v1')

    logger.info("FitNEase ML Service initialized successfully")

    return app

# Create Flask app
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    logger = logging.getLogger(__name__)
    logger.info("Starting FitNEase ML Service...")
    logger.info(f"Port: {port}, Debug: {debug}")
    logger.info("Available endpoints:")
    logger.info("- GET /health - Health check")
    logger.info("- GET /api/v1/model-health - Detailed model health")
    logger.info("- GET /api/v1/recommendations/{user_id} - Hybrid recommendations")
    logger.info("- POST /api/v1/content-similarity - Content-based similarity")
    logger.info("- POST /api/v1/predict-difficulty - Random Forest predictions")
    logger.info("- POST /api/v1/train-model - Model training")

    app.run(host='0.0.0.0', port=port, debug=debug)