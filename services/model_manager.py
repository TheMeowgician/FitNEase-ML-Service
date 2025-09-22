"""
Model Manager for FitNEase ML Service
====================================

Centralized model loading and management
"""

import os
import logging
import pickle
import joblib
from typing import Optional

from ml_models.content_based_recommender import ContentBasedRecommender
from ml_models.collaborative_recommender import CollaborativeRecommender
from ml_models.hybrid_recommender import HybridRecommender
from ml_models.random_forest_predictor import RandomForestPredictor
# Import all custom classes required for pickle loading
from ml_models.custom_classes import (
    FitNeaseFeatureEngineer, ProperCollaborativeFiltering, FinalHybridRecommender,
    FitNeaseContentBasedRecommender, ContentBasedRecommenderModel, HybridRecommenderModel,
    ContentBasedConfig, ProperCFConfig
)

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized ML model manager"""

    def __init__(self):
        self.content_model = None
        self.collaborative_model = None
        self.hybrid_model = None
        self.random_forest_model = None
        self.models_loaded = False

    def load_models(self) -> bool:
        """Load all ML models"""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_pkl')

            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)

            # Load Content-Based Model
            content_model_path = os.path.join(models_dir, 'fitnease_content_based_model.pkl')
            if os.path.exists(content_model_path):
                try:
                    with open(content_model_path, 'rb') as f:
                        content_model_data = pickle.load(f)
                    # The pickle file contains the complete model, use it directly
                    self.content_model = ContentBasedRecommender()
                    self.content_model.set_model_data(content_model_data)
                    logger.info("Content-based model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load content-based model: {e}")
                    # Create fallback content-based model
                    self.content_model = ContentBasedRecommender()
                    logger.info("Content-based model fallback initialized")
            else:
                # Create fallback content-based model if file doesn't exist
                self.content_model = ContentBasedRecommender()
                logger.info("Content-based model fallback initialized")

            # Load Collaborative Model (placeholder)
            collaborative_model_path = os.path.join(models_dir, 'proper_cf_model.pkl')
            if os.path.exists(collaborative_model_path):
                try:
                    with open(collaborative_model_path, 'rb') as f:
                        collaborative_model_data = pickle.load(f)
                    # The pickle file contains the complete model
                    self.collaborative_model = CollaborativeRecommender(collaborative_model_data)
                    logger.info("Collaborative model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load collaborative model: {e}")
                    # Create placeholder collaborative model
                    self.collaborative_model = CollaborativeRecommender()
                    logger.info("Collaborative model placeholder initialized")
            else:
                # Create placeholder collaborative model
                self.collaborative_model = CollaborativeRecommender()
                logger.info("Collaborative model placeholder initialized")

            # Load Hybrid Model
            hybrid_model_path = os.path.join(models_dir, 'fitnease_hybrid_complete.pkl')
            if os.path.exists(hybrid_model_path):
                try:
                    with open(hybrid_model_path, 'rb') as f:
                        hybrid_model_data = pickle.load(f)
                    # The pickle file contains the complete hybrid model
                    self.hybrid_model = HybridRecommender()
                    self.hybrid_model.set_model_data(hybrid_model_data)
                    logger.info("Hybrid model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load hybrid model: {e}")
                    # Create fallback hybrid model
                    self.hybrid_model = HybridRecommender()
                    logger.info("Hybrid model fallback initialized")
            else:
                # Create fallback hybrid model if file doesn't exist
                self.hybrid_model = HybridRecommender()
                logger.info("Hybrid model fallback initialized")

            # Load Random Forest Model
            rf_model_path = os.path.join(models_dir, 'fitnease_rf_single.pkl')
            if os.path.exists(rf_model_path):
                try:
                    with open(rf_model_path, 'rb') as f:
                        rf_model_data = pickle.load(f)
                    self.random_forest_model = RandomForestPredictor(rf_model_data)
                    logger.info("Random Forest model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Random Forest model: {e}")

            self.models_loaded = True

            # Log summary
            loaded_models = [
                name for name, model in [
                    ('content_based', self.content_model),
                    ('collaborative', self.collaborative_model),
                    ('hybrid', self.hybrid_model),
                    ('random_forest', self.random_forest_model)
                ] if model is not None
            ]

            logger.info(f"Model loading completed. Loaded models: {loaded_models}")
            return True

        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return False

    def reload_models(self) -> bool:
        """Reload all models"""
        logger.info("Reloading ML models...")
        return self.load_models()

    def get_content_based_model(self) -> Optional[ContentBasedRecommender]:
        """Get content-based model"""
        return self.content_model

    def get_collaborative_model(self) -> Optional[CollaborativeRecommender]:
        """Get collaborative model"""
        return self.collaborative_model

    def get_hybrid_model(self) -> Optional[HybridRecommender]:
        """Get hybrid model"""
        return self.hybrid_model

    def get_random_forest_model(self) -> Optional[RandomForestPredictor]:
        """Get Random Forest model"""
        return self.random_forest_model

    def get_model_status(self) -> dict:
        """Get status of all models"""
        return {
            'content_based_loaded': self.content_model is not None,
            'collaborative_loaded': self.collaborative_model is not None,
            'hybrid_loaded': self.hybrid_model is not None,
            'random_forest_loaded': self.random_forest_model is not None,
            'models_loaded': self.models_loaded
        }

    def get_models_directory(self) -> str:
        """Get the models directory path"""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_pkl')

    def list_available_model_files(self) -> list:
        """List available model files in the models directory"""
        models_dir = self.get_models_directory()
        if not os.path.exists(models_dir):
            return []

        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(models_dir, file)
                file_size = os.path.getsize(file_path)
                model_files.append({
                    'filename': file,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })

        return model_files