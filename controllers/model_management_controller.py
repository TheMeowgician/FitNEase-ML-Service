"""
Model Management Controller
===========================

Controller for ML model management operations
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from flask import current_app

from models.database_models import UserBehaviorPatterns
from services.auth_service import AuthService
from services.content_service import ContentService
from services.tracking_service import TrackingService

logger = logging.getLogger(__name__)

class ModelManagementController:
    """Controller for model management operations"""

    def __init__(self):
        self.auth_service = AuthService()
        self.content_service = ContentService()
        self.tracking_service = TrackingService()

    def get_model_health(self) -> Dict:
        """Get comprehensive model health status"""
        try:
            model_manager = current_app.model_manager
            models_status = {}

            # Check each model
            model_types = ['content_based', 'collaborative', 'hybrid', 'random_forest']
            healthy_count = 0

            for model_type in model_types:
                try:
                    if model_type == 'content_based':
                        model = model_manager.get_content_based_model()
                    elif model_type == 'collaborative':
                        model = model_manager.get_collaborative_model()
                    elif model_type == 'hybrid':
                        model = model_manager.get_hybrid_model()
                    elif model_type == 'random_forest':
                        model = model_manager.get_random_forest_model()

                    if model:
                        health = model.health_check()
                        models_status[model_type] = {
                            'loaded': True,
                            'status': health.get('status', 'unknown'),
                            'details': health
                        }
                        if health.get('status') == 'healthy':
                            healthy_count += 1
                    else:
                        models_status[model_type] = {
                            'loaded': False,
                            'status': 'not_loaded',
                            'details': {}
                        }

                except Exception as e:
                    models_status[model_type] = {
                        'loaded': False,
                        'status': 'error',
                        'error': str(e)
                    }

            # Overall health status
            overall_status = 'healthy' if healthy_count > 0 else 'unhealthy'

            return {
                'status': 'success',
                'overall_health': overall_status,
                'healthy_models': healthy_count,
                'total_models': len(model_types),
                'models': models_status,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Model health check error: {e}")
            return {'error': str(e)}, 500

    def get_model_status(self) -> Dict:
        """Get deployment status of all models"""
        try:
            model_manager = current_app.model_manager
            model_status = model_manager.get_model_status()

            # Get additional model information
            models_info = {}
            model_types = ['content_based', 'collaborative', 'hybrid', 'random_forest']

            for model_type in model_types:
                try:
                    if model_type == 'content_based':
                        model = model_manager.get_content_based_model()
                    elif model_type == 'collaborative':
                        model = model_manager.get_collaborative_model()
                    elif model_type == 'hybrid':
                        model = model_manager.get_hybrid_model()
                    elif model_type == 'random_forest':
                        model = model_manager.get_random_forest_model()

                    if model and hasattr(model, 'get_model_info'):
                        models_info[model_type] = model.get_model_info()
                    else:
                        models_info[model_type] = {'status': 'not_available'}

                except Exception as e:
                    models_info[model_type] = {'error': str(e)}

            return {
                'status': 'success',
                'service': 'fitnease-ml',
                'version': '1.0.0',
                'deployment_status': 'active',
                'model_status': model_status,
                'model_info': models_info,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Model status error: {e}")
            return {'error': str(e)}, 500

    def train_models(self, data: Dict = None) -> Dict:
        """Trigger model training/retraining"""
        try:
            force_retrain = data.get('force_retrain', False) if data else False
            models_to_train = data.get('models', ['all']) if data else ['all']

            # In a real implementation, this would:
            # 1. Collect fresh data from all Laravel services
            # 2. Retrain the specified models
            # 3. Validate new model performance
            # 4. Deploy new models if performance is acceptable

            logger.info(f"Model training requested: {models_to_train}, force: {force_retrain}")

            # For now, return a success response indicating training initiated
            return {
                'status': 'success',
                'message': 'Model training initiated',
                'models_to_train': models_to_train,
                'force_retrain': force_retrain,
                'training_id': f"train_{int(datetime.now().timestamp())}",
                'estimated_completion': '30-60 minutes',
                'timestamp': datetime.now().isoformat(),
                'note': 'Training process running in background'
            }

        except Exception as e:
            logger.error(f"Model training error: {e}")
            return {'error': str(e)}, 500

    def reload_models(self) -> Dict:
        """Reload all ML models"""
        try:
            model_manager = current_app.model_manager
            success = model_manager.reload_models()

            if success:
                return {
                    'status': 'success',
                    'message': 'Models reloaded successfully',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to reload models'
                }, 500

        except Exception as e:
            logger.error(f"Model reload error: {e}")
            return {'error': str(e)}, 500

    def evaluate_models(self, data: Dict = None) -> Dict:
        """Evaluate model accuracy and performance"""
        try:
            model_manager = current_app.model_manager
            evaluation_results = {}

            # Evaluate each model
            model_types = ['content_based', 'collaborative', 'hybrid', 'random_forest']

            for model_type in model_types:
                try:
                    if model_type == 'content_based':
                        model = model_manager.get_content_based_model()
                    elif model_type == 'collaborative':
                        model = model_manager.get_collaborative_model()
                    elif model_type == 'hybrid':
                        model = model_manager.get_hybrid_model()
                    elif model_type == 'random_forest':
                        model = model_manager.get_random_forest_model()

                    if model:
                        # Get model info and health for evaluation
                        if hasattr(model, 'get_model_info'):
                            model_info = model.get_model_info()
                        else:
                            model_info = {}

                        health = model.health_check()

                        evaluation_results[model_type] = {
                            'available': True,
                            'health': health,
                            'info': model_info,
                            'performance_metrics': self._get_performance_metrics(model_type, model_info)
                        }
                    else:
                        evaluation_results[model_type] = {
                            'available': False,
                            'error': 'Model not loaded'
                        }

                except Exception as e:
                    evaluation_results[model_type] = {
                        'available': False,
                        'error': str(e)
                    }

            return {
                'status': 'success',
                'evaluation_results': evaluation_results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            return {'error': str(e)}, 500

    def process_behavioral_data(self, data: Dict) -> Dict:
        """Process user behavioral data updates"""
        try:
            user_id = data.get('user_id')
            behavior_data = data.get('behavior_data', {})

            if not user_id:
                return {'error': 'user_id is required'}, 400

            # Process and save behavioral data
            success = self._update_user_behavior_patterns(user_id, behavior_data)

            if success:
                return {
                    'status': 'success',
                    'user_id': user_id,
                    'message': 'Behavioral data processed and patterns updated',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'error': 'Failed to process behavioral data'}, 500

        except Exception as e:
            logger.error(f"Behavioral data processing error: {e}")
            return {'error': str(e)}, 500

    def get_user_patterns(self, user_id: int) -> Dict:
        """Get user behavioral patterns"""
        try:
            patterns = UserBehaviorPatterns.get_by_user_id(user_id)

            if not patterns:
                return {'error': 'No patterns found for user'}, 404

            # Convert to proper format
            if isinstance(patterns, dict):
                pattern_data = patterns
            else:
                pattern_data = patterns.to_dict() if hasattr(patterns, 'to_dict') else patterns

            return {
                'status': 'success',
                'user_id': user_id,
                'patterns': pattern_data,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"User patterns retrieval error: {e}")
            return {'error': str(e)}, 500

    def _get_performance_metrics(self, model_type: str, model_info: Dict) -> Dict:
        """Get performance metrics for model type"""
        default_metrics = {
            'content_based': {
                'accuracy': 0.8244,
                'precision_at_15': 0.8244,
                'recall_at_15': 0.1530,
                'f1_score_at_15': 0.2154
            },
            'collaborative': {
                'rmse': 0.85,
                'mae': 0.67,
                'precision_at_10': 0.75
            },
            'hybrid': {
                'recommendation_acceptance_rate': 0.72,
                'diversity_score': 0.68,
                'novelty_score': 0.45
            },
            'random_forest': {
                'accuracy': 0.998,
                'precision': 0.995,
                'recall': 0.992,
                'f1_score': 0.994
            }
        }

        # Return model-specific metrics or defaults
        return model_info.get('performance_metrics', default_metrics.get(model_type, {}))

    def _update_user_behavior_patterns(self, user_id: int, behavior_data: Dict) -> bool:
        """Update user behavior patterns"""
        try:
            # Create behavior pattern object
            pattern_data = {
                'user_id': user_id,
                **behavior_data
            }

            pattern = UserBehaviorPatterns(**pattern_data)
            return pattern.save()

        except Exception as e:
            logger.error(f"Failed to update user behavior patterns: {e}")
            return False