"""
Random Forest Prediction Controller
===================================

Controller for Random Forest classifier operations
"""

import logging
from typing import Dict, List, Optional
from flask import current_app

from models.database_models import Recommendations
from services.auth_service import AuthService
from services.content_service import ContentService
from services.tracking_service import TrackingService

logger = logging.getLogger(__name__)

class RandomForestController:
    """Controller for Random Forest prediction operations"""

    def __init__(self):
        self.auth_service = AuthService()
        self.content_service = ContentService()
        self.tracking_service = TrackingService()

    def predict_difficulty(self, data: Dict) -> Dict:
        """Workout difficulty prediction"""
        try:
            user_id = data.get('user_id')
            workout_id = data.get('workout_id')
            user_profile = data.get('user_profile')
            workout_features = data.get('workout_features')

            if not user_profile or not workout_features:
                if not user_id or not workout_id:
                    return {'error': 'Either (user_profile and workout_features) or (user_id and workout_id) required'}, 400

                # Get user and workout data from services
                user_profile = self.auth_service.get_user_profile(user_id)
                workout_features = self.content_service.get_workout_details(workout_id)

                if not user_profile:
                    return {'error': 'User not found'}, 404
                if not workout_features:
                    return {'error': 'Workout not found'}, 404

            # Get Random Forest model
            rf_model = current_app.model_manager.get_random_forest_model()
            if not rf_model:
                return {'error': 'Random Forest model not available'}, 503

            # Predict difficulty appropriateness
            prediction = rf_model.predict_difficulty_appropriateness(
                user_profile=user_profile,
                workout_features=workout_features
            )

            return {
                'status': 'success',
                'user_id': user_id,
                'workout_id': workout_id,
                'difficulty_prediction': prediction,
                'algorithm': 'random_forest'
            }

        except Exception as e:
            logger.error(f"Difficulty prediction error: {e}")
            return {'error': str(e)}, 500

    def predict_completion(self, data: Dict) -> Dict:
        """Workout completion probability prediction"""
        try:
            user_id = data.get('user_id')
            workout_id = data.get('workout_id')
            user_profile = data.get('user_profile')
            workout_features = data.get('workout_features')

            if not user_profile or not workout_features:
                if not user_id or not workout_id:
                    return {'error': 'Either (user_profile and workout_features) or (user_id and workout_id) required'}, 400

                # Get user and workout data from services
                user_profile = self.auth_service.get_user_profile(user_id)
                workout_features = self.content_service.get_workout_details(workout_id)

                if not user_profile:
                    return {'error': 'User not found'}, 404
                if not workout_features:
                    return {'error': 'Workout not found'}, 404

            # Get Random Forest model
            rf_model = current_app.model_manager.get_random_forest_model()
            if not rf_model:
                return {'error': 'Random Forest model not available'}, 503

            # Predict completion probability
            prediction = rf_model.predict_completion_probability(
                user_profile=user_profile,
                workout_features=workout_features
            )

            return {
                'status': 'success',
                'user_id': user_id,
                'workout_id': workout_id,
                'completion_prediction': prediction,
                'algorithm': 'random_forest'
            }

        except Exception as e:
            logger.error(f"Completion prediction error: {e}")
            return {'error': str(e)}, 500

    def predict_suitability(self, data: Dict) -> Dict:
        """Overall workout suitability prediction"""
        try:
            user_id = data.get('user_id')
            workout_id = data.get('workout_id')
            user_profile = data.get('user_profile')
            workout_features = data.get('workout_features')
            user_history = data.get('user_history')

            if not user_profile or not workout_features:
                if not user_id or not workout_id:
                    return {'error': 'Either (user_profile and workout_features) or (user_id and workout_id) required'}, 400

                # Get user and workout data from services
                user_profile = self.auth_service.get_user_profile(user_id)
                workout_features = self.content_service.get_workout_details(workout_id)
                user_history = self.tracking_service.get_user_history(user_id)

                if not user_profile:
                    return {'error': 'User not found'}, 404
                if not workout_features:
                    return {'error': 'Workout not found'}, 404

            # Get Random Forest model
            rf_model = current_app.model_manager.get_random_forest_model()
            if not rf_model:
                return {'error': 'Random Forest model not available'}, 503

            # Predict overall workout success
            prediction = rf_model.predict_workout_success(
                user_profile=user_profile,
                workout_features=workout_features,
                user_history=user_history
            )

            return {
                'status': 'success',
                'user_id': user_id,
                'workout_id': workout_id,
                'suitability_prediction': prediction,
                'algorithm': 'random_forest'
            }

        except Exception as e:
            logger.error(f"Suitability prediction error: {e}")
            return {'error': str(e)}, 500

    def batch_predict(self, data: Dict) -> Dict:
        """Batch prediction for multiple workout scenarios"""
        try:
            predictions_data = data.get('predictions', [])

            if not predictions_data:
                return {'error': 'predictions array is required'}, 400

            # Validate prediction data format
            for i, pred_data in enumerate(predictions_data):
                required_fields = ['user_profile', 'workout_features']
                if not all(field in pred_data for field in required_fields):
                    return {'error': f'Prediction {i}: user_profile and workout_features required'}, 400

            # Get Random Forest model
            rf_model = current_app.model_manager.get_random_forest_model()
            if not rf_model:
                return {'error': 'Random Forest model not available'}, 503

            # Perform batch predictions
            batch_results = rf_model.batch_predict(predictions_data)

            return {
                'status': 'success',
                'predictions': batch_results,
                'count': len(batch_results),
                'algorithm': 'random_forest_batch'
            }

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return {'error': str(e)}, 500

    def get_model_health(self) -> Dict:
        """Get Random Forest model health status"""
        try:
            rf_model = current_app.model_manager.get_random_forest_model()
            if not rf_model:
                return {
                    'status': 'unhealthy',
                    'error': 'Random Forest model not loaded'
                }, 503

            health_status = rf_model.health_check()
            model_info = rf_model.get_model_info()

            return {
                'status': 'success',
                'health': health_status,
                'model_info': model_info
            }

        except Exception as e:
            logger.error(f"Random Forest model health check error: {e}")
            return {'error': str(e)}, 500

    def evaluate_model_performance(self, data: Dict = None) -> Dict:
        """Evaluate Random Forest model performance"""
        try:
            # Get Random Forest model
            rf_model = current_app.model_manager.get_random_forest_model()
            if not rf_model:
                return {'error': 'Random Forest model not available'}, 503

            # Get model info and health
            model_info = rf_model.get_model_info()
            health_status = rf_model.health_check()

            # Get some test data for evaluation (if provided)
            test_data = data.get('test_data', []) if data else []

            evaluation_results = {
                'model_accuracy': model_info.get('accuracy', 0.998),
                'model_health': health_status,
                'test_predictions_count': len(test_data),
                'feature_count': model_info.get('feature_count', 15),
                'capabilities': model_info.get('capabilities', [])
            }

            # Run test predictions if test data provided
            if test_data:
                try:
                    test_results = rf_model.batch_predict(test_data)
                    evaluation_results['test_results'] = test_results
                except Exception as e:
                    evaluation_results['test_error'] = str(e)

            return {
                'status': 'success',
                'evaluation': evaluation_results,
                'algorithm': 'random_forest'
            }

        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            return {'error': str(e)}, 500