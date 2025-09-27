"""
Content-Based Filtering Controller
=================================

Controller for content-based recommendation operations
"""

import logging
from typing import Dict, List, Optional
from flask import current_app

from models.database_models import ContentBasedScores, Recommendations
from services.auth_service import AuthService
from services.content_service import ContentService

logger = logging.getLogger(__name__)

class ContentBasedController:
    """Controller for content-based filtering operations"""

    def __init__(self):
        self.auth_service = AuthService()
        self.content_service = ContentService()

    def calculate_similarity(self, data: Dict) -> Dict:
        """Calculate exercise similarity scores"""
        try:
            exercise_name = data.get('exercise_name')
            num_recommendations = data.get('num_recommendations', 10)
            similarity_metric = data.get('similarity_metric', 'cosine')

            if not exercise_name:
                return {'error': 'exercise_name is required'}, 400

            # Get content-based model
            content_model = current_app.model_manager.get_content_based_model()
            if not content_model:
                return {'error': 'Content-based model not available'}, 503

            # Get recommendations
            recommendations = content_model.get_recommendations(
                exercise_name=exercise_name,
                num_recommendations=num_recommendations,
                similarity_metric=similarity_metric
            )

            return {
                'status': 'success',
                'exercise_name': exercise_name,
                'recommendations': recommendations,
                'count': len(recommendations),
                'similarity_metric': similarity_metric
            }

        except Exception as e:
            logger.error(f"Content similarity calculation error: {e}")
            return {'error': str(e)}, 500

    def get_user_recommendations(self, user_id: int, data: Dict = None) -> Dict:
        """Get content-based recommendations for user"""
        try:
            num_recommendations = data.get('num_recommendations', 10) if data else 10
            auth_token = data.get('auth_token') if data else None

            # Get content-based model
            content_model = current_app.model_manager.get_content_based_model()
            if not content_model:
                return {'error': 'Content-based model not available'}, 503

            # Get user preferences from auth service (with token)
            user_data = None
            if auth_token:
                user_data = self.auth_service.get_user_profile_with_token(user_id, auth_token)

            if not user_data:
                # Fallback: use basic user preferences
                user_data = {'preferences': {}}

            # Get exercise data from content service (with token)
            exercise_data = None
            if auth_token:
                exercise_data = self.content_service.get_exercise_attributes_with_token(auth_token)

            # Get user-based recommendations
            recommendations = content_model.get_user_recommendations(
                user_preferences=user_data.get('preferences', {}),
                num_recommendations=num_recommendations,
                exercise_data=exercise_data
            )

            # Save recommendations to database
            self._save_recommendations(user_id, recommendations, 'content_based')

            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'algorithm': 'content_based'
            }

        except Exception as e:
            logger.error(f"Content-based user recommendations error: {e}")
            return {'error': str(e)}, 500

    def get_exercise_similarity(self, data: Dict) -> Dict:
        """Calculate similarity between two exercises"""
        try:
            exercise1_name = data.get('exercise1_name')
            exercise2_name = data.get('exercise2_name')
            similarity_metric = data.get('similarity_metric', 'cosine')

            if not exercise1_name or not exercise2_name:
                return {'error': 'Both exercise1_name and exercise2_name are required'}, 400

            # Get content-based model
            content_model = current_app.model_manager.get_content_based_model()
            if not content_model:
                return {'error': 'Content-based model not available'}, 503

            # Calculate similarity
            similarity_score = content_model.calculate_exercise_similarity(
                exercise1_name, exercise2_name, similarity_metric
            )

            return {
                'status': 'success',
                'exercise1': exercise1_name,
                'exercise2': exercise2_name,
                'similarity_score': similarity_score,
                'similarity_metric': similarity_metric
            }

        except Exception as e:
            logger.error(f"Exercise similarity calculation error: {e}")
            return {'error': str(e)}, 500

    def get_similar_exercises(self, data: Dict) -> Dict:
        """Get exercises similar to given exercise above threshold"""
        try:
            exercise_name = data.get('exercise_name')
            threshold = data.get('threshold', 0.7)
            similarity_metric = data.get('similarity_metric', 'cosine')

            if not exercise_name:
                return {'error': 'exercise_name is required'}, 400

            # Get content-based model
            content_model = current_app.model_manager.get_content_based_model()
            if not content_model:
                return {'error': 'Content-based model not available'}, 503

            # Get similar exercises
            similar_exercises = content_model.get_similar_exercises(
                exercise_name=exercise_name,
                threshold=threshold,
                similarity_metric=similarity_metric
            )

            return {
                'status': 'success',
                'exercise_name': exercise_name,
                'threshold': threshold,
                'similar_exercises': similar_exercises,
                'count': len(similar_exercises),
                'similarity_metric': similarity_metric
            }

        except Exception as e:
            logger.error(f"Similar exercises retrieval error: {e}")
            return {'error': str(e)}, 500

    def get_model_health(self) -> Dict:
        """Get content-based model health status"""
        try:
            content_model = current_app.model_manager.get_content_based_model()
            if not content_model:
                return {
                    'status': 'unhealthy',
                    'error': 'Content-based model not loaded'
                }, 503

            health_status = content_model.health_check()
            model_info = content_model.get_model_info()

            return {
                'status': 'success',
                'health': health_status,
                'model_info': model_info
            }

        except Exception as e:
            logger.error(f"Content-based model health check error: {e}")
            return {'error': str(e)}, 500

    def _save_recommendations(self, user_id: int, recommendations: List[Dict], algorithm: str):
        """Save recommendations to database"""
        try:
            for rec in recommendations:
                recommendation = Recommendations(
                    user_id=user_id,
                    workout_id=rec.get('exercise_id'),
                    recommendation_score=rec.get('similarity_score', rec.get('preference_score', 0.0)),
                    algorithm_used=algorithm,
                    recommendation_type=algorithm,
                    content_based_score=rec.get('similarity_score', rec.get('preference_score', 0.0)),
                    collaborative_score=0.0,
                    recommendation_reason=f"Content-based recommendation based on exercise similarity"
                )
                recommendation.save()

        except Exception as e:
            logger.warning(f"Failed to save content-based recommendations: {e}")

    def _save_content_scores(self, user_id: int, exercise_id: int, similarity_score: float, feature_vector: Dict):
        """Save content-based scores to database"""
        try:
            content_score = ContentBasedScores(
                user_id=user_id,
                exercise_id=exercise_id,
                similarity_score=similarity_score,
                feature_vector=feature_vector
            )
            content_score.save()

        except Exception as e:
            logger.warning(f"Failed to save content-based score: {e}")