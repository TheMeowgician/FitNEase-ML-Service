"""
Hybrid Recommendation Controller
===============================

Controller for hybrid recommendation system operations
"""

import logging
from typing import Dict, List, Optional
from flask import current_app

from models.database_models import Recommendations, UserBehaviorPatterns
from services.auth_service import AuthService
from services.content_service import ContentService
from services.tracking_service import TrackingService

logger = logging.getLogger(__name__)

class HybridController:
    """Controller for hybrid recommendation operations"""

    def __init__(self):
        self.auth_service = AuthService()
        self.content_service = ContentService()
        self.tracking_service = TrackingService()

    def get_recommendations(self, user_id: int, data: Dict = None) -> Dict:
        """Main hybrid recommendations endpoint (PRIMARY ENDPOINT)"""
        try:
            num_recommendations = data.get('num_recommendations', 10) if data else 10
            content_weight = data.get('content_weight', 0.7) if data else 0.7
            collaborative_weight = data.get('collaborative_weight', 0.3) if data else 0.3
            auth_token = data.get('auth_token') if data else None

            # Validate weights
            if abs(content_weight + collaborative_weight - 1.0) > 0.01:
                return {'error': 'content_weight and collaborative_weight must sum to 1.0'}, 400

            # Get hybrid model
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {'error': 'Hybrid model not available'}, 503

            # Get user data from auth service
            user_data = self.auth_service.get_user_profile(user_id)

            # Get user history from tracking service
            user_history = self.tracking_service.get_user_history(user_id)

            # Get exercise data from content service
            exercises = self.content_service.get_exercise_attributes()

            # Generate hybrid recommendations
            recommendations = hybrid_model.get_recommendations(
                user_id=user_id,
                user_preferences=user_data.get('preferences') if user_data else None,
                num_recommendations=num_recommendations,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight
            )

            # FALLBACK: If hybrid returns no recommendations (insufficient collaborative data),
            # automatically fallback to pure content-based recommendations
            if len(recommendations) == 0:
                logger.warning(f"Hybrid model returned 0 recommendations for user {user_id}, falling back to content-based")
                from controllers.content_based_controller import ContentBasedController
                content_controller = ContentBasedController()
                fallback_data = {
                    'num_recommendations': num_recommendations,
                    'auth_token': auth_token
                }
                content_result = content_controller.get_user_recommendations(user_id, fallback_data)

                if isinstance(content_result, tuple):
                    return content_result

                # Return content-based recommendations with hybrid format
                return {
                    'status': 'success',
                    'user_id': user_id,
                    'recommendations': content_result.get('recommendations', []),
                    'count': content_result.get('count', 0),
                    'algorithm': 'hybrid_fallback_to_content',
                    'weights': {
                        'content_weight': 1.0,
                        'collaborative_weight': 0.0
                    },
                    'note': 'Insufficient collaborative data, using content-based recommendations'
                }

            # Save recommendations to database
            self._save_recommendations(user_id, recommendations, 'hybrid')

            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'algorithm': 'hybrid',
                'weights': {
                    'content_weight': content_weight,
                    'collaborative_weight': collaborative_weight
                }
            }

        except Exception as e:
            logger.error(f"Hybrid recommendations error: {e}")
            return {'error': str(e)}, 500

    def get_content_based_recommendations(self, data: Dict) -> Dict:
        """Get pure content-based recommendations via hybrid model"""
        try:
            exercise_name = data.get('exercise_name')
            user_preferences = data.get('user_preferences')
            num_recommendations = data.get('num_recommendations', 10)

            if not exercise_name and not user_preferences:
                return {'error': 'Either exercise_name or user_preferences required'}, 400

            # Get hybrid model
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {'error': 'Hybrid model not available'}, 503

            # Get content-based recommendations
            recommendations = hybrid_model.get_content_based_recommendations(
                exercise_name=exercise_name,
                user_preferences=user_preferences,
                num_recommendations=num_recommendations
            )

            return {
                'status': 'success',
                'recommendations': recommendations,
                'count': len(recommendations),
                'method': 'content_based_via_hybrid'
            }

        except Exception as e:
            logger.error(f"Hybrid content-based recommendations error: {e}")
            return {'error': str(e)}, 500

    def get_collaborative_recommendations(self, data: Dict) -> Dict:
        """Get pure collaborative recommendations via hybrid model"""
        try:
            user_id = data.get('user_id')
            num_recommendations = data.get('num_recommendations', 10)

            if not user_id:
                return {'error': 'user_id is required'}, 400

            # Get hybrid model
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {'error': 'Hybrid model not available'}, 503

            # Get collaborative recommendations
            recommendations = hybrid_model.get_collaborative_recommendations(
                user_id=user_id,
                num_recommendations=num_recommendations
            )

            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'method': 'collaborative_via_hybrid'
            }

        except Exception as e:
            logger.error(f"Hybrid collaborative recommendations error: {e}")
            return {'error': str(e)}, 500

    def get_detailed_scores(self, user_id: int) -> Dict:
        """Get detailed hybrid scoring breakdown"""
        try:
            # Get hybrid model
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {'error': 'Hybrid model not available'}, 503

            # Get user behavior insights
            insights = hybrid_model.get_user_behavior_insights(user_id)

            # Get model info
            model_info = hybrid_model.get_model_info()

            return {
                'status': 'success',
                'user_id': user_id,
                'behavior_insights': insights,
                'model_info': model_info
            }

        except Exception as e:
            logger.error(f"Hybrid detailed scores error: {e}")
            return {'error': str(e)}, 500

    def calculate_exercise_similarity(self, data: Dict) -> Dict:
        """Calculate exercise similarity via hybrid model"""
        try:
            exercise1_name = data.get('exercise1_name')
            exercise2_name = data.get('exercise2_name')
            similarity_metric = data.get('similarity_metric', 'cosine')

            if not exercise1_name or not exercise2_name:
                return {'error': 'Both exercise1_name and exercise2_name required'}, 400

            # Get hybrid model
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {'error': 'Hybrid model not available'}, 503

            # Calculate similarity
            similarity_score = hybrid_model.calculate_exercise_similarity(
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

    def get_model_health(self) -> Dict:
        """Get hybrid model health status"""
        try:
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {
                    'status': 'unhealthy',
                    'error': 'Hybrid model not loaded'
                }, 503

            health_status = hybrid_model.health_check()
            model_info = hybrid_model.get_model_info()

            return {
                'status': 'success',
                'health': health_status,
                'model_info': model_info
            }

        except Exception as e:
            logger.error(f"Hybrid model health check error: {e}")
            return {'error': str(e)}, 500

    def update_user_patterns(self, data: Dict) -> Dict:
        """Update user behavioral patterns"""
        try:
            patterns = data.get('patterns', [])
            updated_count = 0

            for pattern_data in patterns:
                user_id = pattern_data.get('user_id')
                if not user_id:
                    continue

                # Create or update behavior pattern
                pattern = UserBehaviorPatterns(**pattern_data)
                if pattern.save():
                    updated_count += 1

            return {
                'status': 'success',
                'updated_count': updated_count,
                'total_patterns': len(patterns)
            }

        except Exception as e:
            logger.error(f"User patterns update error: {e}")
            return {'error': str(e)}, 500

    def _save_recommendations(self, user_id: int, recommendations: List[Dict], algorithm: str):
        """Save recommendations to database"""
        try:
            for rec in recommendations:
                recommendation = Recommendations(
                    user_id=user_id,
                    workout_id=rec.get('exercise_id'),
                    recommendation_score=rec.get('hybrid_score', rec.get('rating', 0.0)),
                    algorithm_used=algorithm,
                    recommendation_type=algorithm,
                    content_based_score=rec.get('content_score', 0.0),
                    collaborative_score=rec.get('collaborative_score', 0.0),
                    recommendation_reason=f"Hybrid recommendation combining content-based and collaborative filtering"
                )
                recommendation.save()

        except Exception as e:
            logger.warning(f"Failed to save hybrid recommendations: {e}")