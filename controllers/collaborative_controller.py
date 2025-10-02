"""
Collaborative Filtering Controller
=================================

Controller for collaborative filtering operations
"""

import logging
from typing import Dict, List, Optional
from flask import current_app

from models.database_models import CollaborativeScores, Recommendations
from services.tracking_service import TrackingService

logger = logging.getLogger(__name__)

class CollaborativeController:
    """Controller for collaborative filtering operations"""

    def __init__(self):
        self.tracking_service = TrackingService()

    def calculate_user_similarity(self, data: Dict) -> Dict:
        """Calculate user similarity scores"""
        try:
            user_id = data.get('user_id')
            top_k = data.get('top_k', 20)

            if not user_id:
                return {'error': 'user_id is required'}, 400

            # Get collaborative model
            collaborative_model = current_app.model_manager.get_collaborative_model()
            if not collaborative_model:
                return {'error': 'Collaborative model not available'}, 503

            # Get user workout history
            user_history = self.tracking_service.get_user_history(user_id)
            if not user_history:
                return {'error': 'No user history found'}, 404

            # Get similar users
            similar_users = collaborative_model.get_user_similarities(
                user_id=user_id,
                top_k=top_k
            )

            # Save collaborative scores
            self._save_collaborative_scores(user_id, similar_users)

            return {
                'status': 'success',
                'user_id': user_id,
                'similar_users': similar_users,
                'count': len(similar_users),
                'algorithm': 'collaborative_filtering'
            }

        except Exception as e:
            logger.error(f"Collaborative similarity calculation error: {e}")
            return {'error': str(e)}, 500

    def get_user_recommendations(self, user_id: int, data: Dict = None) -> Dict:
        """Get collaborative recommendations for user - REQUIRES RATING DATA"""
        try:
            num_recommendations = data.get('num_recommendations', 10) if data else 10

            # Get collaborative model
            collaborative_model = current_app.model_manager.get_collaborative_model()
            if not collaborative_model:
                return {'error': 'Collaborative model not available'}, 503

            # Get user ratings from tracking service (REQUIRED for collaborative filtering)
            user_ratings = self.tracking_service.get_user_ratings(user_id)

            # Check if user has sufficient rating data
            if not user_ratings or len(user_ratings) == 0:
                logger.warning(f"Collaborative filtering unavailable for user {user_id}: No ratings found")
                return {
                    'status': 'unavailable',
                    'user_id': user_id,
                    'recommendations': [],
                    'count': 0,
                    'algorithm': 'collaborative_filtering',
                    'message': 'Collaborative filtering requires rating data. Please complete workouts and rate exercises to enable this feature.',
                    'required_actions': [
                        'Complete at least 10 workout sessions',
                        'Rate at least 5 exercises (thumbs up/down)',
                        'Try different exercise types for better recommendations'
                    ]
                }

            # Get collaborative recommendations with REAL rating data
            recommendations = collaborative_model.get_recommendations(
                user_id=user_id,
                num_recommendations=num_recommendations,
                user_ratings=user_ratings  # Pass real rating data
            )

            # Save recommendations to database
            self._save_recommendations(user_id, recommendations, 'collaborative')

            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'algorithm': 'collaborative_filtering'
            }

        except Exception as e:
            logger.error(f"Collaborative recommendations error: {e}")
            return {'error': str(e)}, 500

    def get_model_health(self) -> Dict:
        """Get collaborative model health status"""
        try:
            collaborative_model = current_app.model_manager.get_collaborative_model()
            if not collaborative_model:
                return {
                    'status': 'unhealthy',
                    'error': 'Collaborative model not loaded'
                }, 503

            health_status = collaborative_model.health_check()

            return {
                'status': 'success',
                'health': health_status,
                'note': 'Collaborative filtering functionality integrated into hybrid model'
            }

        except Exception as e:
            logger.error(f"Collaborative model health check error: {e}")
            return {'error': str(e)}, 500

    def update_user_behavior(self, data: Dict) -> Dict:
        """Update user behavior data for collaborative filtering"""
        try:
            user_id = data.get('user_id')
            behavior_data = data.get('behavior_data', {})

            if not user_id:
                return {'error': 'user_id is required'}, 400

            # Update behavioral data through tracking service
            success = self.tracking_service.update_behavioral_data(user_id, behavior_data)

            if success:
                return {
                    'status': 'success',
                    'user_id': user_id,
                    'message': 'User behavior data updated successfully'
                }
            else:
                return {'error': 'Failed to update user behavior data'}, 500

        except Exception as e:
            logger.error(f"User behavior update error: {e}")
            return {'error': str(e)}, 500

    def _save_recommendations(self, user_id: int, recommendations: List[Dict], algorithm: str):
        """Save recommendations to database"""
        try:
            for rec in recommendations:
                recommendation = Recommendations(
                    user_id=user_id,
                    workout_id=rec.get('exercise_id'),
                    recommendation_score=rec.get('collaborative_score', 0.0),
                    algorithm_used=algorithm,
                    recommendation_type=algorithm,
                    content_based_score=0.0,
                    collaborative_score=rec.get('collaborative_score', 0.0),
                    recommendation_reason=f"Collaborative recommendation based on similar users"
                )
                recommendation.save()

        except Exception as e:
            logger.warning(f"Failed to save collaborative recommendations: {e}")

    def _save_collaborative_scores(self, user_id: int, similar_users: List[Dict]):
        """Save collaborative scores to database"""
        try:
            for similar_user in similar_users:
                collaborative_score = CollaborativeScores(
                    user_id=user_id,
                    similar_user_id=similar_user.get('user_id'),
                    similarity_score=similar_user.get('similarity_score', 0.0),
                    common_workouts_count=similar_user.get('common_workouts', 0),
                    rating_correlation=similar_user.get('rating_correlation', 0.0)
                )
                collaborative_score.save()

        except Exception as e:
            logger.warning(f"Failed to save collaborative scores: {e}")