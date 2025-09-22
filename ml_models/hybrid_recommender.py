"""
Hybrid Recommender Model
=======================

Wrapper for the trained hybrid filtering model that combines
content-based and collaborative filtering approaches
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class HybridRecommender:
    """Hybrid recommendation model wrapper"""

    def __init__(self, model_data: Optional[Dict] = None):
        """Initialize with loaded model data"""
        self.feature_engineer = None
        self.content_based_recommender = None
        self.collaborative_recommender = None
        self.hybrid_recommender = None
        self.model_info = {}
        self.model_data = None

        if model_data:
            self._load_model_data(model_data)
        else:
            self._create_fallback_model()

    def _load_model_data(self, model_data: Dict):
        """Load model data"""
        try:
            self.model_data = model_data

            # Check if model_data contains a FinalHybridRecommender instance
            if 'recommender' in model_data:
                self.hybrid_recommender = model_data.get('recommender')
                self.model_info = {
                    'model_type': model_data.get('model_type', 'hybrid'),
                    'version': model_data.get('version', '1.0'),
                    'timestamp': model_data.get('timestamp'),
                    'performance': model_data.get('performance', {}),
                    'weights': model_data.get('weights', {}),
                    'configuration': model_data.get('configuration', {})
                }
                logger.info("Hybrid recommender loaded successfully from trained model")
            else:
                # Legacy format with separate components
                self.feature_engineer = model_data.get('feature_engineer')
                self.content_based_recommender = model_data.get('content_based_recommender')
                self.collaborative_recommender = model_data.get('collaborative_recommender')
                self.model_info = model_data.get('model_info', {})
                self.hybrid_recommender = None

                if not self.content_based_recommender:
                    logger.warning("Content-based recommender not found in model data, using fallback")
                    self._create_fallback_model()
                else:
                    logger.info("Hybrid recommender initialized successfully from legacy format")

        except Exception as e:
            logger.error(f"Error loading hybrid model data: {e}")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Create a fallback hybrid model"""
        try:
            from .custom_classes import HybridRecommenderModel
            self.hybrid_model = HybridRecommenderModel()
            self.hybrid_model.fit()  # Fit with dummy data
            self.model_info = {'status': 'fallback', 'content_weight': 0.6, 'collaborative_weight': 0.4}
            logger.info("Hybrid fallback model created")
        except Exception as e:
            logger.error(f"Error creating hybrid fallback model: {e}")

    def set_model_data(self, model_data: Dict):
        """Set model data after initialization"""
        self._load_model_data(model_data)

    def get_recommendations(self, user_id: int = None, exercise_name: str = None,
                          user_preferences: Dict = None, num_recommendations: int = 10,
                          content_weight: float = 0.7, collaborative_weight: float = 0.3) -> List[Dict]:
        """Get hybrid recommendations combining content-based and collaborative filtering"""
        try:
            # Use trained hybrid recommender if available
            if self.hybrid_recommender and hasattr(self.hybrid_recommender, 'recommend'):
                try:
                    # Use the loaded FinalHybridRecommender
                    recommendations = self.hybrid_recommender.recommend(
                        user_id=user_id or 1,  # Provide default user_id if none given
                        n_recommendations=num_recommendations,
                        user_profile=user_preferences
                    )

                    # Format recommendations for API response
                    formatted_recs = []
                    for rec in recommendations:
                        formatted_rec = {
                            'exercise_id': rec.get('item_id', rec.get('exercise_id')),
                            'exercise_name': rec.get('exercise_name', f"Exercise_{rec.get('item_id', 'Unknown')}"),
                            'target_muscle_group': rec.get('target_muscle_group', 'unknown'),
                            'difficulty_level': rec.get('difficulty_level', 2),
                            'equipment_needed': rec.get('equipment_needed', 'unknown'),
                            'hybrid_score': rec.get('score', 0.5),
                            'content_score': rec.get('content_weight', content_weight),
                            'collaborative_score': rec.get('collaborative_weight', collaborative_weight),
                            'calories_per_minute': rec.get('calories_per_minute', 5),
                            'duration_seconds': rec.get('duration_seconds', 1800),
                            'recommendation_type': rec.get('recommendation_type', 'hybrid'),
                            'weights_used': {
                                'content_weight': rec.get('content_weight', content_weight),
                                'collaborative_weight': rec.get('collaborative_weight', collaborative_weight)
                            }
                        }
                        formatted_recs.append(formatted_rec)

                    logger.info(f"Generated {len(formatted_recs)} hybrid recommendations using trained model")
                    return formatted_recs

                except Exception as e:
                    logger.error(f"Error using trained hybrid recommender: {e}")
                    # Fall back to legacy method below

            # Legacy method: combine separate recommenders
            recommendations = []

            # Get content-based recommendations
            content_recs = []
            if self.content_based_recommender:
                if exercise_name:
                    content_recs = self.content_based_recommender.get_recommendations(
                        exercise_name=exercise_name,
                        num_recommendations=num_recommendations * 2
                    )
                elif user_preferences:
                    content_recs = self.content_based_recommender.get_user_based_recommendations(
                        user_preferences=user_preferences,
                        num_recommendations=num_recommendations * 2
                    )

            # Get collaborative recommendations if available
            collaborative_recs = []
            if self.collaborative_recommender and user_id:
                collaborative_recs = self.collaborative_recommender.get_recommendations(
                    user_id=user_id,
                    num_recommendations=num_recommendations * 2
                )

            # Combine recommendations using weighted scoring
            exercise_scores = {}

            # Process content-based recommendations
            for rec in content_recs:
                exercise_id = rec.get('exercise_id')
                if exercise_id:
                    score = rec.get('similarity_score', rec.get('preference_score', 0))
                    exercise_scores[exercise_id] = {
                        'content_score': score,
                        'collaborative_score': 0.0,
                        'exercise_data': rec
                    }

            # Process collaborative recommendations
            for rec in collaborative_recs:
                exercise_id = rec.get('exercise_id')
                if exercise_id:
                    score = rec.get('collaborative_score', 0)
                    if exercise_id in exercise_scores:
                        exercise_scores[exercise_id]['collaborative_score'] = score
                    else:
                        exercise_scores[exercise_id] = {
                            'content_score': 0.0,
                            'collaborative_score': score,
                            'exercise_data': rec
                        }

            # Calculate hybrid scores and format recommendations
            for exercise_id, scores in exercise_scores.items():
                hybrid_score = (
                    content_weight * scores['content_score'] +
                    collaborative_weight * scores['collaborative_score']
                )

                exercise_data = scores['exercise_data']
                hybrid_rec = {
                    'exercise_id': exercise_id,
                    'exercise_name': exercise_data.get('exercise_name'),
                    'target_muscle_group': exercise_data.get('target_muscle_group'),
                    'difficulty_level': exercise_data.get('difficulty_level'),
                    'equipment_needed': exercise_data.get('equipment_needed'),
                    'hybrid_score': hybrid_score,
                    'content_score': scores['content_score'],
                    'collaborative_score': scores['collaborative_score'],
                    'calories_per_minute': exercise_data.get('calories_per_minute'),
                    'duration_seconds': exercise_data.get('duration_seconds'),
                    'recommendation_type': 'hybrid',
                    'weights_used': {
                        'content_weight': content_weight,
                        'collaborative_weight': collaborative_weight
                    }
                }
                recommendations.append(hybrid_rec)

            # Sort by hybrid score and return top recommendations
            recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
            final_recs = recommendations[:num_recommendations]

            logger.info(f"Generated {len(final_recs)} hybrid recommendations "
                       f"(content: {len(content_recs)}, collaborative: {len(collaborative_recs)})")
            return final_recs

        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {e}")
            return []

    def get_content_based_recommendations(self, exercise_name: str = None,
                                        user_preferences: Dict = None,
                                        num_recommendations: int = 10) -> List[Dict]:
        """Get pure content-based recommendations"""
        try:
            if exercise_name:
                return self.content_based_recommender.get_recommendations(
                    exercise_name=exercise_name,
                    num_recommendations=num_recommendations
                )
            elif user_preferences:
                return self.content_based_recommender.get_user_based_recommendations(
                    user_preferences=user_preferences,
                    num_recommendations=num_recommendations
                )
            else:
                logger.warning("No exercise name or user preferences provided")
                return []

        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []

    def get_collaborative_recommendations(self, user_id: int,
                                        num_recommendations: int = 10) -> List[Dict]:
        """Get pure collaborative filtering recommendations"""
        try:
            if not self.collaborative_recommender:
                logger.warning("Collaborative recommender not available")
                return []

            return self.collaborative_recommender.get_recommendations(
                user_id=user_id,
                num_recommendations=num_recommendations
            )

        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []

    def calculate_exercise_similarity(self, exercise1_name: str, exercise2_name: str,
                                    similarity_metric: str = 'cosine') -> float:
        """Calculate similarity between exercises using content-based model"""
        try:
            return self.content_based_recommender.calculate_exercise_similarity(
                exercise1_name, exercise2_name, similarity_metric
            )

        except Exception as e:
            logger.error(f"Error calculating exercise similarity: {e}")
            return 0.0

    def get_user_behavior_insights(self, user_id: int) -> Dict:
        """Get insights about user behavior patterns"""
        try:
            insights = {
                'user_id': user_id,
                'model_type': 'hybrid',
                'available_methods': []
            }

            if self.content_based_recommender:
                insights['available_methods'].append('content_based')
                insights['content_based_available'] = True

            if self.collaborative_recommender:
                insights['available_methods'].append('collaborative')
                insights['collaborative_available'] = True
            else:
                insights['collaborative_available'] = False

            insights['hybrid_available'] = len(insights['available_methods']) > 1

            return insights

        except Exception as e:
            logger.error(f"Error getting user behavior insights: {e}")
            return {'error': str(e)}

    def get_model_info(self) -> Dict:
        """Get model information and metadata"""
        return {
            'model_type': 'hybrid_filtering',
            'version': self.model_info.get('version', '1.0'),
            'training_date': self.model_info.get('training_date'),
            'components': {
                'content_based': self.content_based_recommender is not None,
                'collaborative': self.collaborative_recommender is not None,
                'feature_engineer': self.feature_engineer is not None
            },
            'default_weights': {
                'content_weight': 0.7,
                'collaborative_weight': 0.3
            },
            'capabilities': [
                'exercise_similarity',
                'user_preferences',
                'hybrid_scoring',
                'content_based_fallback'
            ],
            'status': 'loaded'
        }

    def health_check(self) -> Dict:
        """Check model health and status"""
        try:
            # Check if we have the trained hybrid recommender
            hybrid_healthy = (
                self.hybrid_recommender is not None and
                hasattr(self.hybrid_recommender, 'recommend')
            )

            # Legacy: Check if we have separate content-based recommender
            content_healthy = (
                self.content_based_recommender is not None and
                hasattr(self.content_based_recommender, 'similarity_matrices')
            )

            collaborative_healthy = (
                self.collaborative_recommender is not None or
                True  # Collaborative is optional in hybrid model
            )

            # Model is healthy if we have either the trained hybrid model OR the separate components
            is_healthy = hybrid_healthy or (content_healthy and collaborative_healthy)
            can_recommend = hybrid_healthy or content_healthy

            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'hybrid_model_loaded': self.hybrid_recommender is not None,
                'content_based_loaded': self.content_based_recommender is not None,
                'collaborative_loaded': self.collaborative_recommender is not None,
                'feature_engineer_loaded': self.feature_engineer is not None,
                'can_recommend': can_recommend,
                'hybrid_capable': hybrid_healthy or (content_healthy and (self.collaborative_recommender is not None)),
                'model_type': 'trained_hybrid' if hybrid_healthy else 'component_hybrid'
            }

        except Exception as e:
            logger.error(f"Error in hybrid model health check: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }