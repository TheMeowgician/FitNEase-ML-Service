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

    def _calculate_dynamic_weights(self, rating_count: int) -> tuple:
        """
        Calculate dynamic weights based on user rating count.
        Industry best practice: Smooth transition like Netflix/Spotify.

        Args:
            rating_count: Number of exercise ratings the user has

        Returns:
            (content_weight, collaborative_weight, confidence_level)
        """
        if rating_count == 0:
            # Brand new user - pure content-based
            return (1.0, 0.0, "new_user")
        elif rating_count < 3:
            # Very early - tiny CF signal
            return (0.95, 0.05, "low")
        elif rating_count < 5:
            # Building profile - growing CF
            return (0.85, 0.15, "low-medium")
        elif rating_count < 10:
            # Good data - moderate CF
            return (0.75, 0.25, "medium")
        elif rating_count < 15:
            # Strong data - balanced hybrid
            return (0.65, 0.35, "medium-high")
        elif rating_count < 20:
            # Very strong data - CF dominant
            return (0.55, 0.45, "high")
        else:
            # Excellent data - equal weight (Netflix-style)
            return (0.5, 0.5, "very-high")

    def _filter_by_fitness_level(self, recommendations: List[Dict], user_fitness_level: str,
                                  rf_predictor=None, num_recommendations: int = 10, user_profile: Dict = None) -> List[Dict]:
        """
        Filter and rank recommendations by user's fitness level using Random Forest ML model.

        Two-stage process:
        1. Rule-based filtering by difficulty level (safety filter)
        2. ML-based ranking using Random Forest appropriateness scores

        Fitness level mapping:
        - beginner: difficulty_level = 1
        - intermediate: difficulty_level = 2
        - advanced/expert: difficulty_level = 3
        """
        # Map fitness level to difficulty
        fitness_to_difficulty = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 3
        }

        target_difficulty = fitness_to_difficulty.get(user_fitness_level.lower(), 1)
        logger.info(f"Filtering recommendations for fitness level '{user_fitness_level}' (difficulty: {target_difficulty})")

        # STAGE 1: Rule-based difficulty filtering (safety filter)
        # Filter exact match first (highest priority)
        exact_match = [r for r in recommendations if r.get('difficulty_level') == target_difficulty]

        # DEBUG: Log difficulty distribution of incoming recommendations
        difficulty_counts = {}
        for r in recommendations:
            d = r.get('difficulty_level', 'unknown')
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        logger.info(f"[DEBUG] Incoming recommendations difficulty distribution: {difficulty_counts}")
        logger.info(f"[DEBUG] Target difficulty: {target_difficulty}, Exact matches: {len(exact_match)}")

        # STRICT POLICY FOR BEGINNERS: Beginners should ONLY get beginner exercises
        if target_difficulty == 1:
            filtered_recommendations = exact_match
            if len(filtered_recommendations) < num_recommendations:
                logger.warning(f"Only found {len(filtered_recommendations)} beginner exercises out of {num_recommendations} requested!")
                logger.warning(f"This indicates the content-based model is not prioritizing beginner exercises.")
        else:
            # For intermediate/advanced users, allow flexible ±1 difficulty matching
            if len(exact_match) >= num_recommendations:
                filtered_recommendations = exact_match
                logger.info(f"Found {len(exact_match)} exact difficulty matches")
            else:
                # Allow one level up for variety (but not easier)
                if target_difficulty < 3:
                    filtered_recommendations = [r for r in recommendations
                                    if r.get('difficulty_level') in [target_difficulty, target_difficulty + 1]]
                else:
                    # For advanced users, allow one level down too
                    filtered_recommendations = [r for r in recommendations
                                    if r.get('difficulty_level') in [target_difficulty - 1, target_difficulty]]
                logger.info(f"Found {len(filtered_recommendations)} difficulty matches (exact + ±1 level)")

        # DEBUG: Log what's being returned
        filtered_difficulty_counts = {}
        for r in filtered_recommendations:
            d = r.get('difficulty_level', 'unknown')
            filtered_difficulty_counts[d] = filtered_difficulty_counts.get(d, 0) + 1
        logger.info(f"[DEBUG] After filtering - difficulty distribution: {filtered_difficulty_counts}")

        # If no recommendations after filtering, return empty list
        if not filtered_recommendations:
            return []

        # STAGE 2: Random Forest ML-based ranking (if available)
        if rf_predictor and user_profile:
            try:
                logger.info(f"Applying Random Forest ML ranking to {len(filtered_recommendations)} filtered recommendations...")

                # Score each recommendation using Random Forest
                scored_recommendations = []
                for rec in filtered_recommendations:
                    # Prepare workout features for RF prediction
                    workout_features = {
                        'difficulty_level': rec.get('difficulty_level', target_difficulty),
                        'target_muscle_group': rec.get('target_muscle_group', 'core'),
                        'exercise_category': rec.get('exercise_category', 'strength'),
                        'equipment_needed': rec.get('equipment_needed', 'bodyweight'),
                        'estimated_duration_minutes': rec.get('duration_seconds', 1800) / 60,
                        'calories_per_minute': rec.get('calories_per_minute', 5.0),
                        'exercise_intensity': rec.get('difficulty_level', target_difficulty)
                    }

                    # Get RF appropriateness prediction
                    rf_prediction = rf_predictor.predict_difficulty_appropriateness(user_profile, workout_features)

                    # Add RF scores to recommendation
                    rec['rf_appropriateness_score'] = rf_prediction.get('appropriateness_score', 0.5)
                    rec['rf_difficulty_rating'] = rf_prediction.get('difficulty_rating', 'unknown')
                    rec['rf_suitability_class'] = rf_prediction.get('suitability_class', 1)
                    rec['rf_confidence'] = rf_prediction.get('suitability_confidence', 0.5)

                    scored_recommendations.append(rec)

                # Sort by RF appropriateness score (highest first)
                scored_recommendations.sort(key=lambda x: x.get('rf_appropriateness_score', 0.5), reverse=True)

                # Log RF ranking results
                top_scores = [f"{r.get('exercise_name', 'Unknown')[:20]}: {r.get('rf_appropriateness_score', 0):.3f}"
                             for r in scored_recommendations[:3]]
                logger.info(f"Random Forest ranking complete. Top 3 scores: {top_scores}")

                logger.info(f"After RF ranking: Returning top {num_recommendations} recommendations")
                return scored_recommendations[:num_recommendations]

            except Exception as e:
                logger.error(f"Error applying Random Forest ranking: {e}")
                logger.warning("Falling back to rule-based filtering without ML ranking")
                # Fall back to rule-based filtering
                return filtered_recommendations[:num_recommendations]
        else:
            # No RF predictor available, use rule-based filtering only
            if not rf_predictor:
                logger.warning("Random Forest predictor not available, using rule-based filtering only")
            if not user_profile:
                logger.warning("User profile not available for RF prediction, using rule-based filtering only")

            logger.info(f"After filtering: {len(filtered_recommendations)} recommendations matching fitness level '{user_fitness_level}'")
            return filtered_recommendations[:num_recommendations]

    def get_recommendations(self, user_id: int, data: Dict = None) -> Dict:
        """Main hybrid recommendations endpoint (PRIMARY ENDPOINT) - Netflix-style dynamic weighting"""
        try:
            num_recommendations = data.get('num_recommendations', 10) if data else 10
            auth_token = data.get('auth_token') if data else None

            # Get hybrid model
            hybrid_model = current_app.model_manager.get_hybrid_model()
            if not hybrid_model:
                return {'error': 'Hybrid model not available'}, 503

            # Get Random Forest predictor for fitness-level filtering
            rf_predictor = current_app.model_manager.get_random_forest_model()

            # Get user data from auth service
            user_data = self.auth_service.get_user_profile(user_id)

            # Extract fitness level (default to beginner for new users)
            user_fitness_level = user_data.get('fitness_level', 'beginner') if user_data else 'beginner'
            logger.info(f"User {user_id} fitness level: {user_fitness_level}")

            # Get user history from tracking service
            user_history = self.tracking_service.get_user_history(user_id)

            # Get user exercise ratings to calculate dynamic weights
            user_ratings = self.tracking_service.get_user_exercise_ratings(user_id)
            user_rating_count = len(user_ratings) if user_ratings else 0

            # Calculate dynamic weights based on rating count (Netflix/Spotify approach)
            # Allow manual override if weights are explicitly provided in request
            if data and 'content_weight' in data and 'collaborative_weight' in data:
                content_weight = float(data['content_weight'])
                collaborative_weight = float(data['collaborative_weight'])
                confidence_level = "manual"
                logger.info(f"Using manually specified weights: Content={content_weight}, CF={collaborative_weight}")

                # Validate manual weights
                if abs(content_weight + collaborative_weight - 1.0) > 0.01:
                    return {'error': 'content_weight and collaborative_weight must sum to 1.0'}, 400
            else:
                # Use dynamic weights based on user rating count
                content_weight, collaborative_weight, confidence_level = self._calculate_dynamic_weights(user_rating_count)
                logger.info(f"User {user_id} has {user_rating_count} ratings - Dynamic weights: Content={content_weight}, CF={collaborative_weight} (confidence: {confidence_level})")

            # Determine algorithm display based on confidence level
            if confidence_level == "new_user":
                algorithm_display = "Content"
                algorithm_name = "hybrid_content_only"
            else:
                algorithm_display = "Hybrid"
                algorithm_name = "hybrid"

            # Get exercise data from content service
            exercises = self.content_service.get_exercise_attributes()

            # Generate hybrid recommendations with dynamic weights
            # Request more recommendations than needed to have buffer for filtering
            # Use 10x buffer for new/beginner users to ensure we get difficulty=1 exercises
            buffer_multiplier = 10 if user_rating_count < 5 else 5
            raw_recommendations = hybrid_model.get_recommendations(
                user_id=user_id,
                user_preferences=user_data.get('preferences') if user_data else None,
                num_recommendations=num_recommendations * buffer_multiplier,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight
            )

            # Filter recommendations by user's fitness level
            logger.info(f"Got {len(raw_recommendations)} raw recommendations, filtering by fitness level...")

            # Prepare user profile for Random Forest prediction
            rf_user_profile = {
                'fitness_level': user_fitness_level,
                'age': user_data.get('age', 30),
                'bmi': user_data.get('bmi', 23.0),
                'experience_months': user_data.get('workout_experience_years', 1) * 12,
                'weekly_workout_frequency': user_data.get('active_days', 3),
                'days_since_last_workout': 2,  # Default assumption
                'fatigue_level': 1  # Default low fatigue
            }

            recommendations = self._filter_by_fitness_level(
                raw_recommendations,
                user_fitness_level,
                rf_predictor,
                num_recommendations,
                rf_user_profile
            )
            logger.info(f"After filtering: {len(recommendations)} recommendations matching fitness level '{user_fitness_level}'")

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
                    'algorithmDisplay': 'Content',
                    'weights': {
                        'content_weight': 1.0,
                        'collaborative_weight': 0.0
                    },
                    'note': 'Insufficient collaborative data, using content-based recommendations'
                }

            # Save recommendations to database
            self._save_recommendations(user_id, recommendations, algorithm_name)

            # Generate informative note based on confidence level
            confidence_messages = {
                "new_user": f"New user (0 ratings) - Using content-based recommendations. Rate exercises after workouts to unlock personalized hybrid recommendations.",
                "low": f"Early stage ({user_rating_count} ratings) - Hybrid active with light collaborative filtering. Rate more exercises for better personalization!",
                "low-medium": f"Building profile ({user_rating_count} ratings) - Hybrid recommendations improving. Keep rating exercises!",
                "medium": f"Good personalization ({user_rating_count} ratings) - Hybrid recommendations with moderate collaborative filtering.",
                "medium-high": f"Strong personalization ({user_rating_count} ratings) - Hybrid recommendations with balanced content + collaborative filtering.",
                "high": f"Very strong personalization ({user_rating_count} ratings) - Collaborative filtering is dominant. Highly tailored to your preferences!",
                "very-high": f"Excellent personalization ({user_rating_count} ratings) - Netflix-style hybrid with equal content + collaborative filtering. Maximum accuracy!",
                "manual": f"Using manually specified weights (Content: {content_weight}, CF: {collaborative_weight})"
            }

            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'count': len(recommendations),
                'algorithm': algorithm_name,
                'algorithmDisplay': algorithm_display,
                'weights': {
                    'content_weight': content_weight,
                    'collaborative_weight': collaborative_weight
                },
                'confidence_level': confidence_level,  # For testing/debugging
                'rating_count': user_rating_count,     # For testing/debugging
                'note': confidence_messages.get(confidence_level, f'{user_rating_count} ratings')
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