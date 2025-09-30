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
                # Use default preferences instead of mock data
                logger.warning(f"Auth service unavailable for user {user_id}, using default preferences")
                user_data = {
                    'target_muscle_groups': ['core', 'upper_body', 'lower_body'],
                    'fitness_level': 'intermediate',
                    'available_equipment': ['bodyweight'],
                    'fitness_goals': ['general_fitness'],
                    'age': 25,
                    'workout_experience_years': 1,
                    'time_constraints_minutes': 30,
                    'activity_level': 'moderate'
                }

            # Note: Exercise data will be loaded by the ML model directly when needed

            # Always get real exercise data from content service (remove token dependency)
            logger.info("Getting real exercise data from content service...")

            # Try to get exercise data without token first (internal ML endpoint)
            exercise_data = self.content_service.get_all_exercises()

            if not exercise_data and auth_token:
                # Fallback to token-based endpoint if internal fails
                exercise_data = self.content_service.get_exercise_attributes_with_token(auth_token)

            if exercise_data and len(exercise_data) > 0:
                # Use REAL exercise data from database for recommendations
                logger.info(f"Using REAL exercise data: {len(exercise_data)} exercises from database")

                # Extract and format user preferences from real user data
                real_preferences = self._extract_user_preferences(user_data)
                logger.info(f"Using real user preferences: {real_preferences}")

                recommendations = self._generate_content_based_recommendations(
                    exercise_data, real_preferences, num_recommendations
                )
            else:
                # API endpoints failed, but we have enhanced exercise data with real names
                logger.warning("API endpoints returned no data, using enhanced exercise data with real names")

                # The content service's _get_mock_exercises now has real exercise names
                enhanced_exercise_data = self.content_service._get_mock_exercises()

                # Extract and format user preferences from real user data
                real_preferences = self._extract_user_preferences(user_data)
                logger.info(f"Using real user preferences: {real_preferences}")

                recommendations = self._generate_content_based_recommendations(
                    enhanced_exercise_data, real_preferences, num_recommendations
                )

            # Transform recommendations to match client interface
            transformed_recommendations = []
            for rec in recommendations:
                # Format muscle group name properly (remove underscores, capitalize)
                muscle_group_raw = rec.get('target_muscle_group', 'core')
                muscle_group_formatted = muscle_group_raw.replace('_', ' ').title()

                # Use correct field names from the recommendation data
                duration_seconds = rec.get('default_duration_seconds', rec.get('duration_seconds', 20))

                # Use the estimated calories for the single exercise duration (already calculated correctly)
                # This is calories burned for one exercise interval, not full Tabata workout
                estimated_calories = rec.get('estimated_calories_burned', 2.0)

                transformed_rec = {
                    'workout_id': rec.get('exercise_id', 1),
                    'exercise_id': rec.get('exercise_id', 1),
                    'exercise_name': rec.get('exercise_name', 'Unknown Exercise'),
                    'recommendation_score': rec.get('preference_score', 0.8),
                    'content_based_score': rec.get('preference_score', 0.8),
                    'collaborative_score': 0.0,
                    'algorithm_used': 'content_based',
                    'recommendation_reason': f"Content-based recommendation for {muscle_group_formatted} exercises",
                    'difficulty_level': rec.get('difficulty_level', 1),
                    'target_muscle_group': muscle_group_formatted,
                    'default_duration_seconds': duration_seconds,
                    'estimated_calories_burned': round(estimated_calories, 1),
                    'equipment_needed': rec.get('equipment_needed', 'bodyweight'),
                    'exercise_category': rec.get('exercise_category', 'tabata')
                }
                transformed_recommendations.append(transformed_rec)

            # Save recommendations to database
            self._save_recommendations(user_id, transformed_recommendations, 'content_based')

            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': transformed_recommendations,
                'count': len(transformed_recommendations),
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
                # Ensure we have a valid recommendation score
                rec_score = rec.get('recommendation_score', rec.get('similarity_score', rec.get('preference_score', 0.8)))
                if rec_score is None:
                    rec_score = 0.8  # Default score

                recommendation = Recommendations(
                    user_id=user_id,
                    workout_id=rec.get('exercise_id'),
                    recommendation_score=float(rec_score),
                    algorithm_used=algorithm,
                    recommendation_type=algorithm,
                    content_based_score=float(rec_score),
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

    def _generate_content_based_recommendations(self, exercise_data: List[Dict], user_preferences: Dict, num_recommendations: int) -> List[Dict]:
        """Generate content-based recommendations using real exercise data"""

        # Simple content-based filtering logic using exercise attributes
        # Score exercises based on user preferences (if any) or use variety

        # Default preferences if none provided
        preferred_muscle_groups = user_preferences.get('muscle_groups', ['core', 'upper_body', 'lower_body'])
        preferred_difficulty = user_preferences.get('difficulty', 2)  # Medium difficulty
        preferred_equipment = user_preferences.get('equipment', ['bodyweight'])

        scored_exercises = []

        for exercise in exercise_data:
            score = 0.0

            # Muscle group preference (40% weight)
            if exercise.get('target_muscle_group') in preferred_muscle_groups:
                score += 0.4
            else:
                score += 0.1  # Small base score for variety

            # Difficulty preference (30% weight)
            # Convert text difficulty to numeric for comparison
            exercise_difficulty_text = exercise.get('difficulty_level', 'medium')
            difficulty_map = {'beginner': 1, 'medium': 2, 'advanced': 3, 'expert': 3, 'intermediate': 2}
            exercise_difficulty = difficulty_map.get(exercise_difficulty_text, 2)

            difficulty_diff = abs(exercise_difficulty - preferred_difficulty)
            if difficulty_diff == 0:
                score += 0.3
            elif difficulty_diff == 1:
                score += 0.2
            else:
                score += 0.1

            # Equipment preference (20% weight)
            if exercise.get('equipment_needed') in preferred_equipment:
                score += 0.2
            elif exercise.get('equipment_needed') == 'bodyweight':
                score += 0.15  # Bodyweight always accessible
            else:
                score += 0.05

            # Calorie efficiency (10% weight)
            calories = exercise.get('estimated_calories_burned', 1)
            calorie_score = min(0.1, calories / 50)  # Normalize calories
            score += calorie_score

            # Use the estimated_calories_burned directly for the single exercise duration
            # This is already calculated correctly by the content service
            estimated_total_calories = exercise.get('estimated_calories_burned', 2.0)
            duration_seconds = exercise.get('default_duration_seconds', 20)

            scored_exercises.append({
                'exercise_id': exercise.get('exercise_id'),
                'exercise_name': exercise.get('exercise_name'),
                'target_muscle_group': exercise.get('target_muscle_group'),
                'difficulty_level': exercise_difficulty,  # Use numeric value for mobile app compatibility
                'difficulty_level_text': exercise_difficulty_text,  # Keep original text for reference
                'equipment_needed': exercise.get('equipment_needed'),
                'exercise_category': exercise.get('exercise_category', 'strength'),
                'estimated_calories_burned': estimated_total_calories,  # Include the correct calories value
                'default_duration_seconds': duration_seconds,
                'preference_score': score,
                'recommendation_reason': f"Content-based match for {exercise.get('target_muscle_group', 'fitness')} training"
            })

        # Sort by score and return top recommendations
        scored_exercises.sort(key=lambda x: x['preference_score'], reverse=True)

        # Add some randomization to provide variety while maintaining personalization
        import random
        random.seed()  # Use current time as seed for true variety on refresh

        # Take top candidates and shuffle slightly for variety
        top_candidates = scored_exercises[:num_recommendations * 2] if len(scored_exercises) > num_recommendations * 2 else scored_exercises
        random.shuffle(top_candidates)

        return top_candidates[:num_recommendations]

    def _extract_user_preferences(self, user_data: Dict) -> Dict:
        """Extract and format user preferences from real auth service user data"""
        if not user_data:
            return {}

        # Map real user data fields to expected preference format
        preferences = {}

        # Extract muscle groups from target_muscle_groups (array field)
        target_muscle_groups = user_data.get('target_muscle_groups', [])
        if target_muscle_groups and isinstance(target_muscle_groups, list) and len(target_muscle_groups) > 0:
            preferences['muscle_groups'] = target_muscle_groups
        else:
            preferences['muscle_groups'] = ['core', 'upper_body', 'lower_body']  # Default variety

        # Map fitness_level to difficulty preference (1-3 scale)
        fitness_level = user_data.get('fitness_level', 'intermediate')
        difficulty_map = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 3
        }
        preferences['difficulty'] = difficulty_map.get(fitness_level, 2)

        # Extract equipment from available_equipment (array field)
        available_equipment = user_data.get('available_equipment', [])
        if available_equipment and isinstance(available_equipment, list) and len(available_equipment) > 0:
            preferences['equipment'] = available_equipment
        else:
            preferences['equipment'] = ['bodyweight']  # Default to bodyweight

        # Extract additional preferences for enhanced personalization
        preferences['age'] = user_data.get('age', 25)
        preferences['experience_years'] = user_data.get('workout_experience_years', 1)
        preferences['time_constraints'] = user_data.get('time_constraints_minutes', 30)

        # Handle fitness_goals (array field)
        fitness_goals = user_data.get('fitness_goals', [])
        if fitness_goals and isinstance(fitness_goals, list) and len(fitness_goals) > 0:
            preferences['fitness_goals'] = fitness_goals
        else:
            preferences['fitness_goals'] = ['general_fitness']

        preferences['activity_level'] = user_data.get('activity_level', 'moderate')

        return preferences

