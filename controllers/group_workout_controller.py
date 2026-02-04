"""
GroupWorkoutController

Handles API requests for group workout recommendations
Orchestrates UserProfileBuilder and GroupWorkoutRecommender
"""

from flask import current_app
import logging

from services import UserProfileBuilder, GroupWorkoutRecommender

logger = logging.getLogger(__name__)


class GroupWorkoutController:
    """Controller for group workout recommendation endpoints"""

    def __init__(self):
        self.profile_builder = UserProfileBuilder()

    def get_group_recommendations(self, data: dict):
        """
        Generate group workout recommendations

        Expected payload:
        {
            "user_ids": [1, 2, 3, 4],
            "workout_format": "tabata",  // or "generic"
            "target_exercises": 8,
            "include_alternatives": true,  // NEW: optional, for customization
            "num_alternatives": 6           // NEW: optional, default 6
        }

        Returns:
        {
            "success": true,
            "workout": {
                "workout_format": "tabata",
                "exercises": [...],              // Primary exercises (backward compatible)
                "recommended_exercises": [...],  // NEW: Same as exercises
                "alternative_pool": [...],       // NEW: Additional exercises for voting
                "group_analysis": {...},
                "tabata_structure": {...}
            }
        }
        """
        try:
            # Validate input
            user_ids = data.get('user_ids', [])
            if not user_ids or len(user_ids) < 1:
                return {
                    'success': False,
                    'error': 'At least 1 user ID required for group workout'
                }, 400

            if len(user_ids) > 10:
                return {
                    'success': False,
                    'error': 'Maximum 10 users per group workout'
                }, 400

            # Get auth token from data
            auth_token = data.get('auth_token')
            if not auth_token:
                return {
                    'success': False,
                    'error': 'Authorization token required'
                }, 401

            # Get workout parameters
            workout_format = data.get('workout_format', 'tabata')
            target_exercises = int(data.get('target_exercises', 8))

            # NEW: Parse customization parameters
            include_alternatives = data.get('include_alternatives', False)
            num_alternatives = int(data.get('num_alternatives', 6))

            # Convert string 'true'/'false' to boolean (for JSON payloads)
            if isinstance(include_alternatives, str):
                include_alternatives = include_alternatives.lower() == 'true'

            # Initialize recommender with models from app
            model_manager = current_app.model_manager
            recommender = GroupWorkoutRecommender(
                model_manager=model_manager,
                user_profile_builder=self.profile_builder
            )

            # Generate recommendations
            logger.info(f"Generating {workout_format} workout for {len(user_ids)} users" +
                       (f" with {num_alternatives} alternatives" if include_alternatives else ""))

            workout = recommender.get_group_recommendations(
                user_ids=user_ids,
                token=auth_token,
                workout_format=workout_format,
                target_exercises=target_exercises,
                include_alternatives=include_alternatives,
                num_alternatives=num_alternatives
            )

            return {
                'success': True,
                'workout': workout,
                'message': f'{workout_format.capitalize()} workout generated for {len(user_ids)} users'
            }

        except Exception as e:
            logger.error(f"Group workout recommendation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }, 500

    def get_user_profile(self, user_id: int, data: dict):
        """
        Get ML-ready user profile

        Returns:
        {
            "success": true,
            "profile": {
                "user_id": 1,
                "age": 25,
                "fitness_level_numeric": 3,
                "bmi_category_numeric": 1,
                "user_experience": 0.5,
                "user_consistency": 3.5,
                "days_since_last_workout": 2,
                "preferred_muscle_groups": ["core", "upper_body"],
                "available_equipment": ["bodyweight", "dumbbells"]
            }
        }
        """
        try:
            # Get auth token
            auth_token = data.get('auth_token')
            if not auth_token:
                return {
                    'success': False,
                    'error': 'Authorization token required'
                }, 401

            # Build profile
            logger.info(f"Building ML profile for user {user_id}")

            profile = self.profile_builder.get_user_profile(
                user_id=user_id,
                token=auth_token
            )

            return {
                'success': True,
                'profile': profile
            }

        except Exception as e:
            logger.error(f"User profile build error: {e}")
            return {
                'success': False,
                'error': str(e)
            }, 500
