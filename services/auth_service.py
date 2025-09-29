"""
Auth Service Communication Module
=================================

Handles communication with fitneaseauth Laravel service
"""

import os
from typing import Dict, Optional, List
import logging
from .base_service import BaseService

logger = logging.getLogger(__name__)

class AuthService(BaseService):
    """Communication with fitneaseauth service"""

    def __init__(self):
        base_url = os.environ.get('LARAVEL_AUTH_URL', 'http://fitnease-auth:80')
        super().__init__('fitnease-auth', base_url)

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """
        Get user profile including preferences, fitness assessments, demographics
        Endpoint: GET /auth/user-profile/{user_id}
        """
        try:
            response = self.get(f'/auth/user-profile/{user_id}')

            if self.validate_response(response, ['user_id']):
                return response

            # Fallback: return mock user data
            logger.warning(f"Auth service unavailable, using mock data for user {user_id}")
            return self._get_mock_user_profile(user_id)

        except Exception as e:
            logger.error(f"Error getting user profile for user {user_id}: {e}")
            return self._get_mock_user_profile(user_id)

    def get_user_profile_with_token(self, user_id: int, token: str) -> Optional[Dict]:
        """
        Get user profile with authentication token
        Endpoint: GET /api/auth/auth/user-profile/{user_id}
        Returns: User profile with preferences for ML personalization
        """
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = self.get(f'/api/auth/user-profile/{user_id}', headers=headers)

            if self.validate_response(response):
                return response

            # Auth service failed - return None to use defaults
            logger.warning(f"Auth service unavailable for user {user_id}, status: {response.get('status', 'unknown') if isinstance(response, dict) else 'error'}")
            return None

        except Exception as e:
            logger.error(f"Error getting user profile for user {user_id}: {e}")
            return None

    def validate_token(self, token: str) -> Optional[Dict]:
        """
        Validate authentication token
        Endpoint: GET /auth/validate
        """
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = self.get('/auth/validate', headers=headers)

            if self.validate_response(response):
                return response

            return None

        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None

    def get_all_fitness_assessments(self) -> Optional[List[Dict]]:
        """
        Get bulk fitness data for Random Forest training
        Endpoint: GET /auth/all-fitness-assessments
        """
        try:
            response = self.get('/auth/all-fitness-assessments')

            if self.validate_response(response):
                assessments = response.get('assessments', [])
                return assessments

            # Fallback: return mock assessments
            return self._get_mock_fitness_assessments()

        except Exception as e:
            logger.error(f"Error getting all fitness assessments: {e}")
            return self._get_mock_fitness_assessments()

    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user workout preferences"""
        try:
            response = self.get(f'/auth/user-preferences/{user_id}')

            if self.validate_response(response, ['user_id']):
                return response

            return self._get_mock_user_preferences(user_id)

        except Exception as e:
            logger.error(f"Error getting user preferences for user {user_id}: {e}")
            return self._get_mock_user_preferences(user_id)

    def get_user_fitness_assessment(self, user_id: int) -> Optional[Dict]:
        """Get user fitness assessment data"""
        try:
            response = self.get(f'/auth/fitness-assessment/{user_id}')

            if self.validate_response(response, ['user_id']):
                return response

            return self._get_mock_fitness_assessment(user_id)

        except Exception as e:
            logger.error(f"Error getting fitness assessment for user {user_id}: {e}")
            return self._get_mock_fitness_assessment(user_id)

    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        try:
            data = {
                'user_id': user_id,
                'preferences': preferences
            }

            response = self.put(f'/auth/user-preferences/{user_id}', data=data)

            if self.validate_response(response):
                return response.get('success', False)

            return False

        except Exception as e:
            logger.error(f"Error updating user preferences for user {user_id}: {e}")
            return False

    def _get_mock_user_profile(self, user_id: int) -> Dict:
        """Mock user profile data for testing"""
        return {
            'user_id': user_id,
            'age': 25 + (user_id % 40),
            'gender': ['male', 'female'][user_id % 2],
            'fitness_level': ['beginner', 'intermediate', 'expert'][user_id % 3],
            'bmi': 20.0 + (user_id % 10),
            'experience_months': 6 + (user_id % 36),
            'weekly_workout_frequency': 2 + (user_id % 5),
            'goals': ['weight_loss', 'muscle_gain', 'endurance'][user_id % 3],
            'preferences': {
                'preferred_workout_duration': 30 + (user_id % 30),
                'preferred_muscle_groups': ['core', 'upper_body', 'lower_body'][user_id % 3],
                'equipment_preference': ['bodyweight', 'dumbbells', 'barbell'][user_id % 3],
                'difficulty_preference': (user_id % 3) + 1
            },
            'health_conditions': [],
            'last_assessment_date': '2025-09-15'
        }

    def _get_mock_fitness_assessments(self) -> List[Dict]:
        """Mock fitness assessments for training data"""
        assessments = []

        for i in range(1, 101):  # 100 users
            assessment = {
                'user_id': i,
                'fitness_level': ['beginner', 'intermediate', 'expert'][i % 3],
                'bmi': 18.5 + (i % 15),
                'age': 18 + (i % 50),
                'experience_months': (i % 48),
                'weekly_frequency': 1 + (i % 7),
                'strength_score': 40 + (i % 60),
                'endurance_score': 35 + (i % 65),
                'flexibility_score': 30 + (i % 70),
                'assessment_date': f'2025-{9 - (i % 12):02d}-{1 + (i % 28):02d}'
            }
            assessments.append(assessment)

        return assessments

    def _get_mock_user_preferences(self, user_id: int) -> Dict:
        """Mock user preferences"""
        return {
            'user_id': user_id,
            'preferred_workout_duration': 20 + (user_id % 40),
            'preferred_time_of_day': ['morning', 'afternoon', 'evening'][user_id % 3],
            'preferred_muscle_groups': ['core', 'upper_body', 'lower_body'][user_id % 3],
            'equipment_preference': ['bodyweight', 'dumbbells', 'barbell'][user_id % 3],
            'difficulty_preference': (user_id % 3) + 1,
            'workout_frequency_goal': 2 + (user_id % 5),
            'fitness_goals': ['weight_loss', 'muscle_gain', 'endurance', 'flexibility'][user_id % 4]
        }

    def _get_mock_fitness_assessment(self, user_id: int) -> Dict:
        """Mock fitness assessment data"""
        return {
            'user_id': user_id,
            'fitness_level': ['beginner', 'intermediate', 'expert'][user_id % 3],
            'overall_score': 50 + (user_id % 50),
            'strength_score': 40 + (user_id % 60),
            'endurance_score': 45 + (user_id % 55),
            'flexibility_score': 35 + (user_id % 65),
            'balance_score': 50 + (user_id % 50),
            'bmi': 18.5 + (user_id % 15),
            'body_fat_percentage': 10 + (user_id % 25),
            'resting_heart_rate': 60 + (user_id % 40),
            'max_heart_rate': 190 - (user_id % 40),
            'assessment_date': '2025-09-15',
            'recommendations': {
                'target_difficulty': (user_id % 3) + 1,
                'recommended_frequency': 3 + (user_id % 4),
                'focus_areas': ['strength', 'cardio', 'flexibility'][user_id % 3]
            }
        }