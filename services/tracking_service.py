"""
Tracking Service Communication Module
=====================================

Handles communication with fitneasetracking Laravel service
"""

import os
from typing import Dict, Optional, List
import logging
from .base_service import BaseService

logger = logging.getLogger(__name__)

class TrackingService(BaseService):
    """Communication with fitneasetracking service"""

    def __init__(self):
        base_url = os.environ.get('LARAVEL_TRACKING_URL', 'http://fitnease-tracking:80')
        super().__init__('fitnease-tracking', base_url)

    def get_user_history(self, user_id: int) -> Optional[Dict]:
        """
        Get user workout history including completion data, ratings, performance metrics
        Endpoint: GET /tracking/user-history/{user_id}
        """
        try:
            response = self.get(f'/tracking/user-history/{user_id}')

            if self.validate_response(response, ['user_id']):
                return response

            # Fallback: return mock user history
            logger.warning(f"Tracking service unavailable, using mock data for user {user_id}")
            return self._get_mock_user_history(user_id)

        except Exception as e:
            logger.error(f"Error getting user history for user {user_id}: {e}")
            return self._get_mock_user_history(user_id)

    def get_all_user_data(self) -> Optional[List[Dict]]:
        """
        Get bulk user behavior data for Collaborative model training
        Endpoint: GET /tracking/all-user-data
        """
        try:
            response = self.get('/tracking/all-user-data')

            if self.validate_response(response):
                user_data = response.get('users', [])
                return user_data

            # Fallback: return mock user data
            return self._get_mock_all_user_data()

        except Exception as e:
            logger.error(f"Error getting all user data: {e}")
            return self._get_mock_all_user_data()

    def get_completion_rates(self) -> Optional[Dict]:
        """
        Get completion statistics for Random Forest training
        Endpoint: GET /tracking/completion-rates
        """
        try:
            response = self.get('/tracking/completion-rates')

            if self.validate_response(response):
                return response

            return self._get_mock_completion_rates()

        except Exception as e:
            logger.error(f"Error getting completion rates: {e}")
            return self._get_mock_completion_rates()

    def get_user_ratings(self, user_id: int) -> Optional[List[Dict]]:
        """Get user workout/exercise ratings"""
        try:
            response = self.get(f'/tracking/user-ratings/{user_id}')

            if self.validate_response(response):
                return response.get('ratings', [])

            return self._get_mock_user_ratings(user_id)

        except Exception as e:
            logger.error(f"Error getting user ratings for user {user_id}: {e}")
            return self._get_mock_user_ratings(user_id)

    def get_all_user_ratings(self) -> Optional[List[Dict]]:
        """
        Get all user ratings for collaborative filtering
        Endpoint: GET /tracking/all-user-ratings
        """
        try:
            response = self.get('/tracking/all-user-ratings')

            if self.validate_response(response):
                return response.get('ratings', [])

            return self._get_mock_all_ratings()

        except Exception as e:
            logger.error(f"Error getting all user ratings: {e}")
            return self._get_mock_all_ratings()

    def update_behavioral_data(self, user_id: int, behavior_data: Dict) -> bool:
        """Send user behavior updates to tracking service"""
        try:
            data = {
                'user_id': user_id,
                'behavior_data': behavior_data
            }

            response = self.post('/tracking/behavioral-data', data=data)

            if self.validate_response(response):
                return response.get('success', False)

            return False

        except Exception as e:
            logger.error(f"Error updating behavioral data for user {user_id}: {e}")
            return False

    def get_user_workout_sessions(self, user_id: int, limit: int = 50) -> Optional[List[Dict]]:
        """Get user workout sessions"""
        try:
            params = {'limit': limit}
            response = self.get(f'/tracking/user-sessions/{user_id}', params=params)

            if self.validate_response(response):
                return response.get('sessions', [])

            return self._get_mock_user_sessions(user_id, limit)

        except Exception as e:
            logger.error(f"Error getting workout sessions for user {user_id}: {e}")
            return self._get_mock_user_sessions(user_id, limit)

    def get_user_performance_metrics(self, user_id: int) -> Optional[Dict]:
        """Get user performance and progress metrics"""
        try:
            response = self.get(f'/tracking/user-performance/{user_id}')

            if self.validate_response(response, ['user_id']):
                return response

            return self._get_mock_performance_metrics(user_id)

        except Exception as e:
            logger.error(f"Error getting performance metrics for user {user_id}: {e}")
            return self._get_mock_performance_metrics(user_id)

    def get_workout_completion_stats(self, workout_id: int) -> Optional[Dict]:
        """Get completion statistics for specific workout"""
        try:
            response = self.get(f'/tracking/workout-completion/{workout_id}')

            if self.validate_response(response, ['workout_id']):
                return response

            return self._get_mock_workout_completion_stats(workout_id)

        except Exception as e:
            logger.error(f"Error getting workout completion stats for workout {workout_id}: {e}")
            return self._get_mock_workout_completion_stats(workout_id)

    def record_workout_start(self, user_id: int, workout_id: int) -> bool:
        """Record workout start"""
        try:
            data = {
                'user_id': user_id,
                'workout_id': workout_id
            }

            response = self.post('/tracking/workout-start', data=data)

            if self.validate_response(response):
                return response.get('success', False)

            return False

        except Exception as e:
            logger.error(f"Error recording workout start: {e}")
            return False

    def record_workout_completion(self, user_id: int, workout_id: int, completion_data: Dict) -> bool:
        """Record workout completion"""
        try:
            data = {
                'user_id': user_id,
                'workout_id': workout_id,
                'completion_data': completion_data
            }

            response = self.post('/tracking/workout-completion', data=data)

            if self.validate_response(response):
                return response.get('success', False)

            return False

        except Exception as e:
            logger.error(f"Error recording workout completion: {e}")
            return False

    def _get_mock_user_history(self, user_id: int) -> Dict:
        """Mock user history data for testing"""
        return {
            'user_id': user_id,
            'total_workouts': 25 + (user_id % 50),
            'completion_rate': 0.7 + (user_id % 30) / 100,
            'average_workout_duration': 25 + (user_id % 20),
            'favorite_muscle_groups': ['core', 'upper_body'][user_id % 2],
            'most_active_days': ['monday', 'wednesday', 'friday'],
            'improvement_trend': ['stable', 'improving', 'declining'][user_id % 3],
            'last_workout_date': '2025-09-20',
            'streak_days': 5 + (user_id % 15),
            'total_calories_burned': 2500 + (user_id % 1000),
            'average_rating': 3.5 + (user_id % 15) / 10,
            'workout_history': [
                {
                    'workout_id': i,
                    'completion_status': (i + user_id) % 3 != 0,  # ~66% completion rate
                    'rating': 3 + ((i + user_id) % 3),  # 3-5 rating
                    'workout_date': f'2025-09-{20 - i}' if i < 20 else '2025-08-30',
                    'duration_minutes': 20 + (i % 40),
                    'calories_burned': 150 + (i % 200)
                }
                for i in range(1, 21)  # Last 20 workouts
            ]
        }

    def _get_mock_all_user_data(self) -> List[Dict]:
        """Mock all user data for collaborative filtering training"""
        return [
            {
                'user_id': i,
                'total_workouts': 20 + (i % 60),
                'completion_rate': 0.5 + (i % 50) / 100,
                'favorite_muscle_groups': ['core', 'upper_body', 'lower_body'][i % 3],
                'average_rating': 3 + (i % 3),
                'workout_frequency': 2 + (i % 5),  # workouts per week
                'last_active': f'2025-09-{20 - (i % 20)}',
                'preferred_difficulty': (i % 3) + 1,
                'preferred_duration': 20 + (i % 40),
                'equipment_preference': ['bodyweight', 'dumbbells', 'barbell'][i % 3]
            }
            for i in range(1, 201)  # 200 users
        ]

    def _get_mock_completion_rates(self) -> Dict:
        """Mock completion statistics"""
        return {
            'overall_completion_rate': 0.72,
            'completion_by_difficulty': {
                '1': 0.85,  # Beginner
                '2': 0.72,  # Intermediate
                '3': 0.58   # Expert
            },
            'completion_by_muscle_group': {
                'core': 0.75,
                'upper_body': 0.70,
                'lower_body': 0.68
            },
            'completion_by_duration': {
                'short': 0.82,    # < 30 minutes
                'medium': 0.74,   # 30-60 minutes
                'long': 0.65      # > 60 minutes
            },
            'completion_by_equipment': {
                'bodyweight': 0.80,
                'dumbbells': 0.70,
                'barbell': 0.65,
                'kettlebell': 0.68
            },
            'completion_by_time_of_day': {
                'morning': 0.78,
                'afternoon': 0.72,
                'evening': 0.68
            }
        }

    def _get_mock_user_ratings(self, user_id: int) -> List[Dict]:
        """Mock user ratings"""
        return [
            {
                'rating_id': i,
                'user_id': user_id,
                'exercise_id': i,
                'workout_id': i,
                'rating': 3 + ((i + user_id) % 3),  # 3-5 rating
                'difficulty_feedback': ['appropriate', 'too_easy', 'too_hard'][i % 3],
                'enjoyment_rating': 3 + ((i + user_id + 1) % 3),
                'effectiveness_rating': 3 + ((i + user_id + 2) % 3),
                'rating_date': f'2025-09-{20 - (i % 20)}',
                'comments': f'Rating comment {i}'
            }
            for i in range(1, 26)  # 25 ratings per user
        ]

    def _get_mock_all_ratings(self) -> List[Dict]:
        """Mock all user ratings for collaborative filtering"""
        all_ratings = []

        for user_id in range(1, 101):  # 100 users
            for exercise_id in range(1, 21):  # 20 exercises each
                # Not all users rate all exercises (sparse matrix)
                if (user_id + exercise_id) % 3 == 0:  # ~33% rating coverage
                    rating = {
                        'user_id': user_id,
                        'exercise_id': exercise_id,
                        'rating': 3 + ((user_id + exercise_id) % 3),  # 3-5
                        'rating_date': f'2025-09-{(user_id + exercise_id) % 20 + 1}',
                        'difficulty_feedback': ['appropriate', 'too_easy', 'too_hard'][(user_id + exercise_id) % 3],
                        'enjoyment_rating': 3 + ((user_id + exercise_id + 1) % 3)
                    }
                    all_ratings.append(rating)

        return all_ratings

    def _get_mock_user_sessions(self, user_id: int, limit: int) -> List[Dict]:
        """Mock user workout sessions"""
        return [
            {
                'session_id': i,
                'user_id': user_id,
                'workout_id': i,
                'duration_minutes': 20 + (i % 40),
                'calories_burned': 150 + (i % 200),
                'completion_percentage': 0.6 + (i % 40) / 100,
                'session_date': f'2025-09-{20 - (i % 20)}',
                'exercises_completed': 5 + (i % 10),
                'average_heart_rate': 120 + (i % 60),
                'max_heart_rate': 160 + (i % 40),
                'session_rating': 3 + (i % 3)
            }
            for i in range(1, min(limit + 1, 51))
        ]

    def _get_mock_performance_metrics(self, user_id: int) -> Dict:
        """Mock performance metrics"""
        return {
            'user_id': user_id,
            'current_fitness_level': ['beginner', 'intermediate', 'expert'][user_id % 3],
            'strength_progress': 60 + (user_id % 40),
            'endurance_progress': 55 + (user_id % 45),
            'consistency_score': 0.6 + (user_id % 40) / 100,
            'improvement_rate': 1.0 + (user_id % 50) / 100,
            'total_workouts_completed': 30 + (user_id % 70),
            'average_workout_rating': 3.5 + (user_id % 15) / 10,
            'last_assessment_date': '2025-09-15',
            'weekly_frequency': 3 + (user_id % 4),
            'preferred_workout_time': ['morning', 'afternoon', 'evening'][user_id % 3],
            'streak_record': 10 + (user_id % 20)
        }

    def _get_mock_workout_completion_stats(self, workout_id: int) -> Dict:
        """Mock workout completion statistics"""
        return {
            'workout_id': workout_id,
            'total_attempts': 50 + (workout_id % 100),
            'total_completions': 35 + (workout_id % 70),
            'completion_rate': 0.6 + (workout_id % 40) / 100,
            'average_duration': 25 + (workout_id % 35),
            'average_rating': 3.5 + (workout_id % 15) / 10,
            'difficulty_feedback': {
                'too_easy': 0.15,
                'appropriate': 0.70,
                'too_hard': 0.15
            },
            'popular_times': ['morning', 'evening'],
            'user_demographics': {
                'beginner': 0.30,
                'intermediate': 0.50,
                'expert': 0.20
            }
        }