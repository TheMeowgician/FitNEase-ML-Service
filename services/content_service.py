"""
Content Service Communication Module
====================================

Handles communication with fitneasecontent Laravel service
"""

import os
from typing import Dict, Optional, List
import logging
from .base_service import BaseService

logger = logging.getLogger(__name__)

class ContentService(BaseService):
    """Communication with fitneasecontent service"""

    def __init__(self):
        base_url = os.environ.get('LARAVEL_CONTENT_URL', 'http://fitnease-content:80')
        super().__init__('fitnease-content', base_url)

    def get_exercise_attributes(self) -> Optional[List[Dict]]:
        """
        Get exercise features for Content-Based filtering
        Endpoint: GET /content/exercise-attributes
        """
        try:
            response = self.get('/content/exercise-attributes')

            if self.validate_response(response):
                exercises = response.get('exercises', [])
                return exercises

            # Fallback: return mock exercise data
            logger.warning("Content service unavailable, using mock exercise data")
            return self._get_mock_exercises()

        except Exception as e:
            logger.error(f"Error getting exercise attributes: {e}")
            return self._get_mock_exercises()

    def get_exercise_attributes_with_token(self, token: str) -> Optional[List[Dict]]:
        """
        Get exercise features for Content-Based filtering with authentication
        Endpoint: GET /api/content/exercise-attributes
        """
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = self.get('/api/content/exercise-attributes', headers=headers)

            if self.validate_response(response):
                exercises = response.get('exercises', [])
                return exercises

            # Fallback: return mock exercise data
            logger.warning("Content service unavailable, using mock exercise data")
            return self._get_mock_exercises()

        except Exception as e:
            logger.error(f"Error getting exercise attributes with token: {e}")
            return self._get_mock_exercises()

    def get_all_exercises(self) -> Optional[List[Dict]]:
        """
        Get bulk exercise data for Content-Based model training
        Endpoint: GET /content/all-exercises
        """
        try:
            response = self.get('/content/all-exercises')

            if self.validate_response(response):
                exercises = response.get('exercises', [])
                return exercises

            return self._get_mock_exercises()

        except Exception as e:
            logger.error(f"Error getting all exercises: {e}")
            return self._get_mock_exercises()

    def get_workout_details(self, workout_id: int) -> Optional[Dict]:
        """
        Get workout details for Random Forest predictions
        Endpoint: GET /content/workouts/{workout_id}
        """
        try:
            response = self.get(f'/content/workouts/{workout_id}')

            if self.validate_response(response, ['workout_id']):
                return response

            # Fallback: return mock workout data
            return self._get_mock_workout(workout_id)

        except Exception as e:
            logger.error(f"Error getting workout details for workout {workout_id}: {e}")
            return self._get_mock_workout(workout_id)

    def get_exercise_by_id(self, exercise_id: int) -> Optional[Dict]:
        """Get specific exercise by ID"""
        try:
            response = self.get(f'/content/exercises/{exercise_id}')

            if self.validate_response(response, ['exercise_id']):
                return response

            return self._get_mock_exercise(exercise_id)

        except Exception as e:
            logger.error(f"Error getting exercise {exercise_id}: {e}")
            return self._get_mock_exercise(exercise_id)

    def calculate_exercise_similarity(self, exercise1_id: int, exercise2_id: int) -> float:
        """Calculate similarity between two exercises"""
        try:
            data = {
                'exercise1_id': exercise1_id,
                'exercise2_id': exercise2_id
            }

            response = self.post('/content/calculate-similarity', data=data)

            if self.validate_response(response, ['similarity_score']):
                return response['similarity_score']

            # Fallback: return default similarity
            return 0.5

        except Exception as e:
            logger.error(f"Error calculating exercise similarity: {e}")
            return 0.5

    def get_exercises_by_muscle_group(self, muscle_group: str) -> Optional[List[Dict]]:
        """Get exercises filtered by muscle group"""
        try:
            params = {'muscle_group': muscle_group}
            response = self.get('/content/exercises', params=params)

            if self.validate_response(response):
                return response.get('exercises', [])

            return []

        except Exception as e:
            logger.error(f"Error getting exercises for muscle group {muscle_group}: {e}")
            return []

    def get_exercises_by_difficulty(self, difficulty_level: int) -> Optional[List[Dict]]:
        """Get exercises filtered by difficulty level"""
        try:
            params = {'difficulty': difficulty_level}
            response = self.get('/content/exercises', params=params)

            if self.validate_response(response):
                return response.get('exercises', [])

            return []

        except Exception as e:
            logger.error(f"Error getting exercises for difficulty {difficulty_level}: {e}")
            return []

    def get_exercise_categories(self) -> Optional[List[Dict]]:
        """Get all exercise categories"""
        try:
            response = self.get('/content/exercise-categories')

            if self.validate_response(response):
                return response.get('categories', [])

            return self._get_mock_categories()

        except Exception as e:
            logger.error(f"Error getting exercise categories: {e}")
            return self._get_mock_categories()

    def search_exercises(self, query: str, filters: Dict = None) -> Optional[List[Dict]]:
        """Search exercises with optional filters"""
        try:
            params = {'q': query}
            if filters:
                params.update(filters)

            response = self.get('/content/search-exercises', params=params)

            if self.validate_response(response):
                return response.get('exercises', [])

            return []

        except Exception as e:
            logger.error(f"Error searching exercises with query '{query}': {e}")
            return []

    def _get_mock_exercises(self) -> List[Dict]:
        """Mock exercise data for testing"""
        exercises = []

        # Mock exercises based on real data structure
        muscle_groups = ['core', 'upper_body', 'lower_body']
        equipment_types = ['bodyweight', 'dumbbells', 'barbell', 'kettlebell']
        categories = ['strength', 'cardio', 'flexibility', 'functional']

        for i in range(1, 401):  # 400 exercises like your real data
            exercise = {
                'exercise_id': i,
                'exercise_name': f'Exercise {i}',
                'target_muscle_group': muscle_groups[i % len(muscle_groups)],
                'difficulty_level': (i % 3) + 1,  # 1-3
                'default_duration_seconds': 20 + (i % 60),  # 20-80 seconds
                'calories_burned_per_minute': 3 + (i % 12),  # 3-15 calories
                'equipment_needed': equipment_types[i % len(equipment_types)],
                'exercise_category': categories[i % len(categories)],
                'instructions': f'Instructions for exercise {i}',
                'safety_tips': f'Safety tips for exercise {i}',
                'video_url': f'https://example.com/videos/exercise_{i}.mp4',
                'image_url': f'https://example.com/images/exercise_{i}.jpg'
            }
            exercises.append(exercise)

        return exercises

    def _get_mock_workout(self, workout_id: int) -> Dict:
        """Mock workout data for testing"""
        return {
            'workout_id': workout_id,
            'workout_name': f'Workout {workout_id}',
            'difficulty_level': (workout_id % 3) + 1,
            'estimated_duration_minutes': 20 + (workout_id % 40),
            'target_muscle_groups': ['core', 'upper_body'][workout_id % 2],
            'equipment_needed': ['bodyweight', 'dumbbells'][workout_id % 2],
            'calories_burned_estimate': 150 + (workout_id % 300),
            'exercise_count': 5 + (workout_id % 10),
            'description': f'Description for workout {workout_id}',
            'exercises': [
                {
                    'exercise_id': i,
                    'sets': 3,
                    'reps': 10 + (i % 5),
                    'duration_seconds': 30 + (i % 30)
                }
                for i in range(workout_id, workout_id + 5)
            ]
        }

    def _get_mock_exercise(self, exercise_id: int) -> Dict:
        """Mock single exercise data"""
        muscle_groups = ['core', 'upper_body', 'lower_body']
        equipment_types = ['bodyweight', 'dumbbells', 'barbell']

        return {
            'exercise_id': exercise_id,
            'exercise_name': f'Exercise {exercise_id}',
            'target_muscle_group': muscle_groups[exercise_id % len(muscle_groups)],
            'difficulty_level': (exercise_id % 3) + 1,
            'default_duration_seconds': 20 + (exercise_id % 60),
            'calories_burned_per_minute': 3 + (exercise_id % 12),
            'equipment_needed': equipment_types[exercise_id % len(equipment_types)],
            'exercise_category': 'strength',
            'instructions': f'Instructions for exercise {exercise_id}',
            'safety_tips': f'Safety tips for exercise {exercise_id}',
            'video_url': f'https://example.com/videos/exercise_{exercise_id}.mp4',
            'image_url': f'https://example.com/images/exercise_{exercise_id}.jpg',
            'muscle_group_primary': muscle_groups[exercise_id % len(muscle_groups)],
            'muscle_group_secondary': muscle_groups[(exercise_id + 1) % len(muscle_groups)]
        }

    def _get_mock_categories(self) -> List[Dict]:
        """Mock exercise categories"""
        return [
            {
                'category_id': 1,
                'category_name': 'Strength Training',
                'description': 'Exercises focused on building muscle strength',
                'exercise_count': 150
            },
            {
                'category_id': 2,
                'category_name': 'Cardiovascular',
                'description': 'Exercises for heart health and endurance',
                'exercise_count': 100
            },
            {
                'category_id': 3,
                'category_name': 'Flexibility',
                'description': 'Stretching and mobility exercises',
                'exercise_count': 75
            },
            {
                'category_id': 4,
                'category_name': 'Functional',
                'description': 'Real-world movement patterns',
                'exercise_count': 75
            }
        ]