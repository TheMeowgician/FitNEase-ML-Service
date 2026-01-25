"""
UserProfileBuilder Service

Fetches user data from auth and tracking services and builds
ML-ready user profiles for group workout recommendations.

NO ML TRAINING - just data fetching and simple calculations.
"""

import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class UserProfileBuilder:
    """
    Build user fitness profile from existing services
    NO ML TRAINING - just data fetching + simple calculations
    """

    def __init__(self):
        self.auth_service_url = os.getenv('LARAVEL_AUTH_URL', 'http://fitnease-auth')
        self.tracking_service_url = os.getenv('LARAVEL_TRACKING_URL', 'http://fitnease-tracking')

    def get_user_profile(self, user_id: int, token: str) -> Dict[str, Any]:
        """
        Build complete user profile for ML predictions

        Args:
            user_id: User ID
            token: Authorization token

        Returns:
            {
                'user_id': int,
                'age': int,
                'fitness_level_numeric': int (1, 3, or 5),
                'bmi_category_numeric': int (1, 2, or 3),
                'user_experience': float (0-1),
                'user_consistency': float (workouts per week),
                'days_since_last_workout': int,
                'preferred_muscle_groups': list,
                'available_equipment': list
            }
        """

        # Fetch from auth service
        assessment = self._get_fitness_assessment(user_id, token)

        # Fetch from tracking service
        history = self._get_workout_history(user_id, token, days=30)

        # Build profile with simple calculations
        return {
            'user_id': user_id,
            'age': assessment.get('age', 30),
            'fitness_level_numeric': self._map_fitness_level(
                assessment.get('fitness_level', 'beginner')  # Matches assessment_data.fitness_level
            ),
            'bmi_category_numeric': self._calculate_bmi_category(assessment),
            'user_experience': self._calculate_experience(history),
            'user_consistency': self._calculate_consistency(history),
            'days_since_last_workout': self._get_days_since_last(history),
            'preferred_muscle_groups': self._analyze_muscle_preferences(history),
            'available_equipment': self._get_available_equipment(assessment)
        }

    def _get_fitness_assessment(self, user_id: int, token: str) -> Dict[str, Any]:
        """Fetch latest fitness assessment from auth service"""
        try:
            response = requests.get(
                f"{self.auth_service_url}/api/users/{user_id}/assessments",
                headers={'Authorization': f'Bearer {token}'},
                timeout=5
            )

            if response.status_code == 200:
                # API returns assessments list directly (not wrapped in 'data')
                data = response.json()

                # Handle case where data might be a list or dict
                if isinstance(data, list):
                    assessment_row = data[0] if data else {}
                elif isinstance(data, dict):
                    # Check if it's wrapped in 'data' key
                    assessment_row = data.get('data', [data])[0] if data.get('data') else data
                else:
                    assessment_row = {}

                # Extract assessment_data JSON field (contains fitness_level, age, etc.)
                assessment_data = assessment_row.get('assessment_data', {})

                # Handle if assessment_data is a string (JSON encoded)
                if isinstance(assessment_data, str):
                    import json
                    try:
                        assessment_data = json.loads(assessment_data)
                    except:
                        assessment_data = {}

                # Debug log
                fitness_level = assessment_data.get('fitness_level', 'NOT_FOUND')
                print(f"[DEBUG] User {user_id} assessment_data: fitness_level={fitness_level}")
                return assessment_data

            return {}
        except Exception as e:
            print(f"[ERROR] Failed to fetch assessment: {e}")
            # Return empty dict which will trigger 'beginner' default
            return {}

    def _get_workout_history(self, user_id: int, token: str, days: int = 30) -> List[Dict[str, Any]]:
        """Fetch workout history from tracking service"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            response = requests.get(
                f"{self.tracking_service_url}/api/tracking/workouts",
                headers={'Authorization': f'Bearer {token}'},
                params={
                    'user_id': user_id,
                    'start_date': start_date
                },
                timeout=5
            )

            if response.status_code == 200:
                return response.json().get('data', [])

            return []
        except Exception as e:
            print(f"[ERROR] Failed to fetch workout history: {e}")
            return []

    def _map_fitness_level(self, level_string: str) -> int:
        """Map string to numeric (1=beginner, 2=intermediate, 3=advanced)

        IMPORTANT: This must match the exercise database difficulty levels:
        - 1 = Beginner
        - 2 = Intermediate
        - 3 = Advanced
        """
        mapping = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3,
            'expert': 3
        }
        return mapping.get(level_string.lower(), 2)  # Default to intermediate

    def _calculate_bmi_category(self, assessment: Dict[str, Any]) -> int:
        """Calculate BMI category (1=normal, 2=overweight, 3=obese)"""
        weight = assessment.get('weight', 70)
        height = assessment.get('height', 170) / 100  # cm to m

        if height > 0:
            bmi = weight / (height ** 2)

            if bmi < 25:
                return 1  # normal
            elif bmi < 30:
                return 2  # overweight
            else:
                return 3  # obese

        return 1  # default

    def _calculate_experience(self, history: List[Dict[str, Any]]) -> float:
        """Calculate experience score (0-1) based on workout count"""
        if not history:
            return 0.0

        total_workouts = len(history)

        # Simple ratio: experience increases with workouts (max at 100)
        experience = min(1.0, total_workouts / 100)

        return experience

    def _calculate_consistency(self, history: List[Dict[str, Any]]) -> float:
        """Calculate workouts per week"""
        if not history or len(history) < 2:
            return 0.0

        # Get date range
        dates = [datetime.fromisoformat(w['created_at']) for w in history]
        date_range = (max(dates) - min(dates)).days

        if date_range == 0:
            return len(history)

        weeks = date_range / 7
        return len(history) / weeks

    def _get_days_since_last(self, history: List[Dict[str, Any]]) -> int:
        """Get days since last workout"""
        if not history:
            return 999  # Large number if no history

        # Find most recent workout
        dates = [datetime.fromisoformat(w['created_at']) for w in history]
        last_workout = max(dates)

        return (datetime.now() - last_workout).days

    def _analyze_muscle_preferences(self, history: List[Dict[str, Any]]) -> List[str]:
        """Analyze which muscle groups user prefers"""
        if not history:
            return ['core', 'upper_body', 'lower_body']

        # Count muscle groups from workout history
        muscle_counts = {}

        for workout in history:
            exercises = workout.get('exercises', [])
            for exercise in exercises:
                muscle = exercise.get('target_muscle_group', 'core')
                muscle_counts[muscle] = muscle_counts.get(muscle, 0) + 1

        # Return top 3 most worked muscle groups
        sorted_muscles = sorted(
            muscle_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        preferred = [muscle for muscle, count in sorted_muscles[:3]]

        # Default if none found
        if not preferred:
            preferred = ['core', 'upper_body', 'lower_body']

        return preferred

    def _get_available_equipment(self, assessment: Dict[str, Any]) -> List[str]:
        """Get available equipment from assessment"""
        equipment = assessment.get('available_equipment', [])

        if not equipment:
            # Default to bodyweight if not specified
            return ['bodyweight']

        return equipment
