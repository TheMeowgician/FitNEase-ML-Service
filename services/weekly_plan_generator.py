"""
Weekly Plan Generator Service
==============================

Feature #4: Weekly Workout Plans with Preferred Days
Generates weekly workout plans by distributing recommended exercises across user's preferred workout days.
"""

import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class WeeklyPlanGenerator:
    """
    Generates weekly workout plans based on user preferences and ML recommendations
    """

    def __init__(self, model_manager=None):
        """
        Initialize Weekly Plan Generator

        Args:
            model_manager: Reference to ModelManager for accessing ML models
        """
        self.model_manager = model_manager
        self.days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    def generate_weekly_plan(
        self,
        user_id: int,
        workout_days: List[str],
        fitness_level: str = 'beginner',
        target_muscle_groups: List[str] = None,
        goals: List[str] = None,
        time_constraints: int = 30,
        week_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate weekly workout plan for a user

        Args:
            user_id: User ID
            workout_days: List of preferred workout days (e.g., ['monday', 'wednesday', 'friday'])
            fitness_level: User's fitness level ('beginner', 'intermediate', 'advanced')
            target_muscle_groups: List of target muscle groups
            goals: List of fitness goals
            time_constraints: Time available per workout in minutes
            week_seed: Optional seed for exercise variety (default: current ISO week number)

        Returns:
            Dictionary containing:
            - weekly_plan: Dict with daily workout assignments
            - metadata: Statistics about the plan
        """
        try:
            # Calculate week seed if not provided (ensures variety across weeks)
            if week_seed is None:
                week_seed = datetime.now().isocalendar()[1]  # ISO week number (1-53)

            logger.info(f"[WEEKLY_PLAN] Generating plan for user {user_id} (week seed: {week_seed})")
            logger.info(f"[WEEKLY_PLAN] Workout days: {workout_days}, Level: {fitness_level}")

            # Validate inputs
            workout_days = self._validate_workout_days(workout_days)
            if not workout_days:
                raise ValueError("At least one workout day must be specified")

            # Determine exercises per day based on fitness level and time
            exercises_per_day = self._calculate_exercises_per_day(fitness_level, time_constraints)

            # Get exercise recommendations from ML model
            all_exercises = self._get_exercise_recommendations(
                user_id, fitness_level, target_muscle_groups, goals,
                len(workout_days) * exercises_per_day, week_seed
            )

            if not all_exercises:
                raise ValueError("No exercises available for recommendations")

            # Group exercises by muscle group for balanced distribution
            grouped_exercises = self._group_exercises_by_muscle_group(all_exercises)

            # Distribute exercises across workout days
            weekly_plan_data = self._distribute_exercises_across_days(
                workout_days, grouped_exercises, exercises_per_day, target_muscle_groups
            )

            # Calculate metadata
            metadata = self._calculate_plan_metadata(weekly_plan_data, time_constraints)
            metadata['week_seed'] = week_seed  # Add week seed for variety tracking

            logger.info(f"[WEEKLY_PLAN] Plan generated: {metadata['total_exercises']} exercises across {len(workout_days)} days (week seed: {week_seed})")

            return {
                'weekly_plan': weekly_plan_data,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"[WEEKLY_PLAN] Generation failed: {e}")
            raise

    def _validate_workout_days(self, workout_days: List[str]) -> List[str]:
        """Validate and normalize workout days"""
        if not workout_days:
            return []

        validated = []
        for day in workout_days:
            day_lower = day.lower()
            if day_lower in self.days_of_week:
                validated.append(day_lower)
            else:
                logger.warning(f"[WEEKLY_PLAN] Invalid day: {day}")

        return validated

    def _calculate_exercises_per_day(self, fitness_level: str, time_constraints: int) -> int:
        """
        Calculate how many exercises fit in the time constraint

        Tabata: 4 minutes per exercise (8 rounds × 30s)
        """
        # Base exercises per day by fitness level
        base_exercises = {
            'beginner': 4,
            'intermediate': 5,
            'advanced': 6
        }

        exercises_by_level = base_exercises.get(fitness_level, 4)

        # Adjust for time constraints (4 min per exercise in Tabata)
        max_exercises_by_time = time_constraints // 4

        # Return the minimum to respect both fitness level and time
        return min(exercises_by_level, max_exercises_by_time)

    def _get_exercise_recommendations(
        self,
        user_id: int,
        fitness_level: str,
        target_muscle_groups: List[str],
        goals: List[str],
        total_needed: int,
        week_seed: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get exercise recommendations using ML models with intelligent fallback

        Flow (same as Dashboard/Workouts):
        1. Try Hybrid ML (60% content + 40% collaborative)
        2. If fails → Try Content-Based ML (100% content, STILL ML!)
        3. If both fail → Use hardcoded fallback (last resort)

        Returns list of exercises with ML-powered personalization
        """
        try:
            # ✅ STEP 1: Try Hybrid ML model first (same as Dashboard/Workouts!)
            if not self.model_manager:
                logger.error("[WEEKLY_PLAN] Model manager not available, skipping ML models")
                return self._get_fallback_exercises(total_needed, fitness_level, target_muscle_groups)

            hybrid_model = self.model_manager.get_hybrid_model()
            if hybrid_model:
                logger.info(f"[WEEKLY_PLAN] Trying HYBRID ML model for user {user_id}")

                recommendations = hybrid_model.get_recommendations(
                    user_id=user_id,
                    user_preferences={
                        'fitness_level': fitness_level,
                        'target_muscle_groups': target_muscle_groups if target_muscle_groups else [],
                        'goals': goals if goals else []
                    },
                    num_recommendations=total_needed,
                    content_weight=0.6,
                    collaborative_weight=0.4
                )

                if recommendations and len(recommendations) > 0:
                    logger.info(f"[WEEKLY_PLAN] ✅ HYBRID ML returned {len(recommendations)} exercises")
                    # Apply week-based shuffling for variety across weeks
                    shuffled_recommendations = self._apply_week_variety(recommendations, week_seed)
                    return shuffled_recommendations
                else:
                    logger.warning(f"[WEEKLY_PLAN] Hybrid ML returned 0 recommendations (new user or insufficient data)")

            # ✅ STEP 2: Fallback to Content-Based ML (STILL ML-POWERED!)
            logger.info(f"[WEEKLY_PLAN] Falling back to CONTENT-BASED ML model")
            content_model = self.model_manager.get_content_based_model()

            if content_model:
                # Build user preferences dict for content-based model
                user_preferences = {
                    'muscle_groups': target_muscle_groups if target_muscle_groups else ['core', 'upper_body', 'lower_body'],
                    'difficulty': self._map_fitness_level_to_difficulty(fitness_level),
                    'equipment': ['bodyweight'],
                    'fitness_goals': goals if goals else ['general_fitness']
                }

                # Content-based model uses user preferences for personalization
                recommendations = content_model.get_recommendations_by_preferences(
                    user_preferences=user_preferences,
                    num_recommendations=total_needed
                )

                if recommendations and len(recommendations) > 0:
                    logger.info(f"[WEEKLY_PLAN] ✅ CONTENT-BASED ML returned {len(recommendations)} exercises")
                    # Apply week-based shuffling for variety across weeks
                    shuffled_recommendations = self._apply_week_variety(recommendations, week_seed)
                    return shuffled_recommendations
                else:
                    logger.warning(f"[WEEKLY_PLAN] Content-based ML returned 0 recommendations")

            # ❌ STEP 3: Both ML models failed, use hardcoded fallback (last resort)
            logger.warning(f"[WEEKLY_PLAN] Both ML models failed, using hardcoded fallback")
            return self._get_fallback_exercises(total_needed, fitness_level, target_muscle_groups)

        except Exception as e:
            logger.error(f"[WEEKLY_PLAN] ML recommendation failed: {e}")
            return self._get_fallback_exercises(total_needed, fitness_level, target_muscle_groups)

    def _map_fitness_level_to_difficulty(self, fitness_level: str) -> int:
        """Map fitness level string to numeric difficulty (1-3)"""
        difficulty_map = {
            'beginner': 1,
            'intermediate': 2,
            'medium': 2,
            'advanced': 3,
            'expert': 3
        }
        return difficulty_map.get(fitness_level.lower() if fitness_level else 'beginner', 1)

    def _apply_week_variety(self, recommendations: List[Dict[str, Any]], week_seed: int) -> List[Dict[str, Any]]:
        """
        Apply week-based shuffling to ensure exercise variety across different weeks

        Uses the week number as a seed to deterministically shuffle recommendations.
        Same week = same shuffle order (consistent within week)
        Different week = different shuffle order (variety across weeks)

        Args:
            recommendations: List of exercise recommendations from ML models
            week_seed: Week number (1-53) to use as shuffle seed

        Returns:
            Shuffled list of recommendations (deterministic for same week_seed)
        """
        if not recommendations or len(recommendations) <= 1:
            return recommendations

        # Create a copy to avoid modifying the original list
        shuffled = recommendations.copy()

        # Use week_seed for deterministic shuffling
        # Same week_seed = same order, different week_seed = different order
        random.seed(week_seed)
        random.shuffle(shuffled)

        logger.info(f"[WEEKLY_PLAN] Applied week variety (seed: {week_seed})")
        logger.info(f"[WEEKLY_PLAN] First exercise after shuffle: {shuffled[0].get('exercise_name', 'Unknown')} (ID: {shuffled[0].get('exercise_id', '?')})")

        return shuffled

    def _get_fallback_exercises(
        self,
        count: int,
        fitness_level: str,
        target_muscle_groups: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate fallback exercises when content service is unavailable"""

        base_exercises = [
            {'exercise_id': 1, 'name': 'Burpees', 'muscle_group': 'full_body', 'difficulty': 'intermediate', 'estimated_calories_per_minute': 12},
            {'exercise_id': 2, 'name': 'Push-ups', 'muscle_group': 'upper_body', 'difficulty': 'beginner', 'estimated_calories_per_minute': 8},
            {'exercise_id': 3, 'name': 'Squats', 'muscle_group': 'lower_body', 'difficulty': 'beginner', 'estimated_calories_per_minute': 9},
            {'exercise_id': 4, 'name': 'Mountain Climbers', 'muscle_group': 'core', 'difficulty': 'intermediate', 'estimated_calories_per_minute': 10},
            {'exercise_id': 5, 'name': 'Jumping Jacks', 'muscle_group': 'full_body', 'difficulty': 'beginner', 'estimated_calories_per_minute': 7},
            {'exercise_id': 6, 'name': 'Lunges', 'muscle_group': 'lower_body', 'difficulty': 'beginner', 'estimated_calories_per_minute': 8},
            {'exercise_id': 7, 'name': 'Plank', 'muscle_group': 'core', 'difficulty': 'intermediate', 'estimated_calories_per_minute': 5},
            {'exercise_id': 8, 'name': 'High Knees', 'muscle_group': 'cardio', 'difficulty': 'beginner', 'estimated_calories_per_minute': 10},
            {'exercise_id': 9, 'name': 'Tricep Dips', 'muscle_group': 'upper_body', 'difficulty': 'intermediate', 'estimated_calories_per_minute': 7},
            {'exercise_id': 10, 'name': 'Jump Squats', 'muscle_group': 'lower_body', 'difficulty': 'advanced', 'estimated_calories_per_minute': 12},
            {'exercise_id': 11, 'name': 'Russian Twists', 'muscle_group': 'core', 'difficulty': 'intermediate', 'estimated_calories_per_minute': 6},
            {'exercise_id': 12, 'name': 'Box Jumps', 'muscle_group': 'lower_body', 'difficulty': 'advanced', 'estimated_calories_per_minute': 11},
            {'exercise_id': 13, 'name': 'Pull-ups', 'muscle_group': 'upper_body', 'difficulty': 'advanced', 'estimated_calories_per_minute': 9},
            {'exercise_id': 14, 'name': 'Bicycle Crunches', 'muscle_group': 'core', 'difficulty': 'beginner', 'estimated_calories_per_minute': 6},
            {'exercise_id': 15, 'name': 'Walking Lunges', 'muscle_group': 'lower_body', 'difficulty': 'beginner', 'estimated_calories_per_minute': 7},
        ]

        # Filter by fitness level if specified
        filtered = [ex for ex in base_exercises if ex['difficulty'] == fitness_level] if fitness_level != 'all' else base_exercises

        # If not enough, add from all levels
        if len(filtered) < count:
            filtered = base_exercises

        return filtered[:count]

    def _group_exercises_by_muscle_group(self, exercises: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group exercises by their primary muscle group"""
        grouped = {}

        for exercise in exercises:
            muscle_group = exercise.get('muscle_group', 'full_body')

            if muscle_group not in grouped:
                grouped[muscle_group] = []

            grouped[muscle_group].append(exercise)

        return grouped

    def _distribute_exercises_across_days(
        self,
        workout_days: List[str],
        grouped_exercises: Dict[str, List[Dict[str, Any]]],
        exercises_per_day: int,
        target_muscle_groups: List[str]
    ) -> Dict[str, Any]:
        """
        Distribute exercises across workout days with balanced muscle group focus

        Strategy:
        - Day 1: Upper body focus
        - Day 2: Lower body focus
        - Day 3: Core/Full body focus
        - Repeat pattern for additional days
        """
        weekly_plan = {}

        # Define focus rotation
        focus_rotation = ['upper_body', 'lower_body', 'core', 'full_body']

        # Initialize all days
        for day in self.days_of_week:
            if day in workout_days:
                weekly_plan[day] = {
                    'planned': True,
                    'rest_day': False,
                    'exercises': [],
                    'focus_areas': [],
                    'estimated_duration': 0,
                    'estimated_calories': 0
                }
            else:
                weekly_plan[day] = {
                    'planned': False,
                    'rest_day': True
                }

        # Distribute exercises
        exercise_pool = []
        for muscle_group, exercises in grouped_exercises.items():
            exercise_pool.extend(exercises)

        used_exercise_ids = set()
        workout_day_index = 0

        for day in workout_days:
            focus_area = focus_rotation[workout_day_index % len(focus_rotation)]
            workout_day_index += 1

            day_exercises = []
            focus_areas_set = set()

            # Try to get exercises matching focus area first
            focus_exercises = grouped_exercises.get(focus_area, [])

            for exercise in focus_exercises:
                if len(day_exercises) >= exercises_per_day:
                    break

                if exercise['exercise_id'] not in used_exercise_ids:
                    day_exercises.append(exercise)
                    used_exercise_ids.add(exercise['exercise_id'])
                    focus_areas_set.add(exercise.get('muscle_group', focus_area))

            # Fill remaining slots with any available exercises
            if len(day_exercises) < exercises_per_day:
                for exercise in exercise_pool:
                    if len(day_exercises) >= exercises_per_day:
                        break

                    if exercise['exercise_id'] not in used_exercise_ids:
                        day_exercises.append(exercise)
                        used_exercise_ids.add(exercise['exercise_id'])
                        focus_areas_set.add(exercise.get('muscle_group', 'full_body'))

            # Calculate duration and calories (Tabata: 4 min per exercise)
            duration = len(day_exercises) * 4
            calories = sum(ex.get('estimated_calories_per_minute', 8) * 4 for ex in day_exercises)

            weekly_plan[day].update({
                'exercises': day_exercises,
                'focus_areas': list(focus_areas_set),
                'estimated_duration': duration,
                'estimated_calories': calories
            })

        return weekly_plan

    def _calculate_plan_metadata(self, weekly_plan: Dict[str, Any], time_constraints: int) -> Dict[str, Any]:
        """Calculate plan statistics"""
        total_exercises = 0
        total_duration = 0
        total_calories = 0
        workout_days_count = 0
        rest_days_count = 0
        focus_distribution = {}

        for day, day_data in weekly_plan.items():
            if day_data.get('planned', False):
                workout_days_count += 1
                total_exercises += len(day_data.get('exercises', []))
                total_duration += day_data.get('estimated_duration', 0)
                total_calories += day_data.get('estimated_calories', 0)

                for focus in day_data.get('focus_areas', []):
                    focus_distribution[focus] = focus_distribution.get(focus, 0) + 1
            else:
                rest_days_count += 1

        return {
            'total_exercises': total_exercises,
            'estimated_weekly_duration': total_duration,
            'estimated_weekly_calories': total_calories,
            'workout_days': workout_days_count,
            'rest_days': rest_days_count,
            'focus_distribution': focus_distribution,
            'average_duration_per_workout': total_duration // workout_days_count if workout_days_count > 0 else 0,
            'confidence_score': 0.85  # Placeholder, can be enhanced with ML confidence
        }
