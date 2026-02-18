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

            # Determine exercises per day based on fitness level, time, and session count (progressive overload)
            session_count = self._get_user_session_count(user_id)
            exercises_per_day = self._calculate_exercises_per_day(fitness_level, time_constraints, session_count)

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

    def _get_user_session_count(self, user_id: int) -> int:
        """Get count of individual workout sessions completed by user (for progressive overload)"""
        try:
            from services.tracking_service import TrackingService
            tracking_service = TrackingService()
            sessions = tracking_service.get_user_workout_sessions(user_id, limit=50)
            if not sessions:
                return 0
            # Count only individual sessions
            return sum(1 for s in sessions if s.get('session_type', 'individual') != 'group')
        except Exception as e:
            logger.warning(f"[WEEKLY_PLAN] Could not get session count for user {user_id}: {e}")
            return 0

    def _calculate_exercises_per_day(self, fitness_level: str, time_constraints: int, session_count: int = 0) -> int:
        """
        Calculate how many exercises fit using progressive overload.

        Tabata: 4 minutes per exercise (8 rounds × 30s)

        Progressive overload ranges (per professor requirements):
          Beginner:     0-5 sessions→4, 6-15→5, 16+→6
          Intermediate: 0-5 sessions→6, 6-15→7, 16+→8
          Advanced:     0-5 sessions→8, 6-15→10, 16+→12
        """
        level = (fitness_level or 'beginner').lower()

        if level == 'beginner':
            if session_count < 6:
                base_exercises = 4
            elif session_count < 16:
                base_exercises = 5
            else:
                base_exercises = 6
        elif level in ('intermediate', 'medium'):
            if session_count < 6:
                base_exercises = 6
            elif session_count < 16:
                base_exercises = 7
            else:
                base_exercises = 8
        elif level in ('advanced', 'expert'):
            if session_count < 6:
                base_exercises = 8
            elif session_count < 16:
                base_exercises = 10
            else:
                base_exercises = 12
        else:
            base_exercises = 4  # Default to beginner

        logger.info(f"[WEEKLY_PLAN] Progressive overload: level={level}, sessions={session_count} → {base_exercises} exercises/day")

        # Adjust for time constraints (4 min per exercise in Tabata)
        max_exercises_by_time = time_constraints // 4

        # Return the minimum to respect both fitness level and time
        return min(base_exercises, max_exercises_by_time)

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
                                  num_recommendations: int) -> List[Dict]:
        """
        Filter recommendations by user's fitness level.

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
        logger.info(f"[WEEKLY_PLAN] Filtering recommendations for fitness level '{user_fitness_level}' (difficulty: {target_difficulty})")

        # Filter exact match first (highest priority)
        exact_match = [r for r in recommendations if r.get('difficulty_level') == target_difficulty]

        # STRICT POLICY FOR BEGINNERS: Beginners should ONLY get beginner exercises
        if target_difficulty == 1:
            filtered_recommendations = exact_match
            if len(filtered_recommendations) < num_recommendations:
                logger.warning(f"[WEEKLY_PLAN] Only found {len(filtered_recommendations)} beginner exercises out of {num_recommendations} requested!")
        elif target_difficulty == 2:
            # INTERMEDIATE: Prefer difficulty 2, allow difficulty 3 as supplement
            if len(exact_match) >= num_recommendations:
                filtered_recommendations = exact_match
                logger.info(f"[WEEKLY_PLAN] Found {len(exact_match)} exact difficulty 2 matches for intermediate")
            else:
                # Fill remaining with difficulty 3 exercises
                remaining_needed = num_recommendations - len(exact_match)
                harder_exercises = [r for r in recommendations if r.get('difficulty_level') == 3]
                filtered_recommendations = exact_match + harder_exercises[:remaining_needed]
                logger.info(f"[WEEKLY_PLAN] Intermediate: {len(exact_match)} exact + {min(remaining_needed, len(harder_exercises))} harder exercises")
        else:
            # ADVANCED (difficulty 3): STRICTLY prefer difficulty 3, use difficulty 2 ONLY as fallback
            if len(exact_match) >= num_recommendations:
                filtered_recommendations = exact_match
                logger.info(f"[WEEKLY_PLAN] Found {len(exact_match)} exact difficulty 3 matches for advanced")
            else:
                # Fill remaining with difficulty 2 exercises (NOT preferred, just fallback)
                remaining_needed = num_recommendations - len(exact_match)
                easier_exercises = [r for r in recommendations if r.get('difficulty_level') == 2]
                filtered_recommendations = exact_match + easier_exercises[:remaining_needed]
                logger.info(f"[WEEKLY_PLAN] Advanced: {len(exact_match)} exact (difficulty 3) + {min(remaining_needed, len(easier_exercises))} fallback (difficulty 2)")

        # Return ALL exercises that pass the filter (no hard cap).
        # The distribution algorithm needs the largest possible pool to fill
        # every workout day, especially for 7-day plans.
        # If the pool is still smaller than needed, Change 2 (cycling) handles it.
        logger.info(f"[WEEKLY_PLAN] After filtering: {len(filtered_recommendations)} exercises available for '{user_fitness_level}' (pool, min needed: {num_recommendations})")
        return filtered_recommendations

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
        1. Try Hybrid ML with dynamic weights and fitness-level filtering
        2. If fails → Try Content-Based ML (100% content, STILL ML!)
        3. If both fail → Use hardcoded fallback (last resort)

        Returns list of exercises with ML-powered personalization
        """
        try:
            # ✅ STEP 1: Try Hybrid ML model first (same as Dashboard/Workouts!)
            if not self.model_manager:
                logger.error("[WEEKLY_PLAN] Model manager not available, skipping ML models")
                return self._get_fallback_exercises(total_needed, fitness_level, target_muscle_groups)

            # Get user rating count for dynamic weight calculation
            from services.tracking_service import TrackingService
            tracking_service = TrackingService()
            try:
                user_ratings = tracking_service.get_user_exercise_ratings(user_id)
                user_rating_count = len(user_ratings) if user_ratings else 0
            except:
                user_rating_count = 0
                logger.warning(f"[WEEKLY_PLAN] Could not fetch user ratings, defaulting to 0")

            # Calculate dynamic weights (Netflix/Spotify approach)
            content_weight, collaborative_weight, confidence_level = self._calculate_dynamic_weights(user_rating_count)
            logger.info(f"[WEEKLY_PLAN] User {user_id} has {user_rating_count} ratings - Dynamic weights: Content={content_weight}, CF={collaborative_weight} (confidence: {confidence_level})")

            hybrid_model = self.model_manager.get_hybrid_model()
            if hybrid_model:
                logger.info(f"[WEEKLY_PLAN] Trying HYBRID ML model for user {user_id}")

                # Request 5x more recommendations to have buffer for filtering
                buffer_multiplier = 5
                recommendations = hybrid_model.get_recommendations(
                    user_id=user_id,
                    user_preferences={
                        'fitness_level': fitness_level,
                        'target_muscle_groups': target_muscle_groups if target_muscle_groups else [],
                        'goals': goals if goals else []
                    },
                    num_recommendations=total_needed * buffer_multiplier,
                    content_weight=content_weight,
                    collaborative_weight=collaborative_weight
                )

                if recommendations and len(recommendations) > 0:
                    logger.info(f"[WEEKLY_PLAN] ✅ HYBRID ML returned {len(recommendations)} exercises before filtering")

                    # Apply fitness-level filtering (STRICT for beginners)
                    filtered_recommendations = self._filter_by_fitness_level(
                        recommendations,
                        fitness_level,
                        total_needed
                    )

                    if filtered_recommendations and len(filtered_recommendations) > 0:
                        logger.info(f"[WEEKLY_PLAN] ✅ After fitness filtering: {len(filtered_recommendations)} exercises")
                        # Apply week-based shuffling for variety across weeks AND unique per user
                        shuffled_recommendations = self._apply_week_variety(filtered_recommendations, week_seed, user_id)
                        return shuffled_recommendations
                    else:
                        logger.warning(f"[WEEKLY_PLAN] Fitness filtering resulted in 0 exercises, falling back to content-based")
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
                # Request 5x more for filtering buffer
                recommendations = content_model.get_recommendations_by_preferences(
                    user_preferences=user_preferences,
                    num_recommendations=total_needed * 5
                )

                if recommendations and len(recommendations) > 0:
                    logger.info(f"[WEEKLY_PLAN] ✅ CONTENT-BASED ML returned {len(recommendations)} exercises before filtering")

                    # Apply fitness-level filtering (STRICT for beginners)
                    filtered_recommendations = self._filter_by_fitness_level(
                        recommendations,
                        fitness_level,
                        total_needed
                    )

                    if filtered_recommendations and len(filtered_recommendations) > 0:
                        logger.info(f"[WEEKLY_PLAN] ✅ After fitness filtering: {len(filtered_recommendations)} exercises")
                        # Apply week-based shuffling for variety across weeks AND unique per user
                        shuffled_recommendations = self._apply_week_variety(filtered_recommendations, week_seed, user_id)
                        return shuffled_recommendations
                    else:
                        logger.warning(f"[WEEKLY_PLAN] Content-based filtering resulted in 0 exercises")
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

    def _apply_week_variety(self, recommendations: List[Dict[str, Any]], week_seed: int, user_id: int = 0) -> List[Dict[str, Any]]:
        """
        Apply week-based shuffling to ensure exercise variety across different weeks AND users

        Uses BOTH the week number AND user_id as a combined seed to ensure:
        - Same user, same week = same shuffle order (consistent within week for user)
        - Same user, different week = different shuffle order (variety across weeks)
        - Different users, same week = DIFFERENT shuffle order (unique exercises per user)

        Args:
            recommendations: List of exercise recommendations from ML models
            week_seed: Week number (1-53) to use as shuffle seed
            user_id: User ID to make shuffle unique per user

        Returns:
            Shuffled list of recommendations (deterministic for same week_seed + user_id)
        """
        if not recommendations or len(recommendations) <= 1:
            return recommendations

        # Create a copy to avoid modifying the original list
        shuffled = recommendations.copy()

        # Combine week_seed and user_id for unique shuffle per user per week
        # This ensures different users get different exercise orders
        combined_seed = (week_seed * 10000) + (user_id % 10000)
        random.seed(combined_seed)
        random.shuffle(shuffled)

        logger.info(f"[WEEKLY_PLAN] Applied week variety (week_seed: {week_seed}, user_id: {user_id}, combined_seed: {combined_seed})")
        logger.info(f"[WEEKLY_PLAN] First exercise after shuffle: {shuffled[0].get('exercise_name', 'Unknown')} (ID: {shuffled[0].get('exercise_id', '?')})")

        # Normalize exercise data to ensure calorie fields are present
        return self._normalize_exercise_data(shuffled)

    def _normalize_exercise_data(self, exercises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize exercise data to ensure consistent field names for calories.

        The hybrid model returns 'calories_per_minute' but the Planning service
        and client expect 'estimated_calories_per_minute' and 'estimated_calories_burned'.

        Tabata protocol: 4 minutes per exercise (8 rounds × 30 seconds)
        """
        TABATA_DURATION_MINUTES = 4
        DEFAULT_CALORIES_PER_MINUTE = 8  # Default if not provided

        normalized = []
        for ex in exercises:
            # Get calories per minute from various possible field names
            cal_per_min = (
                ex.get('estimated_calories_per_minute') or
                ex.get('calories_per_minute') or
                ex.get('calories_burned_per_minute') or
                DEFAULT_CALORIES_PER_MINUTE
            )

            # Calculate estimated calories burned for Tabata (4 min)
            estimated_calories = int(cal_per_min * TABATA_DURATION_MINUTES)

            # Create normalized exercise with all expected fields
            normalized_ex = {
                **ex,  # Keep all original fields
                'estimated_calories_per_minute': cal_per_min,
                'estimated_calories_burned': estimated_calories,
                'default_duration_seconds': TABATA_DURATION_MINUTES * 60,  # 240 seconds
            }
            normalized.append(normalized_ex)

        logger.info(f"[WEEKLY_PLAN] Normalized {len(normalized)} exercises with calorie data")
        return normalized

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
        """Group exercises by their primary muscle group.

        Accepts both 'target_muscle_group' (content service / ML model field)
        and 'muscle_group' (fallback / legacy field name) so the focus rotation
        (upper_body → lower_body → core → full_body) actually works.
        """
        grouped = {}

        for exercise in exercises:
            # Try both field names — content service uses 'target_muscle_group'
            muscle_group = (
                exercise.get('target_muscle_group')
                or exercise.get('muscle_group')
                or 'full_body'
            )

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
        # Tracks how many times each exercise has been used across all days.
        # Used by the cycling fallback so the least-repeated exercises get reused
        # first, maximising variety when the unique pool is exhausted.
        exercise_use_count: Dict[int, int] = {}
        workout_day_index = 0

        def _get_muscle(ex: Dict) -> str:
            return (
                ex.get('target_muscle_group')
                or ex.get('muscle_group')
                or 'full_body'
            )

        for day in workout_days:
            focus_area = focus_rotation[workout_day_index % len(focus_rotation)]
            workout_day_index += 1

            day_exercises = []
            focus_areas_set = set()

            # ── Pass 1: Focus-area exercises not yet used this week ────────────
            focus_exercises = grouped_exercises.get(focus_area, [])

            for exercise in focus_exercises:
                if len(day_exercises) >= exercises_per_day:
                    break

                if exercise['exercise_id'] not in used_exercise_ids:
                    day_exercises.append(exercise)
                    used_exercise_ids.add(exercise['exercise_id'])
                    exercise_use_count[exercise['exercise_id']] = 1
                    focus_areas_set.add(_get_muscle(exercise))

            # ── Pass 2: Any pool exercise not yet used this week ───────────────
            if len(day_exercises) < exercises_per_day:
                for exercise in exercise_pool:
                    if len(day_exercises) >= exercises_per_day:
                        break

                    if exercise['exercise_id'] not in used_exercise_ids:
                        day_exercises.append(exercise)
                        used_exercise_ids.add(exercise['exercise_id'])
                        exercise_use_count[exercise['exercise_id']] = 1
                        focus_areas_set.add(_get_muscle(exercise))

            # ── Pass 3: Cycling fallback (pool exhausted) ─────────────────────
            # When the unique pool runs out (common with 7-day plans or
            # progressive overload counts), allow exercise reuse.
            # Sort by use count ascending so the least-repeated exercises
            # are picked first — maximises perceived variety.
            if len(day_exercises) < exercises_per_day:
                already_in_today = {ex['exercise_id'] for ex in day_exercises}
                sorted_by_use = sorted(
                    exercise_pool,
                    key=lambda ex: exercise_use_count.get(ex['exercise_id'], 0)
                )
                for exercise in sorted_by_use:
                    if len(day_exercises) >= exercises_per_day:
                        break
                    ex_id = exercise['exercise_id']
                    if ex_id not in already_in_today:   # never repeat within same day
                        day_exercises.append(exercise)
                        exercise_use_count[ex_id] = exercise_use_count.get(ex_id, 0) + 1
                        focus_areas_set.add(_get_muscle(exercise))

                if len(day_exercises) < exercises_per_day:
                    logger.warning(
                        f"[WEEKLY_PLAN] Could only fill {len(day_exercises)}/{exercises_per_day} "
                        f"exercises for {day} even after cycling — exercise pool too small."
                    )

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
