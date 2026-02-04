"""
GroupWorkoutRecommender Service

Simplified group workout generator using existing model methods

RECOMMENDATION STRATEGY: Median-based
- Uses the MEDIAN fitness level of the group (not minimum)
- This ensures workouts are appropriately challenging for the majority
- Avoids under-challenging 80% of users by catering only to the weakest member
- Industry standard: Peloton, Apple Fitness+, CrossFit all use median/average approaches
"""

from typing import Dict, List, Any
import logging
import statistics

logger = logging.getLogger(__name__)


class GroupWorkoutRecommender:
    """
    Generate group workouts using existing ML models
    """

    def __init__(self, model_manager, user_profile_builder):
        self.model_manager = model_manager
        self.profile_builder = user_profile_builder
        self.content_model = model_manager.content_model
        self.collaborative_model = model_manager.collaborative_model

        # Import content service for fetching real exercise data
        from services.content_service import ContentService
        self.content_service = ContentService()

    def get_group_recommendations(
        self,
        user_ids: List[int],
        token: str,
        workout_format: str = "tabata",
        target_exercises: int = 8,
        include_alternatives: bool = False,
        num_alternatives: int = 6
    ) -> Dict[str, Any]:
        """
        Generate group workout recommendations

        Strategy:
        1. Get user profiles for fitness level analysis
        2. Use collaborative filtering to get exercises all members might like
        3. Format as Tabata workout

        NEW: Workout Customization Support
        - include_alternatives: If True, include extra exercises for voting/customization
        - num_alternatives: Number of alternative exercises (default: 6)

        Response includes (when include_alternatives=True):
        - exercises: Primary 8 exercises (backward compatible)
        - recommended_exercises: Same as exercises (NEW)
        - alternative_pool: Additional exercises for voting (NEW)
        """

        try:
            # STEP 1: Build profiles for all members
            logger.info(f"Building profiles for {len(user_ids)} users...")
            user_profiles = []
            for user_id in user_ids:
                try:
                    profile = self.profile_builder.get_user_profile(user_id, token)
                    user_profiles.append(profile)
                except Exception as e:
                    logger.warning(f"Failed to get profile for user {user_id}: {e}")

            if not user_profiles:
                raise Exception("Failed to build any user profiles")

            # STEP 2: Analyze group composition
            # Log each user's fitness level for debugging
            for profile in user_profiles:
                logger.info(f"  User {profile.get('user_id')}: fitness_level_numeric={profile.get('fitness_level_numeric', 'NOT SET')}")

            fitness_levels = [p.get('fitness_level_numeric', 1) for p in user_profiles]
            logger.info(f"All fitness levels: {fitness_levels}")
            avg_level = sum(fitness_levels) / len(fitness_levels)
            min_level = min(fitness_levels)
            max_level = max(fitness_levels)

            # Calculate MEDIAN fitness level (core of our recommendation strategy)
            # Median ensures workouts are appropriate for the majority of the group
            median_level = int(statistics.median(fitness_levels))

            group_analysis = {
                'avg_fitness_level': avg_level,
                'median_fitness_level': median_level,  # PRIMARY metric for exercise selection
                'min_fitness_level': min_level,
                'max_fitness_level': max_level,
                'fitness_level_range': 'homogeneous' if (max_level - min_level <= 1) else 'mixed',
                'total_members': len(user_profiles)
            }

            logger.info(f"Group analysis: {group_analysis} (using MEDIAN level {median_level} for exercise selection)")

            # Calculate total exercises needed (primary + alternatives if requested)
            total_needed = target_exercises
            if include_alternatives:
                total_needed = target_exercises + num_alternatives
                logger.info(f"Including alternatives: {target_exercises} primary + {num_alternatives} alternatives = {total_needed} total")

            # STEP 3: Get recommendations for each member using collaborative filtering
            all_exercises = {}  # {exercise_id: count}

            for profile in user_profiles:
                user_id = profile['user_id']
                try:
                    # Get collaborative recommendations (request more to have buffer)
                    recs = self.collaborative_model.get_recommendations(
                        user_id=user_id,
                        num_recommendations=total_needed * 2
                    )

                    # Count exercise occurrences
                    for rec in recs:
                        ex_id = rec.get('exercise_id') or rec.get('workout_id')
                        if ex_id:
                            all_exercises[ex_id] = all_exercises.get(ex_id, 0) + 1

                except Exception as e:
                    logger.warning(f"Failed to get recommendations for user {user_id}: {e}")

            # STEP 4: Select top exercises that multiple users like
            logger.info(f"Collaborative filtering found {len(all_exercises)} unique exercises across all users")
            if not all_exercises:
                # Fallback: use content-based with MEDIAN fitness level
                logger.warning("No collaborative recommendations, using content-based fallback with MEDIAN level")

                # Find member closest to the MEDIAN fitness level
                median_level = group_analysis['median_fitness_level']
                median_member = min(
                    user_profiles,
                    key=lambda p: abs(p.get('fitness_level_numeric', 1) - median_level)
                )
                logger.info(f"Using median-based selection (median level {median_level}) for exercise selection")
                all_selected_exercises = self._get_content_based_exercises(median_member, total_needed, median_level)
            else:
                # Sort by how many users like each exercise
                sorted_exercises = sorted(all_exercises.items(), key=lambda x: x[1], reverse=True)
                selected_ids = [ex_id for ex_id, count in sorted_exercises[:total_needed]]

                # Get exercise details
                all_selected_exercises = self._get_exercise_details(selected_ids)

            # Split into primary exercises and alternatives
            exercises = all_selected_exercises[:target_exercises]
            alternative_pool = []
            if include_alternatives:
                alternative_pool = all_selected_exercises[target_exercises:target_exercises + num_alternatives]
                logger.info(f"Split: {len(exercises)} primary, {len(alternative_pool)} alternatives from {len(all_selected_exercises)} total selected")
                if len(alternative_pool) < num_alternatives:
                    logger.warning(f"Insufficient alternatives: got {len(alternative_pool)}, wanted {num_alternatives}. Total exercises available: {len(all_selected_exercises)}")

            # STEP 5: Format as workout
            if workout_format == "tabata":
                result = self._format_as_tabata(exercises, group_analysis, alternative_pool, include_alternatives)
            else:
                result = self._format_as_generic(exercises, group_analysis, alternative_pool, include_alternatives)

            logger.info(f"Successfully generated {workout_format} workout with {len(exercises)} exercises" +
                       (f" and {len(alternative_pool)} alternatives" if include_alternatives else ""))
            return result

        except Exception as e:
            logger.error(f"Error generating group workout: {e}")
            raise

    def _get_content_based_exercises(self, user_profile: Dict, count: int, target_difficulty: int = None) -> List[Dict]:
        """
        Get exercises from database based on MEDIAN fitness level strategy.

        Strategy: Select exercises at the MEDIAN difficulty level of the group.
        This ensures workouts are appropriately challenging for the majority,
        rather than being too easy (catering to minimum) or too hard (catering to maximum).
        """
        try:
            # Get all exercises from content service
            all_exercises = self.content_service.get_all_exercises()

            if not all_exercises:
                logger.warning("No exercises found in content service, using defaults")
                return self._get_default_exercises(count)

            # Use MEDIAN fitness level for group workouts
            # This ensures the workout is appropriate for the majority of the group
            if target_difficulty is None:
                target_difficulty = user_profile.get('fitness_level_numeric', 1)

            equipment = user_profile.get('available_equipment', ['bodyweight'])

            logger.info(f"MEDIAN-BASED selection: target_difficulty={target_difficulty}, equipment={equipment}, requesting {count} exercises")

            # Filter exercises suitable for the MEDIAN level
            suitable_exercises = []
            for ex in all_exercises:
                # Convert difficulty to int (handles both string names and numeric values)
                raw_difficulty = ex.get('difficulty_level', 1)
                ex_difficulty = self._map_difficulty_to_numeric(raw_difficulty)

                ex_equipment = ex.get('equipment_needed', 'bodyweight')
                # Normalize equipment value
                if ex_equipment is None or ex_equipment.lower() in ['none', '']:
                    ex_equipment = 'bodyweight'

                # MEDIAN STRATEGY: Include exercises at or below the median level
                # This challenges the majority while still being achievable for lower-level members
                equipment_ok = ex_equipment.lower() in [e.lower() for e in equipment] or ex_equipment.lower() == 'bodyweight'
                if ex_difficulty <= target_difficulty and equipment_ok:
                    suitable_exercises.append({
                        'exercise_id': ex.get('id') or ex.get('exercise_id'),
                        'exercise_name': ex.get('exercise_name', ex.get('name', 'Unknown Exercise')),
                        'difficulty_level': ex_difficulty,
                        'estimated_calories_burned': ex.get('estimated_calories_burned', 100),
                        'target_muscle_group': ex.get('target_muscle_group', 'core'),
                        'equipment_needed': ex_equipment
                    })

            # Count exercises by difficulty for debugging
            difficulty_counts = {}
            for ex in all_exercises:
                d = self._map_difficulty_to_numeric(ex.get('difficulty_level', 1))
                difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
            logger.info(f"Exercise difficulty distribution: {difficulty_counts}")
            logger.info(f"Found {len(suitable_exercises)} suitable exercises from {len(all_exercises)} total (target_diff={target_difficulty}, equipment={equipment})")

            # If we have suitable exercises, prioritize by difficulty
            import random
            if suitable_exercises:
                # MEDIAN-OPTIMIZED SELECTION:
                # - 70% at median level (appropriately challenging for majority)
                # - 30% one level below (provides variety and active recovery)
                exact_level = [ex for ex in suitable_exercises if ex['difficulty_level'] == target_difficulty]
                one_below = [ex for ex in suitable_exercises if ex['difficulty_level'] == target_difficulty - 1] if target_difficulty > 1 else []

                selected = []

                # 70% at median level (challenging for the majority)
                if exact_level:
                    random.shuffle(exact_level)
                    needed = int(count * 0.7)
                    selected.extend(exact_level[:needed])

                # 30% one level below (for variety/recovery moments)
                if one_below and len(selected) < count:
                    random.shuffle(one_below)
                    needed = count - len(selected)
                    selected.extend(one_below[:needed])

                # Fill remaining with any suitable exercises
                if len(selected) < count:
                    remaining = [ex for ex in suitable_exercises if ex not in selected]
                    random.shuffle(remaining)
                    selected.extend(remaining[:count - len(selected)])

                logger.info(f"MEDIAN-BASED: Selected {len(selected)} exercises - {len([e for e in selected if e['difficulty_level'] == target_difficulty])} at median level {target_difficulty}")
                return selected[:count]
            else:
                logger.warning("No suitable exercises found, using defaults")
                return self._get_default_exercises(count)

        except Exception as e:
            logger.error(f"Content-based fallback failed: {e}")
            return self._get_default_exercises(count)

    def _get_exercise_details(self, exercise_ids: List[int]) -> List[Dict]:
        """Get REAL exercise details from content service"""
        exercises = []

        for ex_id in exercise_ids:
            try:
                # Fetch real exercise data from content service
                exercise_data = self.content_service.get_exercise_by_id(ex_id)

                if exercise_data:
                    exercises.append({
                        'exercise_id': ex_id,
                        'exercise_name': exercise_data.get('exercise_name', f'Exercise {ex_id}'),
                        'difficulty_level': self._map_difficulty_to_numeric(exercise_data.get('difficulty_level', 1)),
                        'estimated_calories_burned': exercise_data.get('estimated_calories_burned', 100),
                        'target_muscle_group': exercise_data.get('target_muscle_group', 'core'),
                        'equipment_needed': exercise_data.get('equipment_needed', 'bodyweight')
                    })
                else:
                    # Fallback if exercise not found
                    logger.warning(f"Exercise {ex_id} not found, using placeholder")
                    exercises.append({
                        'exercise_id': ex_id,
                        'exercise_name': f'Exercise {ex_id}',
                        'difficulty_level': 1,
                        'estimated_calories_burned': 100,
                        'target_muscle_group': 'core'
                    })
            except Exception as e:
                logger.error(f"Failed to get exercise {ex_id}: {e}")
                # Add placeholder on error
                exercises.append({
                    'exercise_id': ex_id,
                    'exercise_name': f'Exercise {ex_id}',
                    'difficulty_level': 1,
                    'estimated_calories_burned': 100,
                    'target_muscle_group': 'core'
                })

        return exercises

    def _get_default_exercises(self, count: int) -> List[Dict]:
        """Default exercises as last resort - includes enough for alternatives"""
        default_exercises = [
            # Primary 8 exercises
            {'name': 'Burpees', 'difficulty': 2, 'muscle': 'full_body', 'calories': 120},
            {'name': 'Mountain Climbers', 'difficulty': 2, 'muscle': 'core', 'calories': 100},
            {'name': 'Jump Squats', 'difficulty': 2, 'muscle': 'lower_body', 'calories': 110},
            {'name': 'High Knees', 'difficulty': 1, 'muscle': 'lower_body', 'calories': 90},
            {'name': 'Push-ups', 'difficulty': 2, 'muscle': 'upper_body', 'calories': 80},
            {'name': 'Plank', 'difficulty': 1, 'muscle': 'core', 'calories': 50},
            {'name': 'Lunges', 'difficulty': 1, 'muscle': 'lower_body', 'calories': 70},
            {'name': 'Jumping Jacks', 'difficulty': 1, 'muscle': 'full_body', 'calories': 80},
            # Alternative exercises (6 more for voting/customization)
            {'name': 'Squats', 'difficulty': 1, 'muscle': 'lower_body', 'calories': 75},
            {'name': 'Sit-ups', 'difficulty': 1, 'muscle': 'core', 'calories': 60},
            {'name': 'Bicycle Crunches', 'difficulty': 2, 'muscle': 'core', 'calories': 70},
            {'name': 'Box Jumps', 'difficulty': 3, 'muscle': 'lower_body', 'calories': 130},
            {'name': 'Diamond Push-ups', 'difficulty': 3, 'muscle': 'upper_body', 'calories': 85},
            {'name': 'Russian Twists', 'difficulty': 2, 'muscle': 'core', 'calories': 65},
        ]

        exercises = []
        for i, ex in enumerate(default_exercises[:count]):
            exercises.append({
                'exercise_id': i + 1,
                'exercise_name': ex['name'],
                'difficulty_level': ex['difficulty'],
                'estimated_calories_burned': ex['calories'],
                'target_muscle_group': ex['muscle']
            })
        return exercises

    def _map_difficulty_to_numeric(self, difficulty) -> int:
        """
        Map difficulty level to numeric value (1-3).
        Handles both string names ('beginner', 'medium', 'expert') and numeric values.
        """
        if difficulty is None:
            return 1

        # If already an integer, validate and return
        if isinstance(difficulty, int):
            return max(1, min(3, difficulty))

        # Try to convert numeric string
        if isinstance(difficulty, str):
            # Handle string difficulty names from content service
            difficulty_lower = difficulty.lower().strip()
            mapping = {
                'beginner': 1,
                'easy': 1,
                'novice': 1,
                '1': 1,
                'medium': 2,
                'intermediate': 2,
                'moderate': 2,
                '2': 2,
                'expert': 3,
                'advanced': 3,
                'hard': 3,
                'difficult': 3,
                '3': 3,
            }
            return mapping.get(difficulty_lower, 1)

        # Default fallback
        try:
            return int(difficulty)
        except (ValueError, TypeError):
            return 1

    def _format_as_tabata(self, exercises: List[Dict], group_analysis: Dict,
                          alternative_pool: List[Dict] = None, include_alternatives: bool = False) -> Dict[str, Any]:
        """
        Format as Tabata workout

        NEW: Includes alternative_pool when include_alternatives=True
        - exercises: Primary 8 exercises (backward compatible)
        - recommended_exercises: Same as exercises (NEW)
        - alternative_pool: Additional exercises for voting (NEW)
        """
        primary_exercises = exercises[:8]  # Ensure max 8 exercises

        result = {
            'workout_format': 'tabata',
            'exercises': primary_exercises,  # KEPT for backward compatibility
            'recommended_exercises': primary_exercises,  # NEW: Same as exercises
            'group_analysis': group_analysis,
            'tabata_structure': {
                'rounds': 8,
                'work_duration_seconds': 20,
                'rest_duration_seconds': 10,
                'total_duration_minutes': len(primary_exercises) * 4,
                'exercises_per_round': 1
            }
        }

        # Add alternative pool if requested
        if include_alternatives:
            result['alternative_pool'] = alternative_pool or []
            result['alternative_count'] = len(alternative_pool) if alternative_pool else 0

        return result

    def _format_as_generic(self, exercises: List[Dict], group_analysis: Dict,
                           alternative_pool: List[Dict] = None, include_alternatives: bool = False) -> Dict[str, Any]:
        """
        Format as generic workout

        NEW: Includes alternative_pool when include_alternatives=True
        """
        result = {
            'workout_format': 'generic',
            'exercises': exercises,  # KEPT for backward compatibility
            'recommended_exercises': exercises,  # NEW: Same as exercises
            'group_analysis': group_analysis,
            'recommended_sets': 3,
            'recommended_reps': '8-12 or 30 seconds',
            'rest_between_exercises': '30-60 seconds'
        }

        # Add alternative pool if requested
        if include_alternatives:
            result['alternative_pool'] = alternative_pool or []
            result['alternative_count'] = len(alternative_pool) if alternative_pool else 0

        return result
