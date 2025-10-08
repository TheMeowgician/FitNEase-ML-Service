"""
GroupWorkoutRecommender Service

Simplified group workout generator using existing model methods
"""

from typing import Dict, List, Any
import logging

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
        target_exercises: int = 8
    ) -> Dict[str, Any]:
        """
        Generate group workout recommendations

        Strategy:
        1. Get user profiles for fitness level analysis
        2. Use collaborative filtering to get exercises all members might like
        3. Format as Tabata workout
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
            fitness_levels = [p.get('fitness_level_numeric', 1) for p in user_profiles]
            avg_level = sum(fitness_levels) / len(fitness_levels)
            min_level = min(fitness_levels)
            max_level = max(fitness_levels)

            group_analysis = {
                'avg_fitness_level': avg_level,
                'min_fitness_level': min_level,
                'max_fitness_level': max_level,
                'fitness_level_range': 'homogeneous' if (max_level - min_level <= 1) else 'mixed',
                'total_members': len(user_profiles)
            }

            logger.info(f"Group analysis: {group_analysis}")

            # STEP 3: Get recommendations for each member using collaborative filtering
            all_exercises = {}  # {exercise_id: count}

            for profile in user_profiles:
                user_id = profile['user_id']
                try:
                    # Get collaborative recommendations
                    recs = self.collaborative_model.get_recommendations(
                        user_id=user_id,
                        num_recommendations=target_exercises * 2
                    )

                    # Count exercise occurrences
                    for rec in recs:
                        ex_id = rec.get('exercise_id') or rec.get('workout_id')
                        if ex_id:
                            all_exercises[ex_id] = all_exercises.get(ex_id, 0) + 1

                except Exception as e:
                    logger.warning(f"Failed to get recommendations for user {user_id}: {e}")

            # STEP 4: Select top exercises that multiple users like
            if not all_exercises:
                # Fallback: use content-based for the WEAKEST member (lowest fitness level)
                logger.warning("No collaborative recommendations, using content-based fallback")
                # Find the member with LOWEST fitness level (most restrictive)
                weakest_member = min(user_profiles, key=lambda p: p.get('fitness_level_numeric', 1))
                logger.info(f"Using weakest member (fitness level {weakest_member.get('fitness_level_numeric')}) for exercise selection")
                exercises = self._get_content_based_exercises(weakest_member, target_exercises, group_analysis['min_fitness_level'])
            else:
                # Sort by how many users like each exercise
                sorted_exercises = sorted(all_exercises.items(), key=lambda x: x[1], reverse=True)
                selected_ids = [ex_id for ex_id, count in sorted_exercises[:target_exercises]]

                # Get exercise details
                exercises = self._get_exercise_details(selected_ids)

            # STEP 5: Format as workout
            if workout_format == "tabata":
                result = self._format_as_tabata(exercises, group_analysis)
            else:
                result = self._format_as_generic(exercises, group_analysis)

            logger.info(f"Successfully generated {workout_format} workout with {len(exercises)} exercises")
            return result

        except Exception as e:
            logger.error(f"Error generating group workout: {e}")
            raise

    def _get_content_based_exercises(self, user_profile: Dict, count: int, max_difficulty: int = None) -> List[Dict]:
        """Fallback: Get REAL exercises from database that are safe for the group"""
        try:
            # Get all exercises from content service
            all_exercises = self.content_service.get_all_exercises()

            if not all_exercises:
                logger.warning("No exercises found in content service, using defaults")
                return self._get_default_exercises(count)

            # For group workouts, use the MINIMUM fitness level (weakest member)
            # For single users, use their actual level
            if max_difficulty is None:
                max_difficulty = user_profile.get('fitness_level_numeric', 1)

            equipment = user_profile.get('available_equipment', ['bodyweight'])

            logger.info(f"Filtering exercises: max_difficulty={max_difficulty}, equipment={equipment}")

            # Filter exercises suitable for the WEAKEST member
            suitable_exercises = []
            for ex in all_exercises:
                # Convert difficulty to int (might be string from database)
                try:
                    ex_difficulty = int(ex.get('difficulty_level', 1))
                except (ValueError, TypeError):
                    ex_difficulty = 1

                ex_equipment = ex.get('equipment_needed', 'bodyweight')

                # CRITICAL: Only include exercises AT OR BELOW the weakest member's level
                # This ensures EVERYONE in the group can do the exercise
                if ex_difficulty <= max_difficulty and (ex_equipment in equipment or ex_equipment == 'bodyweight'):
                    suitable_exercises.append({
                        'exercise_id': ex.get('id') or ex.get('exercise_id'),
                        'exercise_name': ex.get('exercise_name', ex.get('name', 'Unknown Exercise')),
                        'difficulty_level': ex_difficulty,
                        'estimated_calories_burned': ex.get('estimated_calories_burned', 100),
                        'muscle_group': ex.get('target_muscle_group', 'core'),
                        'equipment_needed': ex_equipment
                    })

            # If we have suitable exercises, prioritize by difficulty
            import random
            if suitable_exercises:
                # SMART SELECTION: Prefer exercises at max_difficulty level, but include some easier ones for variety
                exact_level = [ex for ex in suitable_exercises if ex['difficulty_level'] == max_difficulty]
                one_below = [ex for ex in suitable_exercises if ex['difficulty_level'] == max_difficulty - 1] if max_difficulty > 1 else []

                selected = []

                # 60% at exact level (most challenging for the group)
                if exact_level:
                    random.shuffle(exact_level)
                    needed = int(count * 0.6)
                    selected.extend(exact_level[:needed])

                # 40% one level below (for variety/warmup)
                if one_below and len(selected) < count:
                    random.shuffle(one_below)
                    needed = count - len(selected)
                    selected.extend(one_below[:needed])

                # Fill remaining with any suitable exercises
                if len(selected) < count:
                    remaining = [ex for ex in suitable_exercises if ex not in selected]
                    random.shuffle(remaining)
                    selected.extend(remaining[:count - len(selected)])

                logger.info(f"Selected {len(selected)} exercises: {len(exact_level)} at level {max_difficulty}, {len(one_below)} at level {max_difficulty-1}")
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
                        'difficulty_level': exercise_data.get('difficulty_level', 1),
                        'estimated_calories_burned': exercise_data.get('estimated_calories_burned', 100),
                        'muscle_group': exercise_data.get('target_muscle_group', 'core'),
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
                        'muscle_group': 'core'
                    })
            except Exception as e:
                logger.error(f"Failed to get exercise {ex_id}: {e}")
                # Add placeholder on error
                exercises.append({
                    'exercise_id': ex_id,
                    'exercise_name': f'Exercise {ex_id}',
                    'difficulty_level': 1,
                    'estimated_calories_burned': 100,
                    'muscle_group': 'core'
                })

        return exercises

    def _get_default_exercises(self, count: int) -> List[Dict]:
        """Default exercises as last resort"""
        default_names = ['Burpees', 'Mountain Climbers', 'Jump Squats', 'High Knees',
                        'Push-ups', 'Plank', 'Lunges', 'Jumping Jacks']

        exercises = []
        for i in range(min(count, len(default_names))):
            exercises.append({
                'exercise_id': i + 1,
                'exercise_name': default_names[i],
                'difficulty_level': 1,
                'estimated_calories_burned': 100,
                'muscle_group': 'core'
            })
        return exercises

    def _format_as_tabata(self, exercises: List[Dict], group_analysis: Dict) -> Dict[str, Any]:
        """Format as Tabata workout"""
        return {
            'workout_format': 'tabata',
            'exercises': exercises[:8],  # Ensure max 8 exercises
            'group_analysis': group_analysis,
            'tabata_structure': {
                'rounds': 8,
                'work_duration_seconds': 20,
                'rest_duration_seconds': 10,
                'total_duration_minutes': len(exercises[:8]) * 4,
                'exercises_per_round': 1
            }
        }

    def _format_as_generic(self, exercises: List[Dict], group_analysis: Dict) -> Dict[str, Any]:
        """Format as generic workout"""
        return {
            'workout_format': 'generic',
            'exercises': exercises,
            'group_analysis': group_analysis,
            'recommended_sets': 3,
            'recommended_reps': '8-12 or 30 seconds',
            'rest_between_exercises': '30-60 seconds'
        }
