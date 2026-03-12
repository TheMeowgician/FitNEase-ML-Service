"""
Weekly Plan Controller
======================

Feature #4: Weekly Workout Plans with Preferred Days
Controller for weekly workout plan generation endpoint
"""

import logging
from flask import current_app
from services.weekly_plan_generator import WeeklyPlanGenerator

logger = logging.getLogger(__name__)


class WeeklyPlanController:
    """Controller for weekly workout plan generation"""

    def __init__(self):
        """Initialize controller"""
        self.generator = None

    def generate_weekly_plan(self, data: dict) -> tuple:
        """
        Generate weekly workout plan

        Args:
            data: Request data containing:
                - user_id: User ID
                - workout_days: List of preferred workout days
                - fitness_level: Fitness level
                - target_muscle_groups: Target muscle groups
                - goals: Fitness goals
                - time_constraints: Time available per workout

        Returns:
            Tuple of (response_dict, status_code)
        """
        try:
            # Validate required fields
            if 'user_id' not in data:
                return {'error': 'user_id is required'}, 400

            if 'workout_days' not in data or not data['workout_days']:
                return {'error': 'workout_days is required and must not be empty'}, 400

            user_id = data['user_id']
            workout_days = data['workout_days']
            fitness_level = data.get('fitness_level', 'beginner')
            target_muscle_groups = data.get('target_muscle_groups', [])
            goals = data.get('goals', [])
            time_constraints = data.get('time_constraints', 30)
            # Pillar 1: PHP may provide authoritative session_count and exercises_per_day
            session_count = data.get('session_count')       # int or None
            exercises_per_day = data.get('exercises_per_day')  # int or None
            week_seed = data.get('week_seed')               # int or None — overrides default ISO-week seed
            exclude_exercise_ids = data.get('exclude_exercise_ids', [])  # Prevent repetition across weeks

            logger.info(f"[WEEKLY_PLAN_CTRL] Generating plan for user {user_id}")
            logger.info(f"[WEEKLY_PLAN_CTRL] Days: {workout_days}, Level: {fitness_level}")
            if session_count is not None:
                logger.info(f"[WEEKLY_PLAN_CTRL] PHP-provided session_count={session_count}, exercises_per_day={exercises_per_day}")
            if week_seed is not None:
                logger.info(f"[WEEKLY_PLAN_CTRL] Custom week_seed={week_seed} (force_fresh regeneration)")
            if exclude_exercise_ids:
                logger.info(f"[WEEKLY_PLAN_CTRL] Excluding {len(exclude_exercise_ids)} recently completed exercises")

            # Initialize generator with model manager
            if self.generator is None:
                model_manager = getattr(current_app, 'model_manager', None)
                self.generator = WeeklyPlanGenerator(model_manager)

            # Generate plan
            result = self.generator.generate_weekly_plan(
                user_id=user_id,
                workout_days=workout_days,
                fitness_level=fitness_level,
                target_muscle_groups=target_muscle_groups,
                goals=goals,
                time_constraints=time_constraints,
                session_count=session_count,
                exercises_per_day=exercises_per_day,
                week_seed=week_seed,
                exclude_exercise_ids=exclude_exercise_ids,
            )

            logger.info(f"[WEEKLY_PLAN_CTRL] Plan generated successfully")

            return {
                'success': True,
                'message': 'Weekly plan generated successfully',
                'data': result
            }, 200

        except ValueError as e:
            logger.error(f"[WEEKLY_PLAN_CTRL] Validation error: {e}")
            return {'error': str(e)}, 400

        except Exception as e:
            logger.error(f"[WEEKLY_PLAN_CTRL] Generation failed: {e}")
            return {'error': f'Failed to generate weekly plan: {str(e)}'}, 500
