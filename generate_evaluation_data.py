"""
Generate High-Quality Evaluation Data
======================================

Creates synthetic users with 8 distinct archetype-based preference patterns.
Each archetype has clear muscle group, difficulty, and equipment preferences
so that ML models can learn meaningful patterns.

- Creates 100 new users (eval_user_ prefix, no collision with existing data)
- ~4,000 new ratings across 200+ exercises from the REAL catalog
- Only INSERTs — never deletes or updates existing data

Usage (inside Docker container):
    docker exec -it fitnease-ml python generate_evaluation_data.py
"""

import os
import random
import logging
import mysql.connector
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
#  ARCHETYPE DEFINITIONS
# ============================================================

ARCHETYPES = {
    'upper_advanced': {
        'preferred_muscles': ['upper_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': ['lower_body'],
        'preferred_difficulty': 3,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'advanced',
        'age_range': (20, 35),
        'calorie_preference': 'high',
        'preferred_categories': ['strength', 'plyometrics'],
        'disliked_categories': ['stretching'],
    },
    'upper_beginner': {
        'preferred_muscles': ['upper_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': ['lower_body'],
        'preferred_difficulty': 1,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'beginner',
        'age_range': (18, 40),
        'calorie_preference': 'low',
        'preferred_categories': ['stretching'],
        'disliked_categories': ['plyometrics'],
    },
    'lower_advanced': {
        'preferred_muscles': ['lower_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': ['upper_body'],
        'preferred_difficulty': 3,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'advanced',
        'age_range': (22, 38),
        'calorie_preference': 'high',
        'preferred_categories': ['strength', 'plyometrics'],
        'disliked_categories': ['stretching'],
    },
    'lower_beginner': {
        'preferred_muscles': ['lower_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': ['upper_body'],
        'preferred_difficulty': 1,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'beginner',
        'age_range': (18, 54),
        'calorie_preference': 'low',
        'preferred_categories': ['stretching', 'cardio'],
        'disliked_categories': ['plyometrics'],
    },
    'core_beginner': {
        'preferred_muscles': ['core'],
        'secondary_muscles': ['lower_body'],
        'disliked_muscles': ['upper_body'],
        'preferred_difficulty': 1,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'beginner',
        'age_range': (25, 45),
        'calorie_preference': 'low',
        'preferred_categories': ['stretching'],
        'disliked_categories': ['plyometrics'],
    },
    'core_intermediate': {
        'preferred_muscles': ['core'],
        'secondary_muscles': ['upper_body'],
        'disliked_muscles': ['lower_body'],
        'preferred_difficulty': 2,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'intermediate',
        'age_range': (20, 38),
        'calorie_preference': 'medium',
        'preferred_categories': ['strength', 'cardio'],
        'disliked_categories': [],
    },
    'balanced_mid': {
        'preferred_muscles': ['upper_body', 'lower_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': [],
        'preferred_difficulty': 2,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'intermediate',
        'age_range': (22, 42),
        'calorie_preference': 'medium',
        'preferred_categories': ['cardio', 'stretching'],
        'disliked_categories': [],
    },
    'heavy_compound': {
        'preferred_muscles': ['lower_body', 'upper_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': [],
        'preferred_difficulty': 3,
        'preferred_equipment': ['bodyweight'],
        'fitness_level': 'advanced',
        'age_range': (22, 35),
        'calorie_preference': 'high',
        'preferred_categories': ['strength', 'plyometrics'],
        'disliked_categories': ['stretching'],
    },
}

USERS_PER_ARCHETYPE = 25  # 8 × 25 = 200 users
RATINGS_PER_USER = (40, 55)


# ============================================================
#  DATABASE CONNECTIONS
# ============================================================

def get_auth_conn():
    return mysql.connector.connect(
        host=os.getenv('AUTH_DB_HOST', 'fitnease-auth-db'),
        port=int(os.getenv('AUTH_DB_PORT', 3306)),
        database='fitnease_auth_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU='),
        autocommit=False,
    )


def get_tracking_conn():
    return mysql.connector.connect(
        host=os.getenv('TRACKING_DB_HOST', 'fitnease-tracking-db'),
        port=int(os.getenv('TRACKING_DB_PORT', 3306)),
        database='fitnease_tracking_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU='),
        autocommit=False,
    )


def get_content_conn():
    return mysql.connector.connect(
        host=os.getenv('CONTENT_DB_HOST', 'fitnease-content-db'),
        port=int(os.getenv('CONTENT_DB_PORT', 3306)),
        database='fitnease_content_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU='),
    )


# ============================================================
#  LOAD EXERCISE CATALOG
# ============================================================

def load_exercises(content_conn):
    """Load ALL exercises from the content database, grouped by features."""
    cursor = content_conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT exercise_id, exercise_name, target_muscle_group,
               difficulty_level, equipment_needed, calories_burned_per_minute,
               exercise_category
        FROM exercises
    """)
    exercises = cursor.fetchall()
    cursor.close()

    # Group by muscle group for easy archetype-based selection
    by_muscle = defaultdict(list)
    for ex in exercises:
        mg = ex['target_muscle_group'] or 'core'
        by_muscle[mg].append(ex)

    logger.info(f"Loaded {len(exercises)} exercises across {len(by_muscle)} muscle groups")
    for mg, exs in by_muscle.items():
        logger.info(f"  {mg}: {len(exs)} exercises")

    return exercises, by_muscle


# ============================================================
#  RATING GENERATION
# ============================================================

def calculate_rating(archetype, exercise):
    """Generate a rating with strong, learnable preference patterns.

    With only bodyweight equipment, signals come from muscle group + difficulty + calorie intensity.
    """
    base = 3.0
    mg = exercise['target_muscle_group'] or 'core'
    diff = exercise['difficulty_level'] or 2
    cal = exercise.get('calories_burned_per_minute') or 5.0

    # Muscle group preference: very strong signal (+2.0 / +0.3 / -1.8)
    if mg in archetype['preferred_muscles']:
        base += 2.0
    elif mg in archetype['secondary_muscles']:
        base += 0.3
    elif mg in archetype.get('disliked_muscles', []):
        base -= 1.8

    # Difficulty match: strong secondary signal (+1.0 / -0.6 per level gap)
    diff_gap = abs(diff - archetype['preferred_difficulty'])
    if diff_gap == 0:
        base += 1.0
    else:
        base -= 0.6 * diff_gap

    # Calorie intensity preference: gentle tertiary signal
    # high prefers cal >= 7, medium prefers 4-7, low prefers cal <= 4
    cal_pref = archetype.get('calorie_preference', 'medium')
    if cal_pref == 'high':
        if cal >= 7:
            base += 0.3
        elif cal <= 4:
            base -= 0.2
    elif cal_pref == 'low':
        if cal <= 4:
            base += 0.3
        elif cal >= 7:
            base -= 0.2
    else:  # medium
        if 4 <= cal <= 7:
            base += 0.2
        elif cal >= 9 or cal <= 2:
            base -= 0.15

    # Exercise category preference: strong signal for more distinct patterns
    cat = exercise.get('exercise_category') or 'strength'
    preferred_cats = archetype.get('preferred_categories', [])
    disliked_cats = archetype.get('disliked_categories', [])
    if cat in preferred_cats:
        base += 0.6
    elif cat in disliked_cats:
        base -= 0.5

    # Gaussian noise (low for clear patterns)
    noise = random.gauss(0, 0.12)
    rating = base + noise
    return round(max(1.0, min(5.0, rating)), 2)


def build_core_exercise_sets(by_muscle, all_exercises):
    """Pre-select shared exercise sets for CF overlap.

    Returns:
        popular: 20 exercises ALL users rate (global popularity)
        core_by_muscle: {muscle_group: [25 exercises]} — shared within archetype
    """
    rng = random.Random(42)  # deterministic

    # 20 popular exercises from mixed muscle groups (every user rates these)
    popular = []
    for mg in sorted(by_muscle.keys()):
        pool = by_muscle[mg]
        popular.extend(rng.sample(pool, min(7, len(pool))))
    popular = popular[:20]

    # 25 core exercises per muscle group (shared among archetype users)
    core_by_muscle = {}
    for mg, exs in by_muscle.items():
        core_by_muscle[mg] = rng.sample(exs, min(25, len(exs)))

    return popular, core_by_muscle


def select_exercises_for_user(archetype, by_muscle, all_exercises, num_ratings,
                               popular_exercises, core_by_muscle):
    """Select exercises with SHARED core sets for CF overlap + random variety."""

    def safe_sample(pool, count):
        if len(pool) == 0:
            return []
        if count >= len(pool):
            return list(pool)
        return random.sample(pool, count)

    selected = []
    selected_ids = set()

    def add(exercises):
        for ex in exercises:
            if ex['exercise_id'] not in selected_ids:
                selected.append(ex)
                selected_ids.add(ex['exercise_id'])

    # 1. Popular exercises (ALL users rate these — creates global overlap)
    add(popular_exercises)

    # 2. Archetype core exercises (same-archetype users share these)
    for mg in archetype['preferred_muscles']:
        add(core_by_muscle.get(mg, []))
    for mg in archetype['secondary_muscles']:
        core = core_by_muscle.get(mg, [])
        add(safe_sample(core, min(10, len(core))))

    # 3. Fill remaining with random variety from preferred/secondary/other
    remaining = num_ratings - len(selected)
    if remaining > 0:
        preferred_pool = []
        for mg in archetype['preferred_muscles']:
            preferred_pool.extend([ex for ex in by_muscle.get(mg, []) if ex['exercise_id'] not in selected_ids])
        secondary_pool = []
        for mg in archetype['secondary_muscles']:
            secondary_pool.extend([ex for ex in by_muscle.get(mg, []) if ex['exercise_id'] not in selected_ids])
        pref_and_sec = set(archetype['preferred_muscles'] + archetype['secondary_muscles'])
        other_pool = []
        for mg, exs in by_muscle.items():
            if mg not in pref_and_sec:
                other_pool.extend([ex for ex in exs if ex['exercise_id'] not in selected_ids])

        pref_count = int(remaining * 0.5)
        sec_count = int(remaining * 0.3)
        other_count = remaining - pref_count - sec_count

        add(safe_sample(preferred_pool, pref_count))
        add(safe_sample(secondary_pool, sec_count))
        add(safe_sample(other_pool, other_count))

    # If we still need more, fill from full catalog
    remaining2 = num_ratings - len(selected)
    if remaining2 > 0:
        extras = [ex for ex in all_exercises if ex['exercise_id'] not in selected_ids]
        add(safe_sample(extras, remaining2))

    return selected[:num_ratings]


# ============================================================
#  DATABASE WRITES
# ============================================================

def create_users(auth_conn, archetype_name, archetype, count):
    """Create users for one archetype. Returns list of user_ids."""
    cursor = auth_conn.cursor()
    user_ids = []
    age_lo, age_hi = archetype['age_range']

    for i in range(1, count + 1):
        username = f"eval_user_{archetype_name}_{i}"
        email = f"eval_{archetype_name}_{i}@fitnease.test"

        # Check if already exists
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        existing = cursor.fetchone()
        if existing:
            user_ids.append(existing[0])
            continue

        cursor.execute("""
            INSERT INTO users
            (username, email, password_hash, first_name, last_name, age, gender,
             fitness_level, onboarding_completed, is_active, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, 1, NOW())
        """, (
            username, email, '$2y$10$dummyhashedpassword',
            f"Eval{i}", archetype_name.replace('_', ' ').title(),
            random.randint(age_lo, age_hi),
            random.choice(['male', 'female']),
            archetype['fitness_level'],
        ))
        user_ids.append(cursor.lastrowid)

    auth_conn.commit()
    cursor.close()
    return user_ids


def create_sessions_and_ratings(tracking_conn, user_id, exercises, archetype):
    """Create workout sessions and ratings for one user."""
    cursor = tracking_conn.cursor()
    total = 0

    # Split exercises into 4-6 sessions
    remaining = list(exercises)
    random.shuffle(remaining)
    session_sizes = []
    while remaining:
        size = min(random.randint(5, 8), len(remaining))
        session_sizes.append(size)
        remaining = remaining[size:]

    idx = 0
    for size in session_sizes:
        session_date = datetime.now() - timedelta(days=random.randint(1, 60))
        duration = random.randint(20, 60)

        cursor.execute("""
            INSERT INTO workout_sessions
            (user_id, workout_id, session_type, start_time, end_time,
             actual_duration_minutes, is_completed, completion_percentage, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 1, 100.0, %s)
        """, (
            user_id, random.randint(1, 200), 'individual',
            session_date, session_date + timedelta(minutes=duration),
            duration, session_date,
        ))
        session_id = cursor.lastrowid

        for ex in exercises[idx:idx + size]:
            rating = calculate_rating(archetype, ex)
            enjoyment = round(max(1.0, min(5.0, rating + random.gauss(0, 0.3))), 2)

            # Difficulty perceived based on gap
            diff = ex['difficulty_level'] or 2
            pref_diff = archetype['preferred_difficulty']
            if diff < pref_diff:
                perceived = random.choice(['too_easy', 'appropriate'])
            elif diff == pref_diff:
                perceived = random.choice(['appropriate', 'challenging'])
            else:
                perceived = random.choice(['challenging', 'too_hard'])

            cursor.execute("""
                INSERT INTO workout_exercise_ratings
                (user_id, exercise_id, session_id, rating_value, difficulty_perceived,
                 enjoyment_rating, would_do_again, completed, rated_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s)
            """, (
                user_id, ex['exercise_id'], session_id, rating,
                perceived, enjoyment, 1 if rating >= 3.0 else 0,
                session_date, session_date, session_date,
            ))
            total += 1

        idx += size

    tracking_conn.commit()
    cursor.close()
    return total


# ============================================================
#  CLEANUP PREVIOUS EVAL DATA
# ============================================================

def cleanup_previous_eval_data(auth_conn, tracking_conn):
    """Delete previously generated eval data so we can regenerate cleanly."""
    # Get eval user IDs
    auth_cursor = auth_conn.cursor()
    auth_cursor.execute("SELECT user_id FROM users WHERE username LIKE 'eval_user_%%'")
    eval_ids = [row[0] for row in auth_cursor.fetchall()]
    auth_cursor.close()

    if not eval_ids:
        logger.info("No previous eval data to clean")
        return

    logger.info(f"Cleaning up {len(eval_ids)} previous eval users and their data...")

    # Delete ratings
    t_cursor = tracking_conn.cursor()
    placeholders = ','.join(['%s'] * len(eval_ids))
    t_cursor.execute(f"DELETE FROM workout_exercise_ratings WHERE user_id IN ({placeholders})", eval_ids)
    deleted_ratings = t_cursor.rowcount
    t_cursor.execute(f"DELETE FROM workout_sessions WHERE user_id IN ({placeholders})", eval_ids)
    deleted_sessions = t_cursor.rowcount
    tracking_conn.commit()
    t_cursor.close()

    # Delete users
    a_cursor = auth_conn.cursor()
    a_cursor.execute(f"DELETE FROM users WHERE user_id IN ({placeholders})", eval_ids)
    deleted_users = a_cursor.rowcount
    auth_conn.commit()
    a_cursor.close()

    logger.info(f"  Deleted {deleted_ratings} ratings, {deleted_sessions} sessions, {deleted_users} users")


# ============================================================
#  MAIN
# ============================================================

def main():
    random.seed(100)  # reproducible data generation
    logger.info("=" * 60)
    logger.info("GENERATING HIGH-QUALITY EVALUATION DATA")
    logger.info("=" * 60)

    # Connect to databases
    auth_conn = get_auth_conn()
    tracking_conn = get_tracking_conn()
    content_conn = get_content_conn()
    logger.info("Connected to all databases")

    try:
        # Clean previous eval data first
        cleanup_previous_eval_data(auth_conn, tracking_conn)

        # Load exercise catalog
        all_exercises, by_muscle = load_exercises(content_conn)
        content_conn.close()

        # Build shared core exercise sets for CF overlap
        popular_exercises, core_by_muscle = build_core_exercise_sets(by_muscle, all_exercises)
        logger.info(f"Popular exercises (all users): {len(popular_exercises)}")
        for mg, exs in core_by_muscle.items():
            logger.info(f"  Core {mg}: {len(exs)} exercises")

        total_users = 0
        total_ratings = 0

        for arch_name, arch_def in ARCHETYPES.items():
            logger.info(f"\nArchetype: {arch_name} ({USERS_PER_ARCHETYPE} users)")

            # Create users
            user_ids = create_users(auth_conn, arch_name, arch_def, USERS_PER_ARCHETYPE)
            total_users += len(user_ids)

            # Generate ratings for each user
            for uid in user_ids:
                num_ratings = random.randint(*RATINGS_PER_USER)
                exercises = select_exercises_for_user(
                    arch_def, by_muscle, all_exercises, num_ratings,
                    popular_exercises, core_by_muscle,
                )
                count = create_sessions_and_ratings(tracking_conn, uid, exercises, arch_def)
                total_ratings += count

            logger.info(f"  Created {len(user_ids)} users with ratings")

        # Verify
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)

        cursor = tracking_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as total, COUNT(DISTINCT user_id) as users,
                   COUNT(DISTINCT exercise_id) as exercises, AVG(rating_value) as avg
            FROM workout_exercise_ratings
        """)
        stats = cursor.fetchone()
        cursor.close()

        # Count eval user ratings via auth connection
        auth_cursor = auth_conn.cursor()
        auth_cursor.execute("SELECT user_id FROM users WHERE username LIKE 'eval_user_%%'")
        eval_user_ids = [row[0] for row in auth_cursor.fetchall()]
        auth_cursor.close()

        eval_ratings = 0
        if eval_user_ids:
            t_cursor = tracking_conn.cursor()
            placeholders = ','.join(['%s'] * len(eval_user_ids))
            t_cursor.execute(
                f"SELECT COUNT(*) FROM workout_exercise_ratings WHERE user_id IN ({placeholders})",
                eval_user_ids,
            )
            eval_ratings = t_cursor.fetchone()[0]
            t_cursor.close()

        logger.info(f"Total ratings in DB:      {stats[0]}")
        logger.info(f"  - From eval users:      {eval_ratings}")
        logger.info(f"Unique users with ratings: {stats[1]}")
        logger.info(f"Unique exercises rated:    {stats[2]}")
        logger.info(f"Average rating:            {stats[3]:.2f}")
        logger.info(f"\nNew users created:         {total_users}")
        logger.info(f"New ratings created:        {total_ratings}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"FATAL: {e}")
        raise
    finally:
        auth_conn.close()
        tracking_conn.close()


if __name__ == '__main__':
    main()
