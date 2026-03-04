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
    'upper_body_focused': {
        'preferred_muscles': ['upper_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': ['flexibility'],
        'preferred_difficulty': 2,
        'preferred_equipment': ['dumbbells', 'barbell'],
        'fitness_level': 'intermediate',
        'age_range': (22, 35),
    },
    'core_flexibility': {
        'preferred_muscles': ['core', 'flexibility'],
        'secondary_muscles': ['lower_body'],
        'disliked_muscles': ['upper_body'],
        'preferred_difficulty': 1,
        'preferred_equipment': ['bodyweight', 'yoga_mat'],
        'fitness_level': 'beginner',
        'age_range': (25, 45),
    },
    'powerlifter': {
        'preferred_muscles': ['lower_body', 'upper_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': ['flexibility'],
        'preferred_difficulty': 3,
        'preferred_equipment': ['barbell', 'dumbbells'],
        'fitness_level': 'advanced',
        'age_range': (20, 38),
    },
    'full_body_cardio': {
        'preferred_muscles': ['full_body'],
        'secondary_muscles': ['core', 'lower_body'],
        'disliked_muscles': [],
        'preferred_difficulty': 2,
        'preferred_equipment': ['bodyweight', 'kettlebell'],
        'fitness_level': 'intermediate',
        'age_range': (18, 40),
    },
    'home_workout': {
        'preferred_muscles': ['core', 'full_body'],
        'secondary_muscles': ['upper_body', 'lower_body'],
        'disliked_muscles': [],
        'preferred_difficulty': 1,
        'preferred_equipment': ['bodyweight', 'resistance_bands'],
        'fitness_level': 'beginner',
        'age_range': (18, 54),
    },
    'advanced_athlete': {
        'preferred_muscles': ['full_body', 'upper_body', 'lower_body'],
        'secondary_muscles': ['core'],
        'disliked_muscles': [],
        'preferred_difficulty': 3,
        'preferred_equipment': ['barbell', 'cable_machine', 'kettlebell'],
        'fitness_level': 'advanced',
        'age_range': (20, 35),
    },
    'lower_body_specialist': {
        'preferred_muscles': ['lower_body'],
        'secondary_muscles': ['core', 'full_body'],
        'disliked_muscles': ['upper_body'],
        'preferred_difficulty': 2,
        'preferred_equipment': ['barbell', 'dumbbells', 'bodyweight'],
        'fitness_level': 'intermediate',
        'age_range': (22, 42),
    },
    'wellness_seeker': {
        'preferred_muscles': ['flexibility', 'core'],
        'secondary_muscles': ['full_body'],
        'disliked_muscles': ['upper_body'],
        'preferred_difficulty': 1,
        'preferred_equipment': ['yoga_mat', 'bodyweight', 'resistance_bands'],
        'fitness_level': 'beginner',
        'age_range': (30, 54),
    },
}

USERS_PER_ARCHETYPE = 13  # 8 × 13 = 104 users
RATINGS_PER_USER = (35, 50)


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
               difficulty_level, equipment_needed
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
    """Generate a rating with strong, learnable preference patterns."""
    base = 3.0
    mg = exercise['target_muscle_group'] or 'core'
    diff = exercise['difficulty_level'] or 2
    equip = exercise['equipment_needed'] or 'bodyweight'

    # Muscle group preference: biggest signal
    if mg in archetype['preferred_muscles']:
        base += 1.2
    elif mg in archetype['secondary_muscles']:
        base += 0.5
    elif mg in archetype.get('disliked_muscles', []):
        base -= 0.8

    # Difficulty match
    diff_gap = abs(diff - archetype['preferred_difficulty'])
    if diff_gap == 0:
        base += 0.5
    else:
        base -= 0.3 * diff_gap

    # Equipment match
    if equip in archetype['preferred_equipment']:
        base += 0.3

    # Gaussian noise for realism
    noise = random.gauss(0, 0.4)
    rating = base + noise
    return round(max(1.0, min(5.0, rating)), 2)


def select_exercises_for_user(archetype, by_muscle, all_exercises, num_ratings):
    """Select exercises biased toward the archetype's preferences."""
    selected = []

    # 55% from preferred muscles
    preferred_count = int(num_ratings * 0.55)
    # 25% from secondary muscles
    secondary_count = int(num_ratings * 0.25)
    # 20% from other/disliked (creates negative signal)
    other_count = num_ratings - preferred_count - secondary_count

    # Gather preferred exercises
    preferred_pool = []
    for mg in archetype['preferred_muscles']:
        preferred_pool.extend(by_muscle.get(mg, []))

    # Gather secondary exercises
    secondary_pool = []
    for mg in archetype['secondary_muscles']:
        secondary_pool.extend(by_muscle.get(mg, []))

    # Gather other exercises (everything else)
    pref_and_sec = set(archetype['preferred_muscles'] + archetype['secondary_muscles'])
    other_pool = []
    for mg, exs in by_muscle.items():
        if mg not in pref_and_sec:
            other_pool.extend(exs)

    # Sample from each pool (with fallback if pool is too small)
    def safe_sample(pool, count):
        if len(pool) == 0:
            return []
        if count >= len(pool):
            return list(pool)
        return random.sample(pool, count)

    selected.extend(safe_sample(preferred_pool, preferred_count))
    selected.extend(safe_sample(secondary_pool, secondary_count))
    selected.extend(safe_sample(other_pool, other_count))

    # If we still need more, fill from full catalog
    selected_ids = {ex['exercise_id'] for ex in selected}
    remaining = num_ratings - len(selected)
    if remaining > 0:
        extras = [ex for ex in all_exercises if ex['exercise_id'] not in selected_ids]
        selected.extend(safe_sample(extras, remaining))

    # Deduplicate by exercise_id (keep first occurrence)
    seen = set()
    unique = []
    for ex in selected:
        if ex['exercise_id'] not in seen:
            seen.add(ex['exercise_id'])
            unique.append(ex)

    return unique[:num_ratings]


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
#  MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("GENERATING HIGH-QUALITY EVALUATION DATA")
    logger.info("=" * 60)

    # Connect to databases
    auth_conn = get_auth_conn()
    tracking_conn = get_tracking_conn()
    content_conn = get_content_conn()
    logger.info("Connected to all databases")

    try:
        # Load exercise catalog
        all_exercises, by_muscle = load_exercises(content_conn)
        content_conn.close()

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
                    arch_def, by_muscle, all_exercises, num_ratings
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
        cursor.execute("""
            SELECT COUNT(*) FROM workout_exercise_ratings
            WHERE user_id IN (SELECT user_id FROM fitnease_auth_db.users WHERE username LIKE 'eval_user_%%')
        """)
        eval_ratings = cursor.fetchone()[0]
        cursor.close()

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
