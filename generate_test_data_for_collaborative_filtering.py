"""
Generate Test Data for Collaborative Filtering
==============================================

This script generates test users and exercise ratings to enable collaborative filtering.

GOAL: Create 10 test users with 8-12 exercise ratings each = 80-120 total ratings
This will activate the hybrid collaborative filtering model.
"""

import os
import mysql.connector
import logging
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_auth_db_connection():
    """Get auth database connection"""
    return mysql.connector.connect(
        host=os.getenv('AUTH_DB_HOST', 'fitnease-auth-db'),
        port=int(os.getenv('AUTH_DB_PORT', 3306)),
        database='fitnease_auth_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU='),
        autocommit=False
    )

def get_tracking_db_connection():
    """Get tracking database connection"""
    return mysql.connector.connect(
        host=os.getenv('TRACKING_DB_HOST', 'fitnease-tracking-db'),
        port=int(os.getenv('TRACKING_DB_PORT', 3306)),
        database='fitnease_tracking_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU='),
        autocommit=False
    )

def create_test_users(auth_conn, num_users=10):
    """Create test users in auth database"""
    cursor = auth_conn.cursor()
    created_user_ids = []

    try:
        logger.info(f"Creating {num_users} test users...")

        for i in range(1, num_users + 1):
            # Generate user data
            username = f"cf_test_user_{i}"
            email = f"cf_test_{i}@fitnease.test"
            first_name = f"Test{i}"
            last_name = "User"
            age = random.randint(20, 50)
            fitness_level = random.choice(['beginner', 'intermediate', 'advanced'])
            gender = random.choice(['male', 'female'])

            # Check if user already exists
            cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
            existing = cursor.fetchone()

            if existing:
                logger.info(f"✓ User {username} already exists (ID: {existing[0]})")
                created_user_ids.append(existing[0])
                continue

            # Insert user
            insert_query = """
            INSERT INTO users
            (username, email, password_hash, first_name, last_name, age, gender,
             fitness_level, onboarding_completed, is_active, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, 1, NOW())
            """

            cursor.execute(insert_query, (
                username, email, '$2y$10$dummyhashedpassword',  # Dummy password hash
                first_name, last_name, age, gender, fitness_level
            ))

            user_id = cursor.lastrowid
            created_user_ids.append(user_id)
            logger.info(f"✓ Created user: {username} (ID: {user_id})")

        auth_conn.commit()
        logger.info(f"✅ {len(created_user_ids)} users ready")
        return created_user_ids

    except Exception as e:
        logger.error(f"❌ Error creating users: {e}")
        auth_conn.rollback()
        raise
    finally:
        cursor.close()

def get_exercise_ids(tracking_conn):
    """Get available exercise IDs from content service (or use common IDs)"""
    # For testing, we'll use exercise IDs 1-50 (assuming they exist in content service)
    # In production, you'd fetch from the content service API
    return list(range(1, 51))

def create_workout_sessions_and_ratings(tracking_conn, user_ids, exercise_ids):
    """Create workout sessions and exercise ratings for each user"""
    cursor = tracking_conn.cursor()

    try:
        logger.info(f"Creating workout sessions and exercise ratings...")

        total_ratings = 0

        for user_id in user_ids:
            # Each user completes 2-3 workout sessions
            num_sessions = random.randint(2, 3)

            for session_num in range(num_sessions):
                # Create workout session
                session_date = datetime.now() - timedelta(days=random.randint(1, 30))

                session_insert = """
                INSERT INTO workout_sessions
                (user_id, session_type, start_time, end_time, actual_duration_minutes,
                 is_completed, completion_percentage, created_at)
                VALUES (%s, %s, %s, %s, %s, 1, 100.0, %s)
                """

                duration = random.randint(20, 60)
                cursor.execute(session_insert, (
                    user_id, 'general', session_date,
                    session_date + timedelta(minutes=duration),
                    duration, session_date
                ))

                session_id = cursor.lastrowid

                # Each session has 4-6 exercises
                num_exercises = random.randint(4, 6)
                session_exercises = random.sample(exercise_ids, num_exercises)

                for exercise_id in session_exercises:
                    # Create exercise rating
                    # Ratings vary by user preference patterns for collaborative filtering
                    base_rating = 3.0 + (user_id % 3)  # Users have different base preferences
                    rating = min(5.0, max(1.0, base_rating + random.uniform(-1.0, 1.0)))

                    rating_insert = """
                    INSERT INTO workout_exercise_ratings
                    (user_id, exercise_id, session_id, rating_value, difficulty_perceived,
                     enjoyment_rating, would_do_again, completed, rated_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s)
                    """

                    difficulty = random.choice(['too_easy', 'appropriate', 'challenging', 'too_hard'])
                    enjoyment = min(5.0, max(1.0, rating + random.uniform(-0.5, 0.5)))

                    cursor.execute(rating_insert, (
                        user_id, exercise_id, session_id, round(rating, 2),
                        difficulty, round(enjoyment, 2), 1,
                        session_date, session_date, session_date
                    ))

                    total_ratings += 1

        tracking_conn.commit()
        logger.info(f"✅ Created {total_ratings} exercise ratings across {len(user_ids)} users")
        return total_ratings

    except Exception as e:
        logger.error(f"❌ Error creating ratings: {e}")
        tracking_conn.rollback()
        raise
    finally:
        cursor.close()

def verify_data(auth_conn, tracking_conn):
    """Verify the generated data"""
    try:
        # Check auth users
        auth_cursor = auth_conn.cursor()
        auth_cursor.execute("SELECT COUNT(*) FROM users WHERE username LIKE 'cf_test_user_%'")
        user_count = auth_cursor.fetchone()[0]
        auth_cursor.close()

        # Check tracking ratings
        tracking_cursor = tracking_conn.cursor()
        tracking_cursor.execute("""
            SELECT
                COUNT(*) as total_ratings,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT exercise_id) as unique_exercises,
                AVG(rating_value) as avg_rating
            FROM workout_exercise_ratings
        """)
        stats = tracking_cursor.fetchone()
        tracking_cursor.close()

        logger.info("\n" + "="*60)
        logger.info("📊 DATA VERIFICATION")
        logger.info("="*60)
        logger.info(f"✓ Test Users Created: {user_count}")
        logger.info(f"✓ Total Exercise Ratings: {stats[0]}")
        logger.info(f"✓ Unique Users with Ratings: {stats[1]}")
        logger.info(f"✓ Unique Exercises Rated: {stats[2]}")
        logger.info(f"✓ Average Rating: {stats[3]:.2f}")
        logger.info("="*60)

        if stats[0] >= 50 and stats[1] >= 5:
            logger.info("✅ SUFFICIENT DATA FOR COLLABORATIVE FILTERING!")
            logger.info("🎯 Hybrid model should now activate collaborative filtering")
        else:
            logger.warning(f"⚠️  Need more data: {50 - stats[0]} more ratings or {5 - stats[1]} more users")

        return stats[0], stats[1]

    except Exception as e:
        logger.error(f"❌ Error verifying data: {e}")
        return 0, 0

def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("🔬 GENERATING TEST DATA FOR COLLABORATIVE FILTERING")
    logger.info("="*60)

    auth_conn = None
    tracking_conn = None

    try:
        # Connect to databases
        logger.info("Connecting to databases...")
        auth_conn = get_auth_db_connection()
        tracking_conn = get_tracking_db_connection()
        logger.info("✅ Connected to both databases\n")

        # Step 1: Create test users
        user_ids = create_test_users(auth_conn, num_users=10)
        logger.info("")

        # Step 2: Get available exercises
        exercise_ids = get_exercise_ids(tracking_conn)
        logger.info(f"📋 Using {len(exercise_ids)} exercise IDs\n")

        # Step 3: Create workout sessions and ratings
        total_ratings = create_workout_sessions_and_ratings(tracking_conn, user_ids, exercise_ids)
        logger.info("")

        # Step 4: Verify data
        rating_count, user_count = verify_data(auth_conn, tracking_conn)

        logger.info("\n" + "="*60)
        logger.info("✅ TEST DATA GENERATION COMPLETE!")
        logger.info("="*60)
        logger.info("\n📝 NEXT STEPS:")
        logger.info("1. 🧪 Test hybrid model: python3 test_hybrid_status.py")
        logger.info("2. 📱 Open mobile app and check dashboard")
        logger.info("3. 👀 Verify badges show '✨ Hybrid' (not 'Content')")
        logger.info("4. 📊 Check console logs for algorithm confirmation")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}")
        raise

    finally:
        if auth_conn:
            auth_conn.close()
        if tracking_conn:
            tracking_conn.close()
        logger.info("\n🔌 Database connections closed\n")

if __name__ == '__main__':
    main()
