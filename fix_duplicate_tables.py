"""
Fix Duplicate ML Database Tables
=================================

This script merges data from lowercase table names to PascalCase tables
and removes the duplicate lowercase tables.

ISSUE: The codebase uses PascalCase table names (Recommendations, CollaborativeScores, etc.)
but seed.py was creating lowercase versions, causing data to be split across duplicate tables.

This prevented collaborative filtering from working because ML scores were written to
lowercase tables but the code reads from PascalCase tables.
"""

import mysql.connector
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    config = {
        'host': os.environ.get('ML_DB_HOST', 'localhost'),
        'port': int(os.environ.get('ML_DB_PORT', 3306)),
        'database': os.environ.get('ML_DB_NAME', 'fitnease_ml_db'),
        'user': os.environ.get('ML_DB_USER', 'root'),
        'password': os.environ.get('ML_DB_PASSWORD', 'rootpassword'),
        'autocommit': False  # Use transactions
    }
    return mysql.connector.connect(**config)

def migrate_table_data(conn, lowercase_table, pascalcase_table, columns):
    """Migrate data from lowercase to PascalCase table"""
    cursor = conn.cursor()

    try:
        # Check if lowercase table exists and has data
        cursor.execute(f"SELECT COUNT(*) FROM {lowercase_table}")
        lowercase_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM {pascalcase_table}")
        pascalcase_count = cursor.fetchone()[0]

        logger.info(f"📊 {lowercase_table}: {lowercase_count} rows")
        logger.info(f"📊 {pascalcase_table}: {pascalcase_count} rows")

        if lowercase_count == 0:
            logger.info(f"✅ {lowercase_table} is empty, no migration needed")
            return True

        # Copy data from lowercase to PascalCase
        columns_str = ', '.join(columns)
        logger.info(f"📋 Migrating {lowercase_count} rows from {lowercase_table} to {pascalcase_table}...")

        cursor.execute(f"""
            INSERT INTO {pascalcase_table} ({columns_str})
            SELECT {columns_str} FROM {lowercase_table}
        """)

        migrated_rows = cursor.rowcount
        logger.info(f"✅ Migrated {migrated_rows} rows from {lowercase_table} to {pascalcase_table}")

        return True

    except mysql.connector.Error as e:
        logger.error(f"❌ Error migrating {lowercase_table}: {e}")
        return False
    finally:
        cursor.close()

def drop_lowercase_table(conn, table_name):
    """Drop lowercase table after migration"""
    cursor = conn.cursor()

    try:
        logger.info(f"🗑️  Dropping redundant table: {table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        logger.info(f"✅ Dropped {table_name}")
        return True

    except mysql.connector.Error as e:
        logger.error(f"❌ Error dropping {table_name}: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main migration function"""
    logger.info("="*60)
    logger.info("🔧 FIXING DUPLICATE ML DATABASE TABLES")
    logger.info("="*60)

    conn = None

    try:
        conn = get_db_connection()
        logger.info("✅ Connected to ML database")

        # Table migration mappings
        migrations = [
            {
                'lowercase': 'recommendations',
                'pascalcase': 'Recommendations',
                'columns': [
                    'user_id', 'workout_id', 'recommendation_score',
                    'algorithm_used', 'recommendation_reason', 'recommendation_type',
                    'content_based_score', 'collaborative_score',
                    'is_viewed', 'is_accepted', 'viewed_at', 'accepted_at', 'created_at'
                ]
            },
            {
                'lowercase': 'user_behavior_patterns',
                'pascalcase': 'UserBehaviorPatterns',
                'columns': [
                    'user_id', 'most_active_time_of_day', 'most_active_days',
                    'average_workout_duration_minutes', 'preferred_rest_day_pattern',
                    'completion_rate_percentage', 'favorite_muscle_groups',
                    'least_performed_exercises', 'improvement_trend',
                    'notification_response_rate', 'social_engagement_level',
                    'feature_usage_json', 'personalization_confidence_score',
                    'last_pattern_update'
                ]
            },
            {
                'lowercase': 'content_based_scores',
                'pascalcase': 'ContentBasedScores',
                'columns': [
                    'user_id', 'exercise_id', 'similarity_score',
                    'feature_vector', 'calculated_at'
                ]
            },
            {
                'lowercase': 'collaborative_scores',
                'pascalcase': 'CollaborativeScores',
                'columns': [
                    'user_id', 'similar_user_id', 'similarity_score',
                    'common_workouts_count', 'rating_correlation', 'calculated_at'
                ]
            }
        ]

        # Step 1: Migrate all data
        logger.info("\n📦 STEP 1: Migrating data from lowercase to PascalCase tables...")
        logger.info("-"*60)

        all_successful = True
        for migration in migrations:
            success = migrate_table_data(
                conn,
                migration['lowercase'],
                migration['pascalcase'],
                migration['columns']
            )

            if not success:
                all_successful = False
                logger.error(f"❌ Migration failed for {migration['lowercase']}")
                conn.rollback()
                return

            logger.info("-"*60)

        if all_successful:
            conn.commit()
            logger.info("✅ All data migrated successfully!")

        # Step 2: Drop lowercase tables
        logger.info("\n🗑️  STEP 2: Dropping redundant lowercase tables...")
        logger.info("-"*60)

        for migration in migrations:
            drop_lowercase_table(conn, migration['lowercase'])
            logger.info("-"*60)

        conn.commit()

        # Step 3: Verify final state
        logger.info("\n📊 STEP 3: Verifying final table counts...")
        logger.info("-"*60)

        cursor = conn.cursor()
        for migration in migrations:
            cursor.execute(f"SELECT COUNT(*) FROM {migration['pascalcase']}")
            count = cursor.fetchone()[0]
            logger.info(f"✅ {migration['pascalcase']}: {count} rows")
        cursor.close()

        logger.info("\n"+"="*60)
        logger.info("✅ DATABASE FIX COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("\n📝 NEXT STEPS:")
        logger.info("1. ✅ Duplicate tables removed")
        logger.info("2. ✅ Data consolidated to PascalCase tables")
        logger.info("3. 🔄 Collaborative filtering should now work correctly")
        logger.info("4. 🧪 Run: python3 test_hybrid_status.py")
        logger.info("="*60)

    except mysql.connector.Error as e:
        logger.error(f"❌ Database error: {e}")
        if conn:
            conn.rollback()
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()
            logger.info("\n🔌 Database connection closed")

if __name__ == '__main__':
    main()
