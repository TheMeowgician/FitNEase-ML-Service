#!/usr/bin/env python3
"""
Database Seed Script for FitNEase ML Service
===========================================

Populates the database with initial test data for development and testing
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('ML_DB_HOST', 'localhost'),
            port=int(os.getenv('ML_DB_PORT', '3306')),
            database=os.getenv('ML_DB_NAME', 'fitnease_ml_db'),
            user=os.getenv('ML_DB_USER', 'root'),
            password=os.getenv('ML_DB_PASSWORD', 'rootpassword'),
            autocommit=True
        )
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def seed_user_behavior_patterns(connection):
    """Seed user behavior patterns table"""
    cursor = connection.cursor()

    logger.info("Seeding user_behavior_patterns table...")

    # Sample user behavior data
    user_behaviors = [
        {
            'user_id': 1,
            'workout_frequency': 4.5,
            'preferred_difficulty_level': 3,
            'preferred_duration_minutes': 45,
            'preferred_muscle_groups': json.dumps(['core', 'upper_body']),
            'preferred_equipment': json.dumps(['dumbbells', 'barbell']),
            'engagement_score': 0.85,
            'consistency_score': 0.78,
            'last_workout_date': (datetime.now() - timedelta(days=1)).date(),
            'total_workouts_completed': 45,
            'avg_workout_rating': 4.2,
            'behavioral_cluster': 1
        },
        {
            'user_id': 2,
            'workout_frequency': 3.0,
            'preferred_difficulty_level': 2,
            'preferred_duration_minutes': 30,
            'preferred_muscle_groups': json.dumps(['lower_body', 'core']),
            'preferred_equipment': json.dumps(['bodyweight', 'resistance_bands']),
            'engagement_score': 0.65,
            'consistency_score': 0.60,
            'last_workout_date': (datetime.now() - timedelta(days=3)).date(),
            'total_workouts_completed': 28,
            'avg_workout_rating': 3.8,
            'behavioral_cluster': 2
        },
        {
            'user_id': 3,
            'workout_frequency': 6.0,
            'preferred_difficulty_level': 4,
            'preferred_duration_minutes': 60,
            'preferred_muscle_groups': json.dumps(['full_body', 'upper_body']),
            'preferred_equipment': json.dumps(['barbell', 'dumbbells', 'kettlebell']),
            'engagement_score': 0.95,
            'consistency_score': 0.92,
            'last_workout_date': datetime.now().date(),
            'total_workouts_completed': 120,
            'avg_workout_rating': 4.7,
            'behavioral_cluster': 1
        },
        {
            'user_id': 4,
            'workout_frequency': 2.5,
            'preferred_difficulty_level': 1,
            'preferred_duration_minutes': 20,
            'preferred_muscle_groups': json.dumps(['core', 'flexibility']),
            'preferred_equipment': json.dumps(['bodyweight', 'yoga_mat']),
            'engagement_score': 0.45,
            'consistency_score': 0.40,
            'last_workout_date': (datetime.now() - timedelta(days=7)).date(),
            'total_workouts_completed': 12,
            'avg_workout_rating': 3.2,
            'behavioral_cluster': 3
        },
        {
            'user_id': 5,
            'workout_frequency': 4.0,
            'preferred_difficulty_level': 3,
            'preferred_duration_minutes': 40,
            'preferred_muscle_groups': json.dumps(['upper_body', 'core']),
            'preferred_equipment': json.dumps(['dumbbells', 'cable_machine']),
            'engagement_score': 0.72,
            'consistency_score': 0.68,
            'last_workout_date': (datetime.now() - timedelta(days=2)).date(),
            'total_workouts_completed': 35,
            'avg_workout_rating': 4.0,
            'behavioral_cluster': 1
        }
    ]

    try:
        for behavior in user_behaviors:
            insert_query = """
            INSERT INTO user_behavior_patterns
            (user_id, workout_frequency, preferred_difficulty_level, preferred_duration_minutes,
             preferred_muscle_groups, preferred_equipment, engagement_score, consistency_score,
             last_workout_date, total_workouts_completed, avg_workout_rating, behavioral_cluster)
            VALUES (%(user_id)s, %(workout_frequency)s, %(preferred_difficulty_level)s,
                    %(preferred_duration_minutes)s, %(preferred_muscle_groups)s,
                    %(preferred_equipment)s, %(engagement_score)s, %(consistency_score)s,
                    %(last_workout_date)s, %(total_workouts_completed)s,
                    %(avg_workout_rating)s, %(behavioral_cluster)s)
            ON DUPLICATE KEY UPDATE
                workout_frequency = VALUES(workout_frequency),
                preferred_difficulty_level = VALUES(preferred_difficulty_level),
                preferred_duration_minutes = VALUES(preferred_duration_minutes),
                preferred_muscle_groups = VALUES(preferred_muscle_groups),
                preferred_equipment = VALUES(preferred_equipment),
                engagement_score = VALUES(engagement_score),
                consistency_score = VALUES(consistency_score),
                last_workout_date = VALUES(last_workout_date),
                total_workouts_completed = VALUES(total_workouts_completed),
                avg_workout_rating = VALUES(avg_workout_rating),
                behavioral_cluster = VALUES(behavioral_cluster)
            """

            cursor.execute(insert_query, behavior)

        logger.info(f"✓ Seeded {len(user_behaviors)} user behavior patterns")

    except Error as e:
        logger.error(f"Error seeding user behavior patterns: {e}")
        return False
    finally:
        cursor.close()

    return True

def seed_content_based_scores(connection):
    """Seed content-based scores table"""
    cursor = connection.cursor()

    logger.info("Seeding content_based_scores table...")

    # Sample content-based similarity scores
    content_scores = [
        # User 1 scores
        {'user_id': 1, 'exercise_id': 101, 'similarity_score': 0.8945, 'feature_weights': json.dumps({'difficulty': 0.3, 'muscle_group': 0.4, 'equipment': 0.3})},
        {'user_id': 1, 'exercise_id': 102, 'similarity_score': 0.7832, 'feature_weights': json.dumps({'difficulty': 0.3, 'muscle_group': 0.4, 'equipment': 0.3})},
        {'user_id': 1, 'exercise_id': 103, 'similarity_score': 0.9123, 'feature_weights': json.dumps({'difficulty': 0.3, 'muscle_group': 0.4, 'equipment': 0.3})},
        {'user_id': 1, 'exercise_id': 104, 'similarity_score': 0.6754, 'feature_weights': json.dumps({'difficulty': 0.3, 'muscle_group': 0.4, 'equipment': 0.3})},

        # User 2 scores
        {'user_id': 2, 'exercise_id': 105, 'similarity_score': 0.7234, 'feature_weights': json.dumps({'difficulty': 0.4, 'muscle_group': 0.3, 'equipment': 0.3})},
        {'user_id': 2, 'exercise_id': 106, 'similarity_score': 0.8567, 'feature_weights': json.dumps({'difficulty': 0.4, 'muscle_group': 0.3, 'equipment': 0.3})},
        {'user_id': 2, 'exercise_id': 107, 'similarity_score': 0.6891, 'feature_weights': json.dumps({'difficulty': 0.4, 'muscle_group': 0.3, 'equipment': 0.3})},

        # User 3 scores
        {'user_id': 3, 'exercise_id': 108, 'similarity_score': 0.9456, 'feature_weights': json.dumps({'difficulty': 0.2, 'muscle_group': 0.5, 'equipment': 0.3})},
        {'user_id': 3, 'exercise_id': 109, 'similarity_score': 0.8923, 'feature_weights': json.dumps({'difficulty': 0.2, 'muscle_group': 0.5, 'equipment': 0.3})},
        {'user_id': 3, 'exercise_id': 110, 'similarity_score': 0.8745, 'feature_weights': json.dumps({'difficulty': 0.2, 'muscle_group': 0.5, 'equipment': 0.3})},

        # User 4 scores
        {'user_id': 4, 'exercise_id': 111, 'similarity_score': 0.5678, 'feature_weights': json.dumps({'difficulty': 0.5, 'muscle_group': 0.3, 'equipment': 0.2})},
        {'user_id': 4, 'exercise_id': 112, 'similarity_score': 0.6234, 'feature_weights': json.dumps({'difficulty': 0.5, 'muscle_group': 0.3, 'equipment': 0.2})},

        # User 5 scores
        {'user_id': 5, 'exercise_id': 113, 'similarity_score': 0.8012, 'feature_weights': json.dumps({'difficulty': 0.3, 'muscle_group': 0.4, 'equipment': 0.3})},
        {'user_id': 5, 'exercise_id': 114, 'similarity_score': 0.7589, 'feature_weights': json.dumps({'difficulty': 0.3, 'muscle_group': 0.4, 'equipment': 0.3})},
    ]

    try:
        for score in content_scores:
            insert_query = """
            INSERT INTO content_based_scores
            (user_id, exercise_id, similarity_score, feature_weights)
            VALUES (%(user_id)s, %(exercise_id)s, %(similarity_score)s, %(feature_weights)s)
            ON DUPLICATE KEY UPDATE
                similarity_score = VALUES(similarity_score),
                feature_weights = VALUES(feature_weights),
                calculated_at = CURRENT_TIMESTAMP
            """

            cursor.execute(insert_query, score)

        logger.info(f"✓ Seeded {len(content_scores)} content-based scores")

    except Error as e:
        logger.error(f"Error seeding content-based scores: {e}")
        return False
    finally:
        cursor.close()

    return True

def seed_collaborative_scores(connection):
    """Seed collaborative filtering scores table"""
    cursor = connection.cursor()

    logger.info("Seeding collaborative_scores table...")

    # Sample collaborative filtering scores
    collaborative_scores = [
        # User 1 predictions
        {'user_id': 1, 'exercise_id': 201, 'predicted_rating': 4.2, 'confidence_score': 0.85, 'similar_users': json.dumps([3, 5, 8])},
        {'user_id': 1, 'exercise_id': 202, 'predicted_rating': 3.8, 'confidence_score': 0.72, 'similar_users': json.dumps([3, 7, 12])},
        {'user_id': 1, 'exercise_id': 203, 'predicted_rating': 4.5, 'confidence_score': 0.91, 'similar_users': json.dumps([3, 5, 9])},

        # User 2 predictions
        {'user_id': 2, 'exercise_id': 204, 'predicted_rating': 3.5, 'confidence_score': 0.68, 'similar_users': json.dumps([4, 6, 10])},
        {'user_id': 2, 'exercise_id': 205, 'predicted_rating': 4.0, 'confidence_score': 0.75, 'similar_users': json.dumps([4, 11, 15])},
        {'user_id': 2, 'exercise_id': 206, 'predicted_rating': 3.2, 'confidence_score': 0.61, 'similar_users': json.dumps([6, 13, 14])},

        # User 3 predictions
        {'user_id': 3, 'exercise_id': 207, 'predicted_rating': 4.8, 'confidence_score': 0.94, 'similar_users': json.dumps([1, 16, 17])},
        {'user_id': 3, 'exercise_id': 208, 'predicted_rating': 4.6, 'confidence_score': 0.88, 'similar_users': json.dumps([1, 5, 18])},

        # User 4 predictions
        {'user_id': 4, 'exercise_id': 209, 'predicted_rating': 2.8, 'confidence_score': 0.52, 'similar_users': json.dumps([2, 19, 20])},
        {'user_id': 4, 'exercise_id': 210, 'predicted_rating': 3.1, 'confidence_score': 0.58, 'similar_users': json.dumps([2, 6, 21])},

        # User 5 predictions
        {'user_id': 5, 'exercise_id': 211, 'predicted_rating': 4.1, 'confidence_score': 0.79, 'similar_users': json.dumps([1, 3, 22])},
        {'user_id': 5, 'exercise_id': 212, 'predicted_rating': 3.9, 'confidence_score': 0.73, 'similar_users': json.dumps([1, 23, 24])},
    ]

    try:
        for score in collaborative_scores:
            insert_query = """
            INSERT INTO collaborative_scores
            (user_id, exercise_id, predicted_rating, confidence_score, similar_users)
            VALUES (%(user_id)s, %(exercise_id)s, %(predicted_rating)s, %(confidence_score)s, %(similar_users)s)
            ON DUPLICATE KEY UPDATE
                predicted_rating = VALUES(predicted_rating),
                confidence_score = VALUES(confidence_score),
                similar_users = VALUES(similar_users),
                calculated_at = CURRENT_TIMESTAMP
            """

            cursor.execute(insert_query, score)

        logger.info(f"✓ Seeded {len(collaborative_scores)} collaborative scores")

    except Error as e:
        logger.error(f"Error seeding collaborative scores: {e}")
        return False
    finally:
        cursor.close()

    return True

def seed_recommendations(connection):
    """Seed recommendations table"""
    cursor = connection.cursor()

    logger.info("Seeding recommendations table...")

    # Sample recommendations
    recommendations = [
        # Content-based recommendations
        {'user_id': 1, 'exercise_id': 101, 'recommendation_type': 'content_based', 'score': 0.89, 'reason': 'High similarity based on preferred muscle groups and difficulty level'},
        {'user_id': 1, 'exercise_id': 103, 'recommendation_type': 'content_based', 'score': 0.91, 'reason': 'Perfect match for upper body focus with barbell equipment'},
        {'user_id': 2, 'exercise_id': 106, 'recommendation_type': 'content_based', 'score': 0.86, 'reason': 'Matches bodyweight preference and lower body focus'},

        # Collaborative recommendations
        {'user_id': 1, 'exercise_id': 203, 'recommendation_type': 'collaborative', 'score': 0.91, 'reason': 'Users with similar patterns rated this highly'},
        {'user_id': 3, 'exercise_id': 207, 'recommendation_type': 'collaborative', 'score': 0.94, 'reason': 'Recommended by users with similar advanced fitness levels'},
        {'user_id': 5, 'exercise_id': 211, 'recommendation_type': 'collaborative', 'score': 0.79, 'reason': 'Similar users with strength training focus loved this exercise'},

        # Hybrid recommendations
        {'user_id': 1, 'exercise_id': 301, 'recommendation_type': 'hybrid', 'score': 0.88, 'reason': 'Combines content similarity (0.85) with collaborative prediction (4.3/5)'},
        {'user_id': 2, 'exercise_id': 302, 'recommendation_type': 'hybrid', 'score': 0.82, 'reason': 'Balanced recommendation from content features and user behavior patterns'},
        {'user_id': 3, 'exercise_id': 303, 'recommendation_type': 'hybrid', 'score': 0.93, 'reason': 'Strong match in both content similarity and collaborative filtering'},
        {'user_id': 4, 'exercise_id': 304, 'recommendation_type': 'hybrid', 'score': 0.65, 'reason': 'Gentle introduction based on beginner profile and similar user success'},
        {'user_id': 5, 'exercise_id': 305, 'recommendation_type': 'hybrid', 'score': 0.84, 'reason': 'Optimal blend of content preferences and collaborative insights'},
    ]

    try:
        for rec in recommendations:
            insert_query = """
            INSERT INTO recommendations
            (user_id, exercise_id, recommendation_type, score, reason)
            VALUES (%(user_id)s, %(exercise_id)s, %(recommendation_type)s, %(score)s, %(reason)s)
            """

            cursor.execute(insert_query, rec)

        logger.info(f"✓ Seeded {len(recommendations)} recommendations")

    except Error as e:
        logger.error(f"Error seeding recommendations: {e}")
        return False
    finally:
        cursor.close()

    return True

def verify_seed_data(connection):
    """Verify that seed data was inserted correctly"""
    cursor = connection.cursor()

    logger.info("Verifying seed data...")

    tables = [
        'user_behavior_patterns',
        'content_based_scores',
        'collaborative_scores',
        'recommendations'
    ]

    try:
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"✓ Table {table}: {count} records")

        return True

    except Error as e:
        logger.error(f"Error verifying seed data: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main seeding function"""
    logger.info("Starting FitNEase ML Service database seeding...")

    # Get database connection
    connection = get_db_connection()
    if not connection:
        logger.error("Failed to connect to database. Exiting.")
        sys.exit(1)

    try:
        # Seed all tables
        if not seed_user_behavior_patterns(connection):
            logger.error("Failed to seed user behavior patterns. Exiting.")
            sys.exit(1)

        if not seed_content_based_scores(connection):
            logger.error("Failed to seed content-based scores. Exiting.")
            sys.exit(1)

        if not seed_collaborative_scores(connection):
            logger.error("Failed to seed collaborative scores. Exiting.")
            sys.exit(1)

        if not seed_recommendations(connection):
            logger.error("Failed to seed recommendations. Exiting.")
            sys.exit(1)

        # Verify seed data
        if not verify_seed_data(connection):
            logger.error("Seed data verification failed. Exiting.")
            sys.exit(1)

        logger.info("🎉 Database seeding completed successfully!")
        logger.info("You can now test the ML service with realistic data!")

    finally:
        connection.close()

if __name__ == "__main__":
    main()