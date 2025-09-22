#!/usr/bin/env python3
"""
Database Migration Script for FitNEase ML Service
===============================================

Creates all necessary database tables and indexes
"""

import os
import sys
import logging
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

def create_tables(connection):
    """Create all database tables"""
    cursor = connection.cursor()

    # Create recommendations table
    recommendations_table = """
    CREATE TABLE IF NOT EXISTS recommendations (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        exercise_id INT NOT NULL,
        recommendation_type ENUM('content_based', 'collaborative', 'hybrid') NOT NULL,
        score DECIMAL(3,2) NOT NULL,
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_user_id (user_id),
        INDEX idx_exercise_id (exercise_id),
        INDEX idx_recommendation_type (recommendation_type),
        INDEX idx_score (score),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    # Create user_behavior_patterns table
    user_behavior_table = """
    CREATE TABLE IF NOT EXISTS user_behavior_patterns (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL UNIQUE,
        workout_frequency DECIMAL(3,2) DEFAULT 0.00,
        preferred_difficulty_level INT DEFAULT 2,
        preferred_duration_minutes INT DEFAULT 30,
        preferred_muscle_groups JSON,
        preferred_equipment JSON,
        engagement_score DECIMAL(3,2) DEFAULT 0.50,
        consistency_score DECIMAL(3,2) DEFAULT 0.50,
        last_workout_date DATE,
        total_workouts_completed INT DEFAULT 0,
        avg_workout_rating DECIMAL(3,2) DEFAULT 0.00,
        behavioral_cluster INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_user_id (user_id),
        INDEX idx_engagement_score (engagement_score),
        INDEX idx_consistency_score (consistency_score),
        INDEX idx_behavioral_cluster (behavioral_cluster)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    # Create content_based_scores table
    content_based_table = """
    CREATE TABLE IF NOT EXISTS content_based_scores (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        exercise_id INT NOT NULL,
        similarity_score DECIMAL(5,4) NOT NULL,
        feature_weights JSON,
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_exercise (user_id, exercise_id),
        INDEX idx_similarity_score (similarity_score),
        INDEX idx_calculated_at (calculated_at),
        UNIQUE KEY unique_user_exercise (user_id, exercise_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    # Create collaborative_scores table
    collaborative_table = """
    CREATE TABLE IF NOT EXISTS collaborative_scores (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        exercise_id INT NOT NULL,
        predicted_rating DECIMAL(3,2) NOT NULL,
        confidence_score DECIMAL(3,2) NOT NULL,
        similar_users JSON,
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_user_exercise (user_id, exercise_id),
        INDEX idx_predicted_rating (predicted_rating),
        INDEX idx_confidence_score (confidence_score),
        INDEX idx_calculated_at (calculated_at),
        UNIQUE KEY unique_user_exercise_collab (user_id, exercise_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    tables = [
        ("recommendations", recommendations_table),
        ("user_behavior_patterns", user_behavior_table),
        ("content_based_scores", content_based_table),
        ("collaborative_scores", collaborative_table)
    ]

    try:
        for table_name, table_sql in tables:
            logger.info(f"Creating table: {table_name}")
            cursor.execute(table_sql)
            logger.info(f"✓ Table {table_name} created successfully")

        logger.info("All tables created successfully!")
        return True

    except Error as e:
        logger.error(f"Error creating tables: {e}")
        return False
    finally:
        cursor.close()

def verify_tables(connection):
    """Verify that all tables were created"""
    cursor = connection.cursor()

    try:
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]

        expected_tables = [
            'recommendations',
            'user_behavior_patterns',
            'content_based_scores',
            'collaborative_scores'
        ]

        logger.info("Verifying tables...")
        for table in expected_tables:
            if table in tables:
                logger.info(f"✓ Table {table} exists")
            else:
                logger.error(f"✗ Table {table} missing")
                return False

        logger.info("All tables verified successfully!")
        return True

    except Error as e:
        logger.error(f"Error verifying tables: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main migration function"""
    logger.info("Starting FitNEase ML Service database migration...")

    # Get database connection
    connection = get_db_connection()
    if not connection:
        logger.error("Failed to connect to database. Exiting.")
        sys.exit(1)

    try:
        # Create tables
        if not create_tables(connection):
            logger.error("Failed to create tables. Exiting.")
            sys.exit(1)

        # Verify tables
        if not verify_tables(connection):
            logger.error("Table verification failed. Exiting.")
            sys.exit(1)

        logger.info("🎉 Database migration completed successfully!")

    finally:
        connection.close()

if __name__ == "__main__":
    main()