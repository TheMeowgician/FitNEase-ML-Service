"""
Database Models for FitNEase ML Service
=======================================

Models for ML-specific entities and data structures
"""

import mysql.connector
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager"""

    def __init__(self):
        self.config = {
            'host': os.environ.get('ML_DB_HOST', 'localhost'),
            'port': int(os.environ.get('ML_DB_PORT', 3306)),
            'database': os.environ.get('ML_DB_NAME', 'fitnease_ml_db'),
            'user': os.environ.get('ML_DB_USER', 'root'),
            'password': os.environ.get('ML_DB_PASSWORD', 'rootpassword'),
            'autocommit': True
        }

    def get_connection(self):
        """Get database connection"""
        try:
            return mysql.connector.connect(**self.config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

# Database connection instance
db = DatabaseConnection()

def init_db():
    """Initialize database tables"""
    try:
        conn = db.get_connection()
        if not conn:
            logger.warning("Database connection failed, using mock data")
            return

        cursor = conn.cursor()

        # Create tables
        create_tables = [
            """
            CREATE TABLE IF NOT EXISTS Recommendations (
                recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                workout_id INT NOT NULL,
                recommendation_score DECIMAL(5,4) NOT NULL,
                algorithm_used ENUM('content_based', 'collaborative', 'hybrid', 'random_forest'),
                recommendation_reason TEXT,
                recommendation_type ENUM('content_based', 'collaborative', 'hybrid', 'random_forest'),
                content_based_score DECIMAL(5,4) DEFAULT 0.0000,
                collaborative_score DECIMAL(5,4) DEFAULT 0.0000,
                is_viewed BOOLEAN DEFAULT FALSE,
                is_accepted BOOLEAN DEFAULT FALSE,
                viewed_at TIMESTAMP NULL,
                accepted_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_recommendations_user_score (user_id, recommendation_score DESC),
                INDEX idx_recommendations_algorithm (algorithm_used, is_accepted, created_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS UserBehaviorPatterns (
                pattern_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL UNIQUE,
                most_active_time_of_day TIME,
                most_active_days SET('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'),
                average_workout_duration_minutes INT DEFAULT 0,
                preferred_rest_day_pattern VARCHAR(50),
                completion_rate_percentage DECIMAL(5,2) DEFAULT 0.00,
                favorite_muscle_groups SET('core', 'upper_body', 'lower_body'),
                least_performed_exercises TEXT,
                improvement_trend ENUM('declining', 'stable', 'improving') DEFAULT 'stable',
                notification_response_rate DECIMAL(5,2) DEFAULT 0.00,
                social_engagement_level ENUM('low', 'moderate', 'high') DEFAULT 'moderate',
                feature_usage_json JSON,
                personalization_confidence_score DECIMAL(5,4) DEFAULT 0.0000,
                last_pattern_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user_behavior_patterns_confidence (personalization_confidence_score DESC)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ContentBasedScores (
                content_score_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                exercise_id INT NOT NULL,
                similarity_score DECIMAL(5,4) NOT NULL,
                feature_vector JSON,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_user_exercise (user_id, exercise_id),
                INDEX idx_content_scores_user_similarity (user_id, similarity_score DESC)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CollaborativeScores (
                collaborative_score_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                similar_user_id INT NOT NULL,
                similarity_score DECIMAL(5,4) NOT NULL,
                common_workouts_count INT DEFAULT 0,
                rating_correlation DECIMAL(5,4),
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_user_pair (user_id, similar_user_id),
                INDEX idx_collaborative_scores_user (user_id, similarity_score DESC)
            )
            """
        ]

        for table_sql in create_tables:
            cursor.execute(table_sql)

        logger.info("Database tables initialized successfully")
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

class BaseModel:
    """Base model class with common database operations"""

    @classmethod
    def execute_query(cls, query: str, params: tuple = None) -> Any:
        """Execute database query"""
        try:
            conn = db.get_connection()
            if not conn:
                return None

            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())

            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
            else:
                result = cursor.rowcount

            cursor.close()
            conn.close()
            return result

        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return None

class Recommendations(BaseModel):
    """Recommendations model"""

    def __init__(self, **kwargs):
        self.recommendation_id = kwargs.get('recommendation_id')
        self.user_id = kwargs.get('user_id')
        self.workout_id = kwargs.get('workout_id')
        self.recommendation_score = kwargs.get('recommendation_score', 0.0)
        self.algorithm_used = kwargs.get('algorithm_used', 'hybrid')
        self.recommendation_reason = kwargs.get('recommendation_reason')
        self.recommendation_type = kwargs.get('recommendation_type', 'hybrid')
        self.content_based_score = kwargs.get('content_based_score', 0.0)
        self.collaborative_score = kwargs.get('collaborative_score', 0.0)
        self.is_viewed = kwargs.get('is_viewed', False)
        self.is_accepted = kwargs.get('is_accepted', False)
        self.viewed_at = kwargs.get('viewed_at')
        self.accepted_at = kwargs.get('accepted_at')
        self.created_at = kwargs.get('created_at')

    def save(self) -> bool:
        """Save recommendation to database"""
        try:
            query = """
            INSERT INTO Recommendations
            (user_id, workout_id, recommendation_score, algorithm_used,
             recommendation_reason, recommendation_type, content_based_score,
             collaborative_score, is_viewed, is_accepted)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                self.user_id, self.workout_id, self.recommendation_score,
                self.algorithm_used, self.recommendation_reason, self.recommendation_type,
                self.content_based_score, self.collaborative_score,
                self.is_viewed, self.is_accepted
            )

            result = self.execute_query(query, params)
            return result is not None and result > 0

        except Exception as e:
            logger.error(f"Failed to save recommendation: {e}")
            return False

    @classmethod
    def get_by_user_id(cls, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recommendations for user"""
        query = """
        SELECT * FROM Recommendations
        WHERE user_id = %s
        ORDER BY recommendation_score DESC
        LIMIT %s
        """
        return cls.execute_query(query, (user_id, limit)) or []

    @classmethod
    def update_viewed(cls, recommendation_id: int) -> bool:
        """Mark recommendation as viewed"""
        query = """
        UPDATE Recommendations
        SET is_viewed = TRUE, viewed_at = NOW()
        WHERE recommendation_id = %s
        """
        result = cls.execute_query(query, (recommendation_id,))
        return result is not None and result > 0

    @classmethod
    def update_accepted(cls, recommendation_id: int) -> bool:
        """Mark recommendation as accepted"""
        query = """
        UPDATE Recommendations
        SET is_accepted = TRUE, accepted_at = NOW()
        WHERE recommendation_id = %s
        """
        result = cls.execute_query(query, (recommendation_id,))
        return result is not None and result > 0

class UserBehaviorPatterns(BaseModel):
    """User behavior patterns model"""

    def __init__(self, **kwargs):
        self.pattern_id = kwargs.get('pattern_id')
        self.user_id = kwargs.get('user_id')
        self.most_active_time_of_day = kwargs.get('most_active_time_of_day')
        self.most_active_days = kwargs.get('most_active_days')
        self.average_workout_duration_minutes = kwargs.get('average_workout_duration_minutes', 0)
        self.preferred_rest_day_pattern = kwargs.get('preferred_rest_day_pattern')
        self.completion_rate_percentage = kwargs.get('completion_rate_percentage', 0.0)
        self.favorite_muscle_groups = kwargs.get('favorite_muscle_groups')
        self.least_performed_exercises = kwargs.get('least_performed_exercises')
        self.improvement_trend = kwargs.get('improvement_trend', 'stable')
        self.notification_response_rate = kwargs.get('notification_response_rate', 0.0)
        self.social_engagement_level = kwargs.get('social_engagement_level', 'moderate')
        self.feature_usage_json = kwargs.get('feature_usage_json')
        self.personalization_confidence_score = kwargs.get('personalization_confidence_score', 0.0)
        self.last_pattern_update = kwargs.get('last_pattern_update')

    def save(self) -> bool:
        """Save behavior pattern"""
        try:
            query = """
            INSERT INTO UserBehaviorPatterns
            (user_id, most_active_time_of_day, most_active_days,
             average_workout_duration_minutes, preferred_rest_day_pattern,
             completion_rate_percentage, favorite_muscle_groups,
             least_performed_exercises, improvement_trend,
             notification_response_rate, social_engagement_level,
             feature_usage_json, personalization_confidence_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            most_active_time_of_day = VALUES(most_active_time_of_day),
            most_active_days = VALUES(most_active_days),
            average_workout_duration_minutes = VALUES(average_workout_duration_minutes),
            preferred_rest_day_pattern = VALUES(preferred_rest_day_pattern),
            completion_rate_percentage = VALUES(completion_rate_percentage),
            favorite_muscle_groups = VALUES(favorite_muscle_groups),
            least_performed_exercises = VALUES(least_performed_exercises),
            improvement_trend = VALUES(improvement_trend),
            notification_response_rate = VALUES(notification_response_rate),
            social_engagement_level = VALUES(social_engagement_level),
            feature_usage_json = VALUES(feature_usage_json),
            personalization_confidence_score = VALUES(personalization_confidence_score),
            last_pattern_update = NOW()
            """

            feature_json = json.dumps(self.feature_usage_json) if self.feature_usage_json else None

            params = (
                self.user_id, self.most_active_time_of_day, self.most_active_days,
                self.average_workout_duration_minutes, self.preferred_rest_day_pattern,
                self.completion_rate_percentage, self.favorite_muscle_groups,
                self.least_performed_exercises, self.improvement_trend,
                self.notification_response_rate, self.social_engagement_level,
                feature_json, self.personalization_confidence_score
            )

            result = self.execute_query(query, params)
            return result is not None

        except Exception as e:
            logger.error(f"Failed to save behavior pattern: {e}")
            return False

    @classmethod
    def get_by_user_id(cls, user_id: int) -> Optional[Dict]:
        """Get behavior pattern for user"""
        query = "SELECT * FROM UserBehaviorPatterns WHERE user_id = %s"
        results = cls.execute_query(query, (user_id,))
        return results[0] if results else None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'user_id': self.user_id,
            'most_active_time_of_day': str(self.most_active_time_of_day) if self.most_active_time_of_day else None,
            'most_active_days': self.most_active_days,
            'average_workout_duration_minutes': self.average_workout_duration_minutes,
            'preferred_rest_day_pattern': self.preferred_rest_day_pattern,
            'completion_rate_percentage': float(self.completion_rate_percentage),
            'favorite_muscle_groups': self.favorite_muscle_groups,
            'least_performed_exercises': self.least_performed_exercises,
            'improvement_trend': self.improvement_trend,
            'notification_response_rate': float(self.notification_response_rate),
            'social_engagement_level': self.social_engagement_level,
            'feature_usage_json': self.feature_usage_json,
            'personalization_confidence_score': float(self.personalization_confidence_score),
            'last_pattern_update': str(self.last_pattern_update) if self.last_pattern_update else None
        }

class ContentBasedScores(BaseModel):
    """Content-based similarity scores model"""

    def __init__(self, **kwargs):
        self.content_score_id = kwargs.get('content_score_id')
        self.user_id = kwargs.get('user_id')
        self.exercise_id = kwargs.get('exercise_id')
        self.similarity_score = kwargs.get('similarity_score', 0.0)
        self.feature_vector = kwargs.get('feature_vector')
        self.calculated_at = kwargs.get('calculated_at')

    def save(self) -> bool:
        """Save content-based score"""
        try:
            query = """
            INSERT INTO ContentBasedScores
            (user_id, exercise_id, similarity_score, feature_vector)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            similarity_score = VALUES(similarity_score),
            feature_vector = VALUES(feature_vector),
            calculated_at = NOW()
            """

            feature_json = json.dumps(self.feature_vector) if self.feature_vector else None
            params = (self.user_id, self.exercise_id, self.similarity_score, feature_json)

            result = self.execute_query(query, params)
            return result is not None

        except Exception as e:
            logger.error(f"Failed to save content-based score: {e}")
            return False

    @classmethod
    def get_by_user_id(cls, user_id: int, limit: int = 50) -> List[Dict]:
        """Get content scores for user"""
        query = """
        SELECT * FROM ContentBasedScores
        WHERE user_id = %s
        ORDER BY similarity_score DESC
        LIMIT %s
        """
        return cls.execute_query(query, (user_id, limit)) or []

class CollaborativeScores(BaseModel):
    """Collaborative filtering scores model"""

    def __init__(self, **kwargs):
        self.collaborative_score_id = kwargs.get('collaborative_score_id')
        self.user_id = kwargs.get('user_id')
        self.similar_user_id = kwargs.get('similar_user_id')
        self.similarity_score = kwargs.get('similarity_score', 0.0)
        self.common_workouts_count = kwargs.get('common_workouts_count', 0)
        self.rating_correlation = kwargs.get('rating_correlation')
        self.calculated_at = kwargs.get('calculated_at')

    def save(self) -> bool:
        """Save collaborative score"""
        try:
            query = """
            INSERT INTO CollaborativeScores
            (user_id, similar_user_id, similarity_score, common_workouts_count, rating_correlation)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            similarity_score = VALUES(similarity_score),
            common_workouts_count = VALUES(common_workouts_count),
            rating_correlation = VALUES(rating_correlation),
            calculated_at = NOW()
            """

            params = (
                self.user_id, self.similar_user_id, self.similarity_score,
                self.common_workouts_count, self.rating_correlation
            )

            result = self.execute_query(query, params)
            return result is not None

        except Exception as e:
            logger.error(f"Failed to save collaborative score: {e}")
            return False

    @classmethod
    def get_by_user_id(cls, user_id: int, limit: int = 20) -> List[Dict]:
        """Get collaborative scores for user"""
        query = """
        SELECT * FROM CollaborativeScores
        WHERE user_id = %s
        ORDER BY similarity_score DESC
        LIMIT %s
        """
        return cls.execute_query(query, (user_id, limit)) or []