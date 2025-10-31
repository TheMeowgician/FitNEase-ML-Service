"""
Retrain Hybrid Model with Real Production Data
==============================================

This script retrains the hybrid collaborative filtering model using actual
production data from the workout_exercise_ratings table.

Based on: FitNEase_Hybrid_Filtering_Final.ipynb
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List
import mysql.connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import from fitnease-ml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class FinalHybridRecommender:
    """Final working hybrid with proper evaluation methodology"""

    def __init__(self):
        # Configuration (from notebook)
        self.CONTENT_WEIGHT = 0.4  # 40%
        self.COLLABORATIVE_WEIGHT = 0.6  # 60%
        self.RELEVANCE_THRESHOLD = 3.5

        # Data
        self.ratings_df = None
        self.exercises_df = None
        self.user_item_matrix = None

        logger.info(f"Initialized FinalHybridRecommender")
        logger.info(f"Content Weight: {self.CONTENT_WEIGHT} (40%)")
        logger.info(f"Collaborative Weight: {self.COLLABORATIVE_WEIGHT} (60%)")
        logger.info(f"Relevance Threshold: {self.RELEVANCE_THRESHOLD}+ stars")

    def load_data_from_database(self, tracking_conn, content_conn):
        """Load data from production databases"""
        logger.info("Loading data from production databases...")

        # Load ratings from tracking database
        logger.info("Fetching workout_exercise_ratings...")
        ratings_query = """
        SELECT
            user_id,
            exercise_id,
            rating_value as rating,
            difficulty_perceived,
            enjoyment_rating,
            would_do_again,
            rated_at as timestamp
        FROM workout_exercise_ratings
        WHERE completed = 1
        """
        self.ratings_df = pd.read_sql(ratings_query, tracking_conn)
        logger.info(f"Loaded {len(self.ratings_df)} ratings")

        # Load exercises from content database
        logger.info("Fetching exercises...")
        exercises_query = """
        SELECT
            exercise_id,
            exercise_name,
            target_muscle_group,
            difficulty_level,
            equipment_needed,
            estimated_duration_seconds as default_duration_seconds,
            estimated_calories_per_min as calories_burned_per_minute,
            description as instructions
        FROM exercises
        """
        self.exercises_df = pd.read_sql(exercises_query, content_conn)

        # Convert difficulty_level from enum strings to numeric
        difficulty_map = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3
        }
        self.exercises_df['difficulty_level'] = self.exercises_df['difficulty_level'].map(difficulty_map).fillna(2)

        logger.info(f"Loaded {len(self.exercises_df)} exercises")

        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='exercise_id',
            values='rating',
            fill_value=np.nan
        )
        logger.info(f"Matrix shape: {self.user_item_matrix.shape}")

        # Calculate sparsity
        sparsity = 1 - (self.user_item_matrix.count().sum() /
                       (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]))
        logger.info(f"Matrix sparsity: {sparsity:.3f} ({sparsity*100:.1f}%)")

    def get_user_context(self, user_id: int) -> Dict:
        """Get comprehensive user context"""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]

        if user_ratings.empty:
            return {
                'avg_rating': 3.0,
                'rating_count': 0,
                'preferred_muscle_groups': ['core'],
                'preferred_difficulty': 2,
                'preferred_equipment': ['bodyweight']
            }

        detailed = user_ratings.merge(self.exercises_df, on='exercise_id', how='left')

        avg_rating = user_ratings['rating'].mean()
        rating_count = len(user_ratings)

        muscle_groups = detailed['target_muscle_group'].value_counts().head(2).index.tolist()
        avg_difficulty = detailed['difficulty_level'].mean()
        equipment = detailed['equipment_needed'].value_counts().head(2).index.tolist()

        return {
            'avg_rating': avg_rating,
            'rating_count': rating_count,
            'preferred_muscle_groups': muscle_groups if muscle_groups else ['core'],
            'preferred_difficulty': avg_difficulty if not pd.isna(avg_difficulty) else 2,
            'preferred_equipment': equipment if equipment else ['bodyweight']
        }

    def calculate_cf_score(self, user_id: int, exercise_id: int, user_context: Dict) -> float:
        """Collaborative filtering score based on user and item patterns"""

        if (user_id not in self.user_item_matrix.index or
            exercise_id not in self.user_item_matrix.columns):
            return 0.4

        if not pd.isna(self.user_item_matrix.loc[user_id, exercise_id]):
            return 0.1

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        exercise_ratings = self.user_item_matrix[exercise_id].dropna()

        if len(user_ratings) == 0 or len(exercise_ratings) == 0:
            return 0.4

        user_avg = user_ratings.mean()
        exercise_avg = exercise_ratings.mean()

        common_users = len(exercise_ratings)
        popularity_score = min(1.0, common_users / 20)

        if user_context['rating_count'] > 5:
            prediction = 0.7 * user_avg + 0.3 * exercise_avg
        else:
            prediction = 0.4 * user_avg + 0.6 * exercise_avg

        prediction += 0.1 * popularity_score

        cf_score = (prediction - 1) / 4
        return min(1.0, max(0.0, cf_score))

    def calculate_content_score(self, user_id: int, exercise_id: int, user_context: Dict) -> float:
        """Content-based score using exercise attributes"""

        exercise_data = self.exercises_df[self.exercises_df['exercise_id'] == exercise_id]
        if exercise_data.empty:
            return 0.1

        exercise = exercise_data.iloc[0]
        score = 0.0

        # Muscle group matching
        if exercise['target_muscle_group'] in user_context['preferred_muscle_groups']:
            score += 0.4
        else:
            muscle_compatibility = {
                ('core', 'upper_body'): 0.15,
                ('upper_body', 'core'): 0.15,
                ('core', 'lower_body'): 0.1,
                ('lower_body', 'core'): 0.1,
                ('upper_body', 'lower_body'): 0.05,
                ('lower_body', 'upper_body'): 0.05
            }

            for pref_muscle in user_context['preferred_muscle_groups']:
                compatibility = muscle_compatibility.get((exercise['target_muscle_group'], pref_muscle), 0)
                score += compatibility

        # Difficulty matching
        difficulty_diff = abs(exercise['difficulty_level'] - user_context['preferred_difficulty'])
        if difficulty_diff == 0:
            score += 0.25
        elif difficulty_diff == 1:
            score += 0.15
        else:
            score += max(0, 0.25 - (difficulty_diff * 0.05))

        # Equipment accessibility
        if exercise['equipment_needed'] in user_context['preferred_equipment']:
            score += 0.2
        elif exercise['equipment_needed'] == 'bodyweight':
            score += 0.15

        # Exercise quality indicators
        calories_score = min(1.0, exercise['calories_burned_per_minute'] / 10)
        duration_score = 1.0 - min(1.0, exercise['default_duration_seconds'] / 120)

        score += 0.1 * calories_score
        score += 0.05 * duration_score

        return min(1.0, max(0.0, score))

    def get_hybrid_recommendations(self, user_id: int, num_recs: int = 10) -> List[Dict]:
        """Get hybrid recommendations"""

        user_context = self.get_user_context(user_id)

        rated_exercises = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['exercise_id'].values)
        all_exercises = self.exercises_df['exercise_id'].tolist()
        candidate_exercises = [eid for eid in all_exercises if eid not in rated_exercises]

        scored_exercises = []

        for exercise_id in candidate_exercises:
            cf_score = self.calculate_cf_score(user_id, exercise_id, user_context)
            content_score = self.calculate_content_score(user_id, exercise_id, user_context)

            hybrid_score = (
                self.CONTENT_WEIGHT * content_score +
                self.COLLABORATIVE_WEIGHT * cf_score
            )

            scored_exercises.append({
                'exercise_id': exercise_id,
                'hybrid_score': hybrid_score,
                'content_score': content_score,
                'cf_score': cf_score
            })

        scored_exercises.sort(key=lambda x: x['hybrid_score'], reverse=True)

        recommendations = []
        for item in scored_exercises[:num_recs]:
            exercise_data = self.exercises_df[self.exercises_df['exercise_id'] == item['exercise_id']].iloc[0]

            rec = {
                'exercise_id': int(item['exercise_id']),
                'exercise_name': str(exercise_data['exercise_name']),
                'target_muscle_group': str(exercise_data['target_muscle_group']),
                'difficulty_level': int(exercise_data['difficulty_level']),
                'equipment_needed': str(exercise_data['equipment_needed']),
                'hybrid_score': float(item['hybrid_score']),
                'content_score': float(item['content_score']),
                'cf_score': float(item['cf_score']),
                'calories_burned_per_minute': float(exercise_data['calories_burned_per_minute'])
            }
            recommendations.append(rec)

        return recommendations


def get_database_connections():
    """Get database connections"""

    # Tracking database (ratings)
    tracking_conn = mysql.connector.connect(
        host=os.getenv('TRACKING_DB_HOST', 'fitnease-tracking-db'),
        port=int(os.getenv('TRACKING_DB_PORT', 3306)),
        database='fitnease_tracking_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU=')
    )

    # Content database (exercises)
    content_conn = mysql.connector.connect(
        host=os.getenv('CONTENT_DB_HOST', 'fitnease-content-db'),
        port=int(os.getenv('CONTENT_DB_PORT', 3306)),
        database='fitnease_content_db',
        user='root',
        password=os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU=')
    )

    return tracking_conn, content_conn


def main():
    """Main retraining function"""
    logger.info("="*70)
    logger.info("RETRAINING HYBRID MODEL WITH PRODUCTION DATA")
    logger.info("="*70)

    try:
        # Connect to databases
        logger.info("Connecting to databases...")
        tracking_conn, content_conn = get_database_connections()
        logger.info("✅ Connected to databases")

        # Initialize recommender
        recommender = FinalHybridRecommender()

        # Load production data
        recommender.load_data_from_database(tracking_conn, content_conn)

        # Verify minimum data requirements
        if len(recommender.ratings_df) < 50:
            logger.warning(f"⚠️  Only {len(recommender.ratings_df)} ratings found")
            logger.warning("Collaborative filtering requires at least 50-100 ratings")
            logger.warning("Model will work but may use content-only fallback")

        # Test the model
        logger.info("\nTesting model with first user...")
        test_user = recommender.ratings_df['user_id'].iloc[0]
        recommendations = recommender.get_hybrid_recommendations(test_user, 5)
        logger.info(f"✅ Generated {len(recommendations)} recommendations")

        # Display sample recommendations
        logger.info("\nSample Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            logger.info(f"  {i}. {rec['exercise_name']}")
            logger.info(f"     Hybrid: {rec['hybrid_score']:.3f} (Content: {rec['content_score']:.3f}, CF: {rec['cf_score']:.3f})")

        # Prepare model bundle
        complete_hybrid_model = {
            'recommender': recommender,
            'model_type': 'hybrid_filtering_PRODUCTION',
            'timestamp': datetime.now().isoformat(),
            'weights': {
                'content': recommender.CONTENT_WEIGHT,
                'collaborative': recommender.COLLABORATIVE_WEIGHT
            },
            'configuration': {
                'relevance_threshold': recommender.RELEVANCE_THRESHOLD
            },
            'system_info': {
                'total_users': recommender.user_item_matrix.shape[0],
                'total_exercises': recommender.user_item_matrix.shape[1],
                'total_ratings': len(recommender.ratings_df)
            },
            'version': '1.0',
            'description': 'FitNEase Hybrid Recommender - Production Trained'
        }

        # Save model
        model_dir = '/app/ml_models'  # Docker path
        if not os.path.exists(model_dir):
            model_dir = './ml_models'  # Local path
            os.makedirs(model_dir, exist_ok=True)

        model_file = os.path.join(model_dir, 'fitnease_hybrid_complete.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(complete_hybrid_model, f)

        logger.info(f"\n✅ Model saved to: {model_file}")

        # Verify saved model
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)

        test_recs = loaded_model['recommender'].get_hybrid_recommendations(test_user, 3)
        logger.info(f"✅ Verified: Loaded model works ({len(test_recs)} recommendations)")

        logger.info("\n" + "="*70)
        logger.info("✅ MODEL RETRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total Ratings: {len(recommender.ratings_df)}")
        logger.info(f"Total Users: {recommender.user_item_matrix.shape[0]}")
        logger.info(f"Total Exercises: {recommender.user_item_matrix.shape[1]}")
        logger.info(f"Model File: {model_file}")
        logger.info("\n🔄 NEXT STEP: Restart ML service to load new model")
        logger.info("   docker restart fitnease-ml")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if 'tracking_conn' in locals():
            tracking_conn.close()
        if 'content_conn' in locals():
            content_conn.close()
        logger.info("\n🔌 Database connections closed")


if __name__ == '__main__':
    main()
