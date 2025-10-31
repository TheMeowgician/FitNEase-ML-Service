"""
Direct Test of Trained Hybrid Model
====================================

Test the trained model directly to debug why it returns 0 recommendations.
"""

import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
with open('/app/models_pkl/fitnease_hybrid_complete.pkl', 'rb') as f:
    model_data = pickle.load(f)

recommender = model_data['recommender']

logger.info(f"Model loaded successfully")
logger.info(f"Ratings shape: {recommender.ratings_df.shape if recommender.ratings_df is not None else 'None'}")
logger.info(f"Exercises shape: {recommender.exercises_df.shape if recommender.exercises_df is not None else 'None'}")
logger.info(f"Matrix shape: {recommender.user_item_matrix.shape if recommender.user_item_matrix is not None else 'None'}")

# Test with a user who has ratings
test_user = 2050
logger.info(f"\nTesting with user {test_user}...")

# Check user context
user_context = recommender.get_user_context(test_user)
logger.info(f"User context: {user_context}")

# Try to get recommendations
try:
    recommendations = recommender.get_hybrid_recommendations(test_user, 10)
    logger.info(f"\n✅ Generated {len(recommendations)} recommendations")

    for i, rec in enumerate(recommendations[:5], 1):
        logger.info(f"{i}. {rec['exercise_name']}")
        logger.info(f"   Hybrid: {rec['hybrid_score']:.3f} (Content: {rec['content_score']:.3f}, CF: {rec['cf_score']:.3f})")

except Exception as e:
    logger.error(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
