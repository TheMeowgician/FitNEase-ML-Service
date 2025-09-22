#!/usr/bin/env python3
"""
Test script to verify ML model loading with all fixes
"""

import os
import pickle
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all required classes to ensure they're available for pickle
from ml_models.custom_classes import *
from ml_models.content_based_recommender import ContentBasedRecommender
from ml_models.hybrid_recommender import HybridRecommender
from ml_models.collaborative_recommender import CollaborativeRecommender
from ml_models.random_forest_predictor import RandomForestPredictor

def test_content_based_model():
    """Test content-based model loading and functionality"""
    print("\n=== Testing Content-Based Model ===")

    try:
        with open('models_pkl/fitnease_content_based_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print(f"SUCCESS: Content-based model loaded successfully")
        print(f"   Type: {type(model_data)}")
        print(f"   Keys: {list(model_data.keys())}")

        # Test wrapper
        content_model = ContentBasedRecommender(model_data)
        health = content_model.health_check()
        print(f"   Health: {health}")

        if health['status'] == 'healthy':
            # Test recommendations
            recs = content_model.get_recommendations('Push Up', 3)
            print(f"   Generated {len(recs)} recommendations")
            for i, rec in enumerate(recs[:2]):
                print(f"     {i+1}. {rec.get('exercise_name', 'Unknown')} (score: {rec.get('similarity_score', 0):.3f})")

        return True

    except Exception as e:
        print(f"ERROR: Content-based model failed: {e}")
        return False

def test_hybrid_model():
    """Test hybrid model loading and functionality"""
    print("\n=== Testing Hybrid Model ===")

    try:
        with open('models_pkl/fitnease_hybrid_complete.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print(f"SUCCESS: Hybrid model loaded successfully")
        print(f"   Type: {type(model_data)}")
        print(f"   Keys: {list(model_data.keys())}")

        # Test wrapper
        hybrid_model = HybridRecommender(model_data)
        health = hybrid_model.health_check()
        print(f"   Health: {health}")

        if health['can_recommend']:
            # Test recommendations
            recs = hybrid_model.get_recommendations(user_id=123, num_recommendations=3)
            print(f"   Generated {len(recs)} recommendations")
            for i, rec in enumerate(recs[:2]):
                print(f"     {i+1}. {rec.get('exercise_name', 'Unknown')} (score: {rec.get('hybrid_score', 0):.3f})")

        return True

    except Exception as e:
        print(f"ERROR: Hybrid model failed: {e}")
        return False

def test_random_forest_model():
    """Test random forest model loading and functionality"""
    print("\n=== Testing Random Forest Model ===")

    try:
        with open('models_pkl/fitnease_rf_single.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print(f"SUCCESS: Random Forest model loaded successfully")
        print(f"   Type: {type(model_data)}")
        print(f"   Keys: {list(model_data.keys())}")

        # Test wrapper
        rf_model = RandomForestPredictor(model_data)
        health = rf_model.health_check()
        print(f"   Health: {health}")

        if health['status'] == 'healthy':
            # Test prediction
            user_profile = {
                "age": 25,
                "fitness_level": "intermediate",
                "bmi": 22.5,
                "experience_months": 18,
                "weekly_workout_frequency": 4,
                "days_since_last_workout": 1,
                "fatigue_level": 2
            }
            workout_features = {
                "difficulty_level": 2,
                "estimated_duration_minutes": 45,
                "equipment_needed": "dumbbells",
                "target_muscle_group": "upper_body",
                "exercise_category": "strength"
            }

            prediction = rf_model.predict_difficulty_appropriateness(user_profile, workout_features)
            print(f"   Difficulty prediction: {prediction.get('appropriateness_score', 0):.3f} ({prediction.get('difficulty_rating', 'unknown')})")

        return True

    except Exception as e:
        print(f"ERROR: Random Forest model failed: {e}")
        return False

def main():
    """Main test function"""
    print("Testing ML Model Loading with Fixes")
    print("=" * 50)

    # Check if models directory exists
    models_dir = 'models_pkl'
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory '{models_dir}' not found")
        return

    # List available model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        file_path = os.path.join(models_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   - {file} ({size_mb:.2f} MB)")

    # Test each model
    results = {}
    results['content_based'] = test_content_based_model()
    results['hybrid'] = test_hybrid_model()
    results['random_forest'] = test_random_forest_model()

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    for model_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"   {model_name.replace('_', ' ').title()}: {status}")

    total_passed = sum(results.values())
    print(f"\nOverall: {total_passed}/{len(results)} models working correctly")

    if total_passed == len(results):
        print("SUCCESS: All models are now working properly!")
    else:
        print("WARNING: Some models still need attention")

if __name__ == "__main__":
    main()