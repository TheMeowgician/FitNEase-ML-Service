#!/usr/bin/env python3
"""
Comprehensive API Testing for FitNEase ML Service
=================================================

Tests all endpoints with various scenarios and validates responses
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = 'http://localhost:5000'

def test_basic_health():
    """Test basic health endpoint"""
    print("=== Testing Basic Health Endpoint ===")
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_health():
    """Test detailed model health endpoint"""
    print("\n=== Testing Model Health Endpoint ===")
    try:
        response = requests.get(f'{BASE_URL}/api/v1/model-health', timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            # Validate model health
            models = data.get('models', {})
            healthy_models = [name for name, info in models.items()
                            if info.get('status') == 'healthy' or info.get('can_recommend') or info.get('can_predict')]
            print(f"Healthy models: {healthy_models}")
            return len(healthy_models) > 0
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_random_forest_prediction():
    """Test Random Forest difficulty prediction"""
    print("\n=== Testing Random Forest Prediction ===")

    test_data = {
        "user_profile": {
            "age": 25,
            "fitness_level": "intermediate",
            "bmi": 22.5,
            "experience_months": 18,
            "weekly_workout_frequency": 4,
            "days_since_last_workout": 1,
            "fatigue_level": 2
        },
        "workout_features": {
            "difficulty_level": 2,
            "estimated_duration_minutes": 45,
            "equipment_needed": "dumbbells",
            "target_muscle_group": "upper_body",
            "exercise_category": "strength"
        }
    }

    try:
        response = requests.post(
            f'{BASE_URL}/api/v1/predict-difficulty',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            # Validate prediction structure
            if 'difficulty_prediction' in data:
                pred = data['difficulty_prediction']
                required_fields = ['appropriateness_score', 'difficulty_rating', 'recommendation']
                has_required = all(field in pred for field in required_fields)
                print(f"Has required fields: {has_required}")
                print(f"Appropriateness Score: {pred.get('appropriateness_score')}")
                print(f"Rating: {pred.get('difficulty_rating')}")
                return has_required
            return False
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_content_based_recommendations():
    """Test content-based similarity recommendations"""
    print("\n=== Testing Content-Based Recommendations ===")

    test_data = {
        "exercise_name": "Push Up",
        "num_recommendations": 5,
        "similarity_metric": "cosine"
    }

    try:
        response = requests.post(
            f'{BASE_URL}/api/v1/content-similarity',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            # Validate response structure
            has_recommendations = 'recommendations' in data
            recommendation_count = len(data.get('recommendations', []))
            print(f"Has recommendations: {has_recommendations}")
            print(f"Recommendation count: {recommendation_count}")

            # Check recommendation structure
            if recommendation_count > 0:
                first_rec = data['recommendations'][0]
                required_fields = ['exercise_id', 'exercise_name', 'similarity_score']
                has_required = all(field in first_rec for field in required_fields)
                print(f"Recommendations have required fields: {has_required}")
                return has_required
            return has_recommendations
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_hybrid_recommendations():
    """Test hybrid recommendations"""
    print("\n=== Testing Hybrid Recommendations ===")

    user_id = 123

    try:
        response = requests.get(f'{BASE_URL}/api/v1/recommendations/{user_id}', timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            # Validate response structure
            has_recommendations = 'recommendations' in data
            recommendation_count = len(data.get('recommendations', []))
            algorithm = data.get('algorithm')

            print(f"Has recommendations: {has_recommendations}")
            print(f"Recommendation count: {recommendation_count}")
            print(f"Algorithm: {algorithm}")

            return has_recommendations and algorithm == 'hybrid'
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases and Error Handling ===")

    results = {}

    # Test invalid user ID
    print("Testing invalid user ID...")
    try:
        response = requests.get(f'{BASE_URL}/api/v1/recommendations/invalid', timeout=5)
        results['invalid_user_id'] = response.status_code in [400, 404, 422]
        print(f"Invalid user ID handling: {'PASS' if results['invalid_user_id'] else 'FAIL'}")
    except Exception as e:
        results['invalid_user_id'] = False
        print(f"Invalid user ID test failed: {e}")

    # Test malformed JSON
    print("Testing malformed JSON...")
    try:
        response = requests.post(
            f'{BASE_URL}/api/v1/predict-difficulty',
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        results['malformed_json'] = response.status_code in [400, 422]
        print(f"Malformed JSON handling: {'PASS' if results['malformed_json'] else 'FAIL'}")
    except Exception as e:
        results['malformed_json'] = False
        print(f"Malformed JSON test failed: {e}")

    # Test missing required fields
    print("Testing missing required fields...")
    try:
        response = requests.post(
            f'{BASE_URL}/api/v1/predict-difficulty',
            json={"incomplete": "data"},
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        results['missing_fields'] = response.status_code in [400, 422]
        print(f"Missing fields handling: {'PASS' if results['missing_fields'] else 'FAIL'}")
    except Exception as e:
        results['missing_fields'] = False
        print(f"Missing fields test failed: {e}")

    # Test non-existent endpoint
    print("Testing non-existent endpoint...")
    try:
        response = requests.get(f'{BASE_URL}/api/v1/nonexistent', timeout=5)
        results['nonexistent_endpoint'] = response.status_code == 404
        print(f"Non-existent endpoint handling: {'PASS' if results['nonexistent_endpoint'] else 'FAIL'}")
    except Exception as e:
        results['nonexistent_endpoint'] = False
        print(f"Non-existent endpoint test failed: {e}")

    return results

def test_service_performance():
    """Test service performance and response times"""
    print("\n=== Testing Service Performance ===")

    # Test health endpoint response time
    start_time = time.time()
    try:
        response = requests.get(f'{BASE_URL}/health', timeout=10)
        health_time = time.time() - start_time
        print(f"Health endpoint response time: {health_time:.3f}s")

        # Test prediction endpoint response time
        start_time = time.time()
        test_data = {
            "user_profile": {"age": 25, "fitness_level": "intermediate", "bmi": 22.5},
            "workout_features": {"difficulty_level": 2, "estimated_duration_minutes": 30}
        }
        response = requests.post(f'{BASE_URL}/api/v1/predict-difficulty', json=test_data, timeout=10)
        prediction_time = time.time() - start_time
        print(f"Prediction endpoint response time: {prediction_time:.3f}s")

        # Performance thresholds
        health_ok = health_time < 1.0  # Health should be very fast
        prediction_ok = prediction_time < 5.0  # Predictions should be reasonable

        print(f"Health performance: {'PASS' if health_ok else 'FAIL'} ({health_time:.3f}s)")
        print(f"Prediction performance: {'PASS' if prediction_ok else 'FAIL'} ({prediction_time:.3f}s)")

        return health_ok and prediction_ok

    except Exception as e:
        print(f"Performance testing failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and generate report"""
    print("FitNEase ML Service - Comprehensive API Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing service at: {BASE_URL}")
    print()

    # Wait for service to be ready
    print("Waiting for service to be ready...")
    time.sleep(2)

    # Run all tests
    test_results = {}

    test_results['basic_health'] = test_basic_health()
    test_results['model_health'] = test_model_health()
    test_results['random_forest'] = test_random_forest_prediction()
    test_results['content_based'] = test_content_based_recommendations()
    test_results['hybrid'] = test_hybrid_recommendations()
    test_results['performance'] = test_service_performance()

    # Edge case testing
    edge_cases = test_edge_cases()
    test_results.update(edge_cases)

    # Generate summary report
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(test_results.values())
    total = len(test_results)

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print()

    print("Detailed Results:")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print()
    if passed == total:
        print("SUCCESS: All tests passed! FitNEase ML Service is fully operational.")
    elif passed >= total * 0.8:  # 80% pass rate
        print("WARNING: Most tests passed, but some issues detected.")
    else:
        print("ERROR: Multiple test failures detected. Service needs attention.")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return test_results

if __name__ == "__main__":
    results = run_comprehensive_test()