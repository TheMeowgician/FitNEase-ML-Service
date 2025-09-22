#!/usr/bin/env python3
"""
FitNEase ML Service Test Suite
=============================

Comprehensive test suite for ML service endpoints and service-to-service communication
"""

import requests
import json
import time
import sys
from typing import Dict, List, Optional

class MLServiceTester:
    """Test suite for ML service"""

    def __init__(self, base_url: str = "http://localhost:8009"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        self.test_results = []
        self.passed = 0
        self.failed = 0

    def log_test(self, test_name: str, passed: bool, message: str = "", response: Optional[Dict] = None):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        if response and not passed:
            print(f"    Response: {json.dumps(response, indent=2)}")

        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message,
            'response': response
        })

        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            data = response.json()

            passed = (
                response.status_code == 200 and
                data.get('status') == 'healthy' and
                'timestamp' in data and
                'service' in data
            )

            self.log_test(
                "Health Check Endpoint",
                passed,
                f"Status: {response.status_code}, Service: {data.get('service', 'unknown')}",
                data if not passed else None
            )

        except Exception as e:
            self.log_test("Health Check Endpoint", False, f"Exception: {str(e)}")

    def test_content_based_recommendations(self):
        """Test content-based recommendations"""
        test_user_ids = [1, 2, 3]

        for user_id in test_user_ids:
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v1/content-based/recommendations/{user_id}",
                    timeout=15
                )
                data = response.json()

                passed = (
                    response.status_code == 200 and
                    'recommendations' in data and
                    isinstance(data['recommendations'], list) and
                    'metadata' in data
                )

                rec_count = len(data.get('recommendations', []))
                self.log_test(
                    f"Content-Based Recommendations (User {user_id})",
                    passed,
                    f"Status: {response.status_code}, Recommendations: {rec_count}",
                    data if not passed else None
                )

            except Exception as e:
                self.log_test(f"Content-Based Recommendations (User {user_id})", False, f"Exception: {str(e)}")

    def test_collaborative_recommendations(self):
        """Test collaborative filtering recommendations"""
        test_user_ids = [1, 3]

        for user_id in test_user_ids:
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v1/collaborative/recommendations/{user_id}",
                    timeout=15
                )
                data = response.json()

                passed = (
                    response.status_code == 200 and
                    'recommendations' in data and
                    isinstance(data['recommendations'], list) and
                    'metadata' in data
                )

                rec_count = len(data.get('recommendations', []))
                self.log_test(
                    f"Collaborative Recommendations (User {user_id})",
                    passed,
                    f"Status: {response.status_code}, Recommendations: {rec_count}",
                    data if not passed else None
                )

            except Exception as e:
                self.log_test(f"Collaborative Recommendations (User {user_id})", False, f"Exception: {str(e)}")

    def test_hybrid_recommendations(self):
        """Test hybrid recommendations"""
        test_user_ids = [1, 2, 5]

        for user_id in test_user_ids:
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v1/hybrid/recommendations/{user_id}",
                    timeout=20
                )
                data = response.json()

                passed = (
                    response.status_code == 200 and
                    'recommendations' in data and
                    isinstance(data['recommendations'], list) and
                    'metadata' in data and
                    'weights_used' in data.get('metadata', {})
                )

                rec_count = len(data.get('recommendations', []))
                weights = data.get('metadata', {}).get('weights_used', {})
                self.log_test(
                    f"Hybrid Recommendations (User {user_id})",
                    passed,
                    f"Status: {response.status_code}, Recommendations: {rec_count}, Weights: {weights}",
                    data if not passed else None
                )

            except Exception as e:
                self.log_test(f"Hybrid Recommendations (User {user_id})", False, f"Exception: {str(e)}")

    def test_random_forest_predictions(self):
        """Test Random Forest predictions"""
        test_data = [
            {
                "user_id": 1,
                "workout_data": {
                    "difficulty_level": 3,
                    "duration_minutes": 45,
                    "exercise_count": 8,
                    "equipment_needed": "dumbbells",
                    "target_muscle_groups": ["upper_body", "core"]
                }
            },
            {
                "user_id": 2,
                "workout_data": {
                    "difficulty_level": 2,
                    "duration_minutes": 30,
                    "exercise_count": 6,
                    "equipment_needed": "bodyweight",
                    "target_muscle_groups": ["lower_body"]
                }
            }
        ]

        for test_case in test_data:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/random-forest/predict",
                    json=test_case,
                    timeout=15
                )
                data = response.json()

                passed = (
                    response.status_code == 200 and
                    'prediction' in data and
                    'confidence' in data and
                    'features_used' in data
                )

                prediction = data.get('prediction', 'unknown')
                confidence = data.get('confidence', 0)
                self.log_test(
                    f"Random Forest Prediction (User {test_case['user_id']})",
                    passed,
                    f"Status: {response.status_code}, Prediction: {prediction}, Confidence: {confidence}",
                    data if not passed else None
                )

            except Exception as e:
                self.log_test(f"Random Forest Prediction (User {test_case['user_id']})", False, f"Exception: {str(e)}")

    def test_model_health_checks(self):
        """Test individual model health endpoints"""
        models = ['content-based', 'collaborative', 'hybrid', 'random-forest']

        for model in models:
            try:
                response = self.session.get(
                    f"{self.base_url}/api/v1/{model}/health",
                    timeout=10
                )
                data = response.json()

                passed = (
                    response.status_code == 200 and
                    'status' in data and
                    'model_type' in data
                )

                status = data.get('status', 'unknown')
                self.log_test(
                    f"Model Health Check ({model})",
                    passed,
                    f"Status: {response.status_code}, Model Status: {status}",
                    data if not passed else None
                )

            except Exception as e:
                self.log_test(f"Model Health Check ({model})", False, f"Exception: {str(e)}")

    def test_model_management(self):
        """Test model management endpoints"""
        try:
            # Test model info endpoint
            response = self.session.get(f"{self.base_url}/api/v1/models/info", timeout=10)
            data = response.json()

            passed = (
                response.status_code == 200 and
                'models' in data and
                isinstance(data['models'], dict)
            )

            model_count = len(data.get('models', {}))
            self.log_test(
                "Model Management - Info",
                passed,
                f"Status: {response.status_code}, Models: {model_count}",
                data if not passed else None
            )

        except Exception as e:
            self.log_test("Model Management - Info", False, f"Exception: {str(e)}")

    def test_service_communication(self):
        """Test service-to-service communication"""
        # Test if ML service can reach other services
        services_to_test = [
            ('Auth Service', 'http://fitnease-auth:80'),
            ('Content Service', 'http://fitnease-content:80'),
            ('Tracking Service', 'http://fitnease-tracking:80')
        ]

        # Test through a recommendation request that requires service communication
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/hybrid/recommendations/1?include_external_data=true",
                timeout=30
            )
            data = response.json()

            # Check if external service data was included
            metadata = data.get('metadata', {})
            external_data_included = metadata.get('external_services_contacted', False)

            passed = response.status_code == 200

            self.log_test(
                "Service-to-Service Communication",
                passed,
                f"Status: {response.status_code}, External services contacted: {external_data_included}",
                data if not passed else None
            )

        except Exception as e:
            self.log_test("Service-to-Service Communication", False, f"Exception: {str(e)}")

    def test_error_handling(self):
        """Test error handling"""
        error_tests = [
            ("Invalid User ID", f"{self.base_url}/api/v1/content-based/recommendations/999999"),
            ("Invalid Endpoint", f"{self.base_url}/api/v1/nonexistent/endpoint"),
            ("Invalid Method", f"{self.base_url}/api/v1/hybrid/recommendations")  # Missing user_id
        ]

        for test_name, url in error_tests:
            try:
                response = self.session.get(url, timeout=10)

                # Expect 4xx or 5xx errors, not crashes
                passed = 400 <= response.status_code < 600

                self.log_test(
                    f"Error Handling - {test_name}",
                    passed,
                    f"Status: {response.status_code} (Expected 4xx/5xx)",
                    None
                )

            except Exception as e:
                self.log_test(f"Error Handling - {test_name}", False, f"Exception: {str(e)}")

    def test_performance(self):
        """Test basic performance"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/v1/hybrid/recommendations/1", timeout=30)
            end_time = time.time()

            response_time = end_time - start_time
            passed = response.status_code == 200 and response_time < 10.0  # Should respond within 10 seconds

            self.log_test(
                "Performance Test",
                passed,
                f"Response time: {response_time:.2f}s (Expected <10s)",
                None
            )

        except Exception as e:
            self.log_test("Performance Test", False, f"Exception: {str(e)}")

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting FitNEase ML Service Test Suite")
        print("=" * 50)

        # Basic functionality tests
        print("\n📋 Basic Functionality Tests")
        print("-" * 30)
        self.test_health_endpoint()
        self.test_model_health_checks()
        self.test_model_management()

        # ML endpoint tests
        print("\n🤖 ML Endpoint Tests")
        print("-" * 30)
        self.test_content_based_recommendations()
        self.test_collaborative_recommendations()
        self.test_hybrid_recommendations()
        self.test_random_forest_predictions()

        # Advanced tests
        print("\n🔗 Service Integration Tests")
        print("-" * 30)
        self.test_service_communication()
        self.test_error_handling()
        self.test_performance()

        # Summary
        print("\n📊 Test Summary")
        print("=" * 50)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")

        if self.failed > 0:
            print(f"\n⚠️  {self.failed} tests failed. Check the logs above for details.")
            return False
        else:
            print(f"\n🎉 All tests passed! ML service is working correctly.")
            return True

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8009"

    print(f"Testing ML service at: {base_url}")

    tester = MLServiceTester(base_url)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()