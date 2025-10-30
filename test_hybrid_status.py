"""
Test Script: Check if Hybrid Model is Actually Using Collaborative Filtering
============================================================================

This script tests your ML service to see if collaborative filtering is working
or if it's falling back to content-only recommendations.

Usage:
    python test_hybrid_status.py
"""

import requests
import json

# Configuration
ML_SERVICE_URL = "http://localhost:5000"  # Change if different
USER_ID = 2047  # Test user

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_recommendations():
    """Check hybrid recommendations endpoint"""
    print_section("TEST 1: Hybrid Recommendations Endpoint")

    url = f"{ML_SERVICE_URL}/recommendations/{USER_ID}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            print(f"✅ Status: {response.status_code}")
            print(f"\n📊 Algorithm Used: {data.get('algorithm', 'Unknown')}")
            print(f"📈 Total Recommendations: {data.get('count', 0)}")

            # Check weights
            weights = data.get('weights', {})
            print(f"\n⚖️  Weights:")
            print(f"   Content: {weights.get('content_weight', 0)}")
            print(f"   Collaborative: {weights.get('collaborative_weight', 0)}")

            # Check if using collaborative filtering
            algorithm = data.get('algorithm', '')

            if algorithm == 'hybrid':
                print("\n✅ COLLABORATIVE FILTERING IS WORKING!")
                print("   Your hybrid model is successfully using both content and collaborative filtering.")
            elif algorithm == 'hybrid_fallback_to_content':
                print("\n⚠️  FALLBACK MODE - Content-Only")
                print("   Reason:", data.get('note', 'Insufficient collaborative data'))
                print("   📌 You need more rating data to activate collaborative filtering!")
            else:
                print(f"\n❓ Unknown algorithm: {algorithm}")

            # Check first recommendation for score breakdown
            if data.get('recommendations') and len(data['recommendations']) > 0:
                first_rec = data['recommendations'][0]
                print(f"\n📋 Sample Recommendation:")
                print(f"   Exercise: {first_rec.get('exercise_name', 'Unknown')}")
                print(f"   Hybrid Score: {first_rec.get('hybrid_score', 0)}")
                print(f"   Content Score: {first_rec.get('content_score', 'N/A')}")
                print(f"   Collaborative Score: {first_rec.get('collaborative_score', 'N/A')}")

                # Analyze scores
                collab_score = first_rec.get('collaborative_score', 0)
                if collab_score and collab_score > 0:
                    print("\n   ✅ Has collaborative scores - CF is working!")
                else:
                    print("\n   ❌ No collaborative scores - CF not active")

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Cannot connect to {ML_SERVICE_URL}")
        print("   Make sure the ML service is running!")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_model_health():
    """Check hybrid model health"""
    print_section("TEST 2: Hybrid Model Health Check")

    url = f"{ML_SERVICE_URL}/hybrid/health"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            print("✅ Model Status: Healthy")

            health = data.get('health', {})
            print(f"\n📊 Health Metrics:")
            print(f"   Status: {health.get('status', 'Unknown')}")
            print(f"   Last Updated: {health.get('last_updated', 'Unknown')}")

            model_info = data.get('model_info', {})
            print(f"\n📈 Model Info:")
            print(f"   Content Weight: {model_info.get('content_weight', 'Unknown')}")
            print(f"   Collaborative Weight: {model_info.get('collaborative_weight', 'Unknown')}")
            print(f"   Content Model Available: {model_info.get('content_model_available', False)}")
            print(f"   Collaborative Model Available: {model_info.get('collaborative_model_available', False)}")

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Error: {e}")

def check_collaborative_endpoint():
    """Test pure collaborative endpoint"""
    print_section("TEST 3: Pure Collaborative Filtering")

    url = f"{ML_SERVICE_URL}/collaborative-recommendations/{USER_ID}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            count = data.get('count', 0)
            print(f"✅ Collaborative Recommendations: {count}")

            if count > 0:
                print("\n✅ Collaborative filtering is generating recommendations!")
                print(f"   This means you have enough rating data.")
            else:
                print("\n⚠️  No collaborative recommendations")
                print("   Collaborative filtering needs more rating data to work.")

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Error: {e}")

def summary_report():
    """Print summary"""
    print_section("SUMMARY")

    print("""
🎯 How to Know if Hybrid is Working:

1. ✅ Algorithm should be "hybrid" (not "hybrid_fallback_to_content")
2. ✅ Collaborative weight should be > 0 (typically 0.3-0.4)
3. ✅ Recommendations should have collaborative_score values
4. ✅ Collaborative endpoint should return results

❌ If NOT working:
   - Algorithm shows "hybrid_fallback_to_content"
   - Collaborative weight is 0
   - No collaborative_score in recommendations
   - Collaborative endpoint returns 0 results

📋 What to do:
   - Add more users (5-10 test accounts)
   - Have each user complete 3-5 workouts
   - Rate all exercises after each workout
   - Aim for 50-100 total ratings
   - Test again!

Once you have enough rating data, collaborative filtering will
automatically activate and you'll see "algorithm": "hybrid" with
real collaborative scores!
    """)

def main():
    print("\n🔍 HYBRID MODEL STATUS TEST")
    print("Testing collaborative filtering status...")

    # Run tests
    check_recommendations()
    check_model_health()
    check_collaborative_endpoint()
    summary_report()

    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
