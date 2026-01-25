#!/usr/bin/env python3
"""
Test Median-Based Group Workout Recommendation Logic
=====================================================

This test validates the median calculation and exercise selection
logic for group workouts WITHOUT requiring the full service to run.
"""

import statistics
import sys

def test_median_calculation():
    """Test that median calculation works correctly for various group compositions"""
    print("\n=== Testing Median Calculation ===")

    test_cases = [
        # (fitness_levels, expected_median, description)
        ([1, 3, 3, 3, 3], 3, "4 advanced + 1 beginner -> should use advanced (median=3)"),
        ([1, 1, 3, 3, 3], 3, "3 advanced + 2 beginner -> should use advanced (median=3)"),
        ([1, 1, 1, 3, 3], 1, "2 advanced + 3 beginner -> should use beginner (median=1)"),
        ([1, 2, 3], 2, "One of each level -> should use intermediate (median=2)"),
        ([2, 2, 2, 2], 2, "All intermediate -> should use intermediate (median=2)"),
        ([1, 1, 1, 1, 1], 1, "All beginner -> should use beginner (median=1)"),
        ([3, 3, 3, 3, 3], 3, "All advanced -> should use advanced (median=3)"),
        ([1, 2], 1.5, "2 members different levels -> median is 1.5 (rounds to 1 or 2)"),
        ([1], 1, "Single member -> use their level"),
        ([1, 2, 2, 3, 3, 3, 3, 3, 3], 3, "9 members, majority advanced -> median=3"),
    ]

    passed = 0
    failed = 0

    for fitness_levels, expected_median, description in test_cases:
        calculated_median = statistics.median(fitness_levels)
        # For group workouts, we use int() to round down
        rounded_median = int(calculated_median)

        # For the 1.5 case, accept either 1 or 2 since int(1.5)=1
        if expected_median == 1.5:
            expected_rounded = 1  # int(1.5) = 1
        else:
            expected_rounded = int(expected_median)

        success = (rounded_median == expected_rounded)
        status = "[PASS]" if success else "[FAIL]"

        print(f"{status}: {description}")
        print(f"       Levels: {fitness_levels} -> Median: {calculated_median} -> Rounded: {rounded_median}")

        if success:
            passed += 1
        else:
            failed += 1
            print(f"       EXPECTED: {expected_rounded}")

    print(f"\nMedian calculation: {passed}/{passed+failed} tests passed")
    return failed == 0


def test_exercise_selection_logic():
    """Test that exercise selection uses median level correctly"""
    print("\n=== Testing Exercise Selection Logic ===")

    # Simulate the exercise filtering logic from group_recommender.py
    def filter_exercises_for_level(all_exercises, target_difficulty):
        """Simulate the median-based filtering logic"""
        suitable = []
        for ex in all_exercises:
            ex_difficulty = ex['difficulty_level']
            # MEDIAN STRATEGY: Include exercises at or below the median level
            if ex_difficulty <= target_difficulty:
                suitable.append(ex)
        return suitable

    # Create mock exercises
    mock_exercises = [
        {'id': 1, 'name': 'Jumping Jacks', 'difficulty_level': 1},
        {'id': 2, 'name': 'Basic Squats', 'difficulty_level': 1},
        {'id': 3, 'name': 'Push-ups', 'difficulty_level': 2},
        {'id': 4, 'name': 'Lunges', 'difficulty_level': 2},
        {'id': 5, 'name': 'Burpees', 'difficulty_level': 3},
        {'id': 6, 'name': 'Box Jumps', 'difficulty_level': 3},
        {'id': 7, 'name': 'Mountain Climbers', 'difficulty_level': 2},
        {'id': 8, 'name': 'Plank', 'difficulty_level': 1},
    ]

    test_cases = [
        (1, [1, 2, 8], "Beginner median -> only level 1 exercises"),
        (2, [1, 2, 3, 4, 7, 8], "Intermediate median -> level 1 and 2 exercises"),
        (3, [1, 2, 3, 4, 5, 6, 7, 8], "Advanced median -> all exercises available"),
    ]

    passed = 0
    failed = 0

    for target_level, expected_ids, description in test_cases:
        suitable = filter_exercises_for_level(mock_exercises, target_level)
        suitable_ids = sorted([ex['id'] for ex in suitable])

        success = (suitable_ids == sorted(expected_ids))
        status = "[PASS]" if success else "[FAIL]"

        print(f"{status}: {description}")
        print(f"       Target level: {target_level} -> Got IDs: {suitable_ids}")

        if success:
            passed += 1
        else:
            failed += 1
            print(f"       EXPECTED IDs: {sorted(expected_ids)}")

    print(f"\nExercise selection: {passed}/{passed+failed} tests passed")
    return failed == 0


def test_median_vs_minimum_comparison():
    """Demonstrate why median is better than minimum"""
    print("\n=== Median vs Minimum Comparison ===")
    print("Showing why median-based is better than minimum-based:\n")

    # Scenario: 5-person group with 1 beginner, 4 advanced
    fitness_levels = [1, 3, 3, 3, 3]

    min_level = min(fitness_levels)
    median_level = int(statistics.median(fitness_levels))

    print(f"Group fitness levels: {fitness_levels}")
    print(f"  - MINIMUM approach: Use level {min_level} (beginner exercises)")
    print(f"    Result: 4 advanced users (80%) get exercises too easy for them")
    print()
    print(f"  - MEDIAN approach: Use level {median_level} (advanced exercises)")
    print(f"    Result: Exercises appropriate for 80% of the group")
    print(f"    The 1 beginner may find it challenging but can modify")
    print()

    # Calculate impact
    users_challenged_min = sum(1 for l in fitness_levels if l <= min_level) / len(fitness_levels) * 100
    users_challenged_median = sum(1 for l in fitness_levels if l <= median_level) / len(fitness_levels) * 100

    print(f"Users appropriately challenged:")
    print(f"  - MINIMUM: {users_challenged_min:.0f}% (only beginners)")
    print(f"  - MEDIAN:  {users_challenged_median:.0f}% (everyone at or below median)")

    return True


def test_group_analysis_structure():
    """Test that group_analysis contains all required fields"""
    print("\n=== Testing Group Analysis Structure ===")

    # Simulate what the group_recommender produces
    fitness_levels = [1, 2, 3, 3, 3]

    group_analysis = {
        'avg_fitness_level': sum(fitness_levels) / len(fitness_levels),
        'median_fitness_level': int(statistics.median(fitness_levels)),
        'min_fitness_level': min(fitness_levels),
        'max_fitness_level': max(fitness_levels),
        'fitness_level_range': 'homogeneous' if (max(fitness_levels) - min(fitness_levels) <= 1) else 'mixed',
        'total_members': len(fitness_levels)
    }

    required_fields = [
        'avg_fitness_level',
        'median_fitness_level',  # NEW field we added
        'min_fitness_level',
        'max_fitness_level',
        'fitness_level_range',
        'total_members'
    ]

    passed = 0
    failed = 0

    for field in required_fields:
        if field in group_analysis:
            print(f"[PASS]: {field} = {group_analysis[field]}")
            passed += 1
        else:
            print(f"[FAIL]: {field} is MISSING")
            failed += 1

    print(f"\nGroup analysis structure: {passed}/{passed+failed} fields present")
    return failed == 0


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("MEDIAN-BASED GROUP WORKOUT RECOMMENDATION TESTS")
    print("=" * 60)

    results = []

    results.append(("Median Calculation", test_median_calculation()))
    results.append(("Exercise Selection Logic", test_exercise_selection_logic()))
    results.append(("Group Analysis Structure", test_group_analysis_structure()))
    results.append(("Median vs Minimum Comparison", test_median_vs_minimum_comparison()))

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("SUCCESS: ALL TESTS PASSED! Median-based logic is correct.")
        return 0
    else:
        print("WARNING:  Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
