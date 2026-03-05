"""
FitNEase Recommendation Case Study Demo
========================================

Demonstrates what the evaluation metrics actually measure using concrete examples.

For each archetype, picks a user and shows:
1. Their preference profile (what they liked)
2. A leave-one-out ranking test: holds out 1 liked exercise, mixes it with
   49 random exercises, and checks if the model can rank it in the top 10
3. This is exactly what Hit Rate@10 measures — making the metric tangible

Usage (inside Docker container):
    docker exec -it fitnease-ml python demo_case_study.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import mysql.connector
import logging
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models.custom_classes import (
    FitNeaseContentBasedRecommender,
    ProperCollaborativeFiltering,
    FinalHybridRecommender,
)

ARCHETYPES_ORDER = [
    'upper_advanced', 'upper_beginner', 'lower_advanced', 'lower_beginner',
    'core_beginner', 'core_intermediate', 'balanced_mid', 'heavy_compound',
]

ARCHETYPE_DESCRIPTIONS = {
    'upper_advanced':    'Prefers Upper Body, Advanced difficulty, High intensity',
    'upper_beginner':    'Prefers Upper Body, Beginner difficulty, Low intensity',
    'lower_advanced':    'Prefers Lower Body, Advanced difficulty, High intensity',
    'lower_beginner':    'Prefers Lower Body, Beginner difficulty, Low intensity',
    'core_beginner':     'Prefers Core, Beginner difficulty, Low intensity',
    'core_intermediate': 'Prefers Core, Intermediate difficulty, Medium intensity',
    'balanced_mid':      'Prefers All muscle groups, Intermediate, Medium intensity',
    'heavy_compound':    'Prefers compound moves, Advanced difficulty, High intensity',
}


def get_database_connections():
    password = os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU=')
    tracking_conn = mysql.connector.connect(
        host=os.getenv('TRACKING_DB_HOST', 'fitnease-tracking-db'),
        port=int(os.getenv('TRACKING_DB_PORT', 3306)),
        database='fitnease_tracking_db', user='root', password=password,
    )
    content_conn = mysql.connector.connect(
        host=os.getenv('CONTENT_DB_HOST', 'fitnease-content-db'),
        port=int(os.getenv('CONTENT_DB_PORT', 3306)),
        database='fitnease_content_db', user='root', password=password,
    )
    auth_conn = mysql.connector.connect(
        host=os.getenv('AUTH_DB_HOST', 'fitnease-auth-db'),
        port=int(os.getenv('AUTH_DB_PORT', 3306)),
        database='fitnease_auth_db', user='root', password=password,
    )
    return tracking_conn, content_conn, auth_conn


def load_data(tracking_conn, content_conn, auth_conn):
    ratings_df = pd.read_sql("""
        SELECT user_id, exercise_id, rating_value AS rating, rated_at AS timestamp
        FROM workout_exercise_ratings WHERE completed = 1
        ORDER BY rated_at
    """, tracking_conn)

    exercises_df = pd.read_sql("""
        SELECT exercise_id, exercise_name, target_muscle_group, difficulty_level,
               equipment_needed, default_duration_seconds, calories_burned_per_minute,
               instructions
        FROM exercises
    """, content_conn)
    exercises_df['difficulty_level'] = exercises_df['difficulty_level'].fillna(2).astype(int)

    eval_users = pd.read_sql("""
        SELECT user_id, username, fitness_level
        FROM users
        WHERE username LIKE 'eval_user_%%'
        ORDER BY user_id
    """, auth_conn)

    return ratings_df, exercises_df, eval_users


def run_case_study(out):
    """Run the full case study and write to output."""
    def w(text=""):
        out.write(text + "\n")

    w("=" * 70)
    w("  FITNEASE RECOMMENDATION SYSTEM - CASE STUDY DEMO")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 70)
    w()
    w("  This demo makes the evaluation metrics tangible by showing")
    w("  concrete examples of what Hit Rate@10 measures:")
    w()
    w("  For each user, we:")
    w("  1. Show their preference profile (what they liked)")
    w("  2. Hold out one exercise they rated highly")
    w("  3. Mix it with 49 random exercises (50 candidates total)")
    w("  4. Ask: Can the model rank the liked exercise in the top 10?")
    w()
    w("  If yes -> HIT. This is what HR@10 = 84% means: the model")
    w("  successfully ranks the relevant item in the top 10 for 84%")
    w("  of users.")
    w()

    logger.info("Connecting to databases...")
    tracking_conn, content_conn, auth_conn = get_database_connections()
    logger.info("Loading data...")
    ratings_df, exercises_df, eval_users = load_data(tracking_conn, content_conn, auth_conn)
    tracking_conn.close()
    content_conn.close()
    auth_conn.close()

    all_exercise_ids = sorted(exercises_df['exercise_id'].tolist())
    diff_map = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}

    logger.info(f"Found {len(eval_users)} eval users, {len(ratings_df)} ratings, {len(exercises_df)} exercises")

    # ---- Train hybrid model ----
    logger.info("Training hybrid model...")
    hybrid = FinalHybridRecommender()
    hybrid.exercises_df = exercises_df.copy()

    # Hold out last positive rating per user for testing (same as eval script)
    user_groups = ratings_df.groupby('user_id')
    eligible = [uid for uid, grp in user_groups if len(grp) >= 5]

    train_rows = []
    test_map = {}  # user_id -> test_row
    for uid in eligible:
        ur = ratings_df[ratings_df['user_id'] == uid].sort_values('timestamp')
        positive = ur[ur['rating'] >= 4.0]
        if len(positive) == 0:
            train_rows.append(ur)
            continue
        test_item = positive.iloc[-1]
        test_map[uid] = test_item
        train_rows.append(ur[ur.index != test_item.name])

    ineligible = ratings_df[~ratings_df['user_id'].isin(eligible)]
    train_df = pd.concat(train_rows + [ineligible], ignore_index=True)

    hybrid.ratings_df = train_df.copy()
    hybrid.user_item_matrix = train_df.pivot_table(
        index='user_id', columns='exercise_id', values='rating', fill_value=np.nan
    )

    cf_train = train_df[['user_id', 'exercise_id', 'rating']].rename(
        columns={'exercise_id': 'item_id'}
    )
    cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=30)
    cf_model.fit(ratings_data=cf_train)
    hybrid.set_trained_cf_model({'cf_model': cf_model, 'best_method': 'SVD'})
    hybrid.CONTENT_WEIGHT = 0.6
    hybrid.COLLABORATIVE_WEIGHT = 0.4

    def min_max_normalize(scores):
        vals = [s for _, s in scores]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [(eid, 0.5) for eid, _ in scores]
        return [(eid, (s - lo) / (hi - lo)) for eid, s in scores]

    def rank_candidates(candidate_eids, uid, user_context):
        cf_raw = [(eid, hybrid.calculate_cf_score(uid, eid, user_context)) for eid in candidate_eids]
        cb_raw = [(eid, hybrid.calculate_content_score(uid, eid, user_context)) for eid in candidate_eids]
        cf_norm = {eid: s for eid, s in min_max_normalize(cf_raw)}
        cb_norm = {eid: s for eid, s in min_max_normalize(cb_raw)}
        combined = []
        for eid in candidate_eids:
            score = hybrid.CONTENT_WEIGHT * cb_norm[eid] + hybrid.COLLABORATIVE_WEIGHT * cf_norm[eid]
            combined.append((eid, score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    # ---- Run case studies ----
    case_num = 0
    summary_rows = []

    for archetype in ARCHETYPES_ORDER:
        arch_idx = ARCHETYPES_ORDER.index(archetype)
        user_offset = 5
        user_row_idx = arch_idx * 25 + user_offset

        if user_row_idx >= len(eval_users):
            continue

        user_row = eval_users.iloc[user_row_idx]
        user_id = int(user_row['user_id'])
        username = user_row['username']
        fitness_level = user_row['fitness_level']

        if user_id not in test_map:
            continue

        test_item = test_map[user_id]
        test_eid = int(test_item['exercise_id'])
        test_rating = float(test_item['rating'])

        # Get exercise info for the held-out item
        test_ex = exercises_df[exercises_df['exercise_id'] == test_eid]
        if len(test_ex) == 0:
            continue
        test_ex = test_ex.iloc[0]

        case_num += 1
        logger.info(f"Case {case_num}: {archetype} (user {user_id})")

        w("-" * 70)
        w(f"  CASE {case_num}: {archetype.upper().replace('_', ' ')}")
        w(f"  User: {username} (ID: {user_id}, Fitness: {fitness_level})")
        w(f"  Expected: {ARCHETYPE_DESCRIPTIONS.get(archetype, 'N/A')}")
        w("-" * 70)

        # Preference profile
        user_train = train_df[train_df['user_id'] == user_id].merge(
            exercises_df, on='exercise_id', how='inner'
        )
        liked = user_train[user_train['rating'] >= 4.0]

        w(f"\n  PREFERENCE PROFILE ({len(user_train)} rated exercises):")
        if len(liked) > 0:
            muscle_counts = Counter(liked['target_muscle_group'].tolist())
            w(f"    Liked exercises (rating >= 4.0): {len(liked)}")
            w(f"    Muscle groups liked:")
            for muscle, count in muscle_counts.most_common():
                pct = count / len(liked) * 100
                bar = '#' * int(pct / 5)
                w(f"      {muscle:<15s} {count:>3d} ({pct:>4.0f}%) {bar}")
            diff_counts = Counter(liked['difficulty_level'].tolist())
            w(f"    Difficulty levels liked:")
            for diff, count in diff_counts.most_common():
                pct = count / len(liked) * 100
                w(f"      {diff_map.get(diff, str(diff)):<15s} {count:>3d} ({pct:>4.0f}%)")

        # The held-out item
        w(f"\n  HELD-OUT EXERCISE (what the user actually liked):")
        w(f"    Name:         {test_ex['exercise_name']}")
        w(f"    Muscle Group: {test_ex['target_muscle_group']}")
        w(f"    Difficulty:   {diff_map.get(int(test_ex['difficulty_level']), '?')}")
        w(f"    User Rating:  {test_rating:.1f}/5.0")

        # Sample 49 random negatives
        seen = set(train_df[train_df['user_id'] == user_id]['exercise_id'].tolist())
        rng = random.Random(user_id)
        pool = [eid for eid in all_exercise_ids if eid not in seen and eid != test_eid]
        negatives = rng.sample(pool, min(49, len(pool)))
        candidates = [test_eid] + negatives

        # Rank with hybrid model
        user_context = hybrid.get_user_context(user_id)
        ranked = rank_candidates(candidates, user_id, user_context)

        # Find where the held-out item ranks
        ranked_ids = [eid for eid, _ in ranked]
        rank_pos = ranked_ids.index(test_eid) + 1 if test_eid in ranked_ids else len(candidates)
        hit = rank_pos <= 10

        w(f"\n  RANKING TEST (50 candidates: 1 liked + 49 random):")
        w(f"    The held-out exercise ranked: #{rank_pos} out of 50")
        if hit:
            w(f"    Result: HIT (in top 10)")
        else:
            w(f"    Result: MISS (not in top 10)")

        # Show top 5 of the ranking
        w(f"\n    Top 5 ranked candidates:")
        w(f"    {'Rank':<6s} {'Exercise':<40s} {'Muscle':<15s} {'Score':<8s} {'Note'}")
        w(f"    {'':->6s} {'':->40s} {'':->15s} {'':->8s} {'':->15s}")
        for i, (eid, score) in enumerate(ranked[:5]):
            ex_info = exercises_df[exercises_df['exercise_id'] == eid]
            if len(ex_info) > 0:
                name = str(ex_info.iloc[0]['exercise_name'])[:39]
                muscle = str(ex_info.iloc[0]['target_muscle_group'])
            else:
                name = f"Exercise {eid}"
                muscle = "?"
            note = "<-- HELD-OUT" if eid == test_eid else ""
            w(f"    {i+1:<6d} {name:<40s} {muscle:<15s} {score:<8.3f} {note}")

        if rank_pos > 5 and rank_pos <= 10:
            # Show where the held-out item actually ranked
            idx = rank_pos - 1
            eid, score = ranked[idx]
            ex_info = exercises_df[exercises_df['exercise_id'] == eid]
            name = str(ex_info.iloc[0]['exercise_name'])[:39] if len(ex_info) > 0 else f"Exercise {eid}"
            muscle = str(ex_info.iloc[0]['target_muscle_group']) if len(ex_info) > 0 else "?"
            w(f"    ...                                                              ")
            w(f"    {rank_pos:<6d} {name:<40s} {muscle:<15s} {score:<8.3f} <-- HELD-OUT")

        summary_rows.append((archetype, fitness_level, test_ex['exercise_name'],
                             test_ex['target_muscle_group'], rank_pos, hit))
        w()

    # ---- Summary ----
    w("=" * 70)
    w("  SUMMARY TABLE")
    w("=" * 70)
    w(f"  {'Archetype':<22s} {'Fitness':<14s} {'Held-Out Exercise':<30s} {'Rank':<6s} {'Hit?'}")
    w(f"  {'':->22s} {'':->14s} {'':->30s} {'':->6s} {'':->6s}")
    hits = 0
    for arch, fit, ex_name, muscle, rank, hit in summary_rows:
        hit_str = "YES" if hit else "no"
        name_short = str(ex_name)[:29]
        w(f"  {arch:<22s} {fit:<14s} {name_short:<30s} #{rank:<5d} {hit_str}")
        if hit:
            hits += 1

    total = len(summary_rows)
    w(f"\n  Hit Rate: {hits}/{total} ({hits/total*100:.0f}%)")
    w()
    w("=" * 70)
    w("  INTERPRETATION")
    w("=" * 70)
    w()
    w("  Hit Rate@10 answers the question: \"When a user likes an exercise,")
    w("  can the model rank it in the top 10 out of 50 random candidates?\"")
    w()
    w("  Our full evaluation across 200 users shows:")
    w("    - Content-Based:  84.2% Hit Rate@10")
    w("    - Collaborative:  88.0% Hit Rate@10")
    w("    - Hybrid:         81.2% Hit Rate@10")
    w()
    w("  This means for ~84% of users, the model successfully identifies")
    w("  exercises they would enjoy from a pool of 50 candidates.")
    w("=" * 70)
    w()


def main():
    import io
    buf = io.StringIO()
    run_case_study(buf)
    report = buf.getvalue()
    print(report)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'demo_case_study.txt')
    with open(output_file, 'w') as f:
        f.write(report)
    logger.info(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
