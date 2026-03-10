"""
FitNEase ML Model Learning Curves Generator
============================================

Generates learning curve data by training models at different training set sizes
(20%, 40%, 60%, 80%, 100%) and evaluating on the same held-out test set.

Output: Tables with 95% confidence intervals suitable for thesis defense presentation.

Models evaluated:
  - Random Forest (Suitability Classifier): Accuracy, F1
  - Collaborative Filtering (SVD): HR@10, NDCG@10, MRR (sampled 50 candidates)
  - Hybrid (CB+CF): HR@10, NDCG@10, MRR (sampled 50 candidates)

Note: Content-Based is excluded from learning curves because it uses exercise features
(not user ratings), so it does not improve with more training data.

Usage (inside Docker container):
    docker exec -it fitnease-ml python generate_learning_curves.py
"""

import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import mysql.connector
from collections import defaultdict
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models.custom_classes_v2 import (
    FitNeaseContentBasedRecommender,
    ProperCollaborativeFiltering,
    FinalHybridRecommender,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_NEGATIVES = 49  # 49 negatives + 1 positive = 50 candidates
TRAINING_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
RF_RANDOM_SEEDS = [42, 123, 456, 789, 1024]  # Multiple seeds for RF confidence intervals


# ============================================================
#  DATABASE CONNECTIONS
# ============================================================

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


# ============================================================
#  DATA LOADING
# ============================================================

def load_data(tracking_conn, content_conn, auth_conn):
    ratings_df = pd.read_sql("""
        SELECT user_id, exercise_id, rating_value AS rating,
               difficulty_perceived, enjoyment_rating, would_do_again,
               completed, rated_at AS timestamp
        FROM workout_exercise_ratings WHERE completed = 1
        ORDER BY rated_at
    """, tracking_conn)
    logger.info(f"Loaded {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users")

    exercises_df = pd.read_sql("""
        SELECT exercise_id, exercise_name, target_muscle_group, difficulty_level,
               equipment_needed, default_duration_seconds, calories_burned_per_minute,
               instructions
        FROM exercises
    """, content_conn)
    exercises_df['difficulty_level'] = exercises_df['difficulty_level'].fillna(2).astype(int)
    logger.info(f"Loaded {len(exercises_df)} exercises")

    users_df = pd.read_sql("""
        SELECT user_id, age, fitness_level, workout_experience_years
        FROM users WHERE age IS NOT NULL
    """, auth_conn)
    users_df['bmi'] = 23.0
    logger.info(f"Loaded {len(users_df)} user profiles")

    return ratings_df, exercises_df, users_df


# ============================================================
#  METRIC HELPERS
# ============================================================

def ndcg_at_k(ranked_list, relevant_items, k):
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def sampled_candidates(test_exercise_id, all_exercise_ids, seen_ids, num_neg, seed):
    rng = random.Random(seed)
    pool = [eid for eid in all_exercise_ids if eid not in seen_ids and eid != test_exercise_id]
    negatives = rng.sample(pool, min(num_neg, len(pool)))
    return [test_exercise_id] + negatives


def confidence_interval_95(values):
    """Compute 95% confidence interval (1.96 * standard error of the mean)."""
    if len(values) < 2:
        return 0.0
    return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


# ============================================================
#  RF ENCODING HELPERS
# ============================================================

def encode_fitness_level(level):
    mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
    return mapping.get(str(level).lower(), 2) if isinstance(level, str) else 2


def encode_bmi_category(bmi):
    if bmi < 18.5: return 1
    elif bmi < 25: return 2
    elif bmi < 30: return 3
    return 4


def derive_suitability_label(row):
    rating = row['rating']
    gap = row['difficulty_gap']
    if rating < 3: return 0
    if gap < -0.5: return 1
    if gap > 0.5: return 3
    return 2


# ============================================================
#  1. RANDOM FOREST LEARNING CURVE
# ============================================================

def rf_learning_curve(ratings_df, exercises_df, users_df):
    """Train RF at different training sizes with multiple seeds for CI."""
    logger.info("=" * 60)
    logger.info("GENERATING RANDOM FOREST LEARNING CURVES")
    logger.info("=" * 60)

    merged = ratings_df.merge(exercises_df, on='exercise_id', how='inner')
    merged = merged.merge(users_df, on='user_id', how='inner')

    if len(merged) < 50:
        logger.warning(f"Only {len(merged)} samples — skipping RF learning curve")
        return None

    merged['fitness_level_numeric'] = merged['fitness_level'].apply(encode_fitness_level)
    merged['bmi_category_numeric'] = merged['bmi'].apply(encode_bmi_category)
    merged['difficulty_gap'] = merged['difficulty_level'] - merged['fitness_level_numeric']
    merged['user_exercise_difficulty_gap'] = merged['difficulty_gap'].abs()
    merged['suitability'] = merged.apply(derive_suitability_label, axis=1)

    le_muscle = LabelEncoder()
    merged['target_muscle_group_encoded'] = le_muscle.fit_transform(merged['target_muscle_group'].fillna('unknown'))
    le_equip = LabelEncoder()
    merged['equipment_needed_encoded'] = le_equip.fit_transform(merged['equipment_needed'].fillna('bodyweight'))

    feature_cols = [
        'age', 'fitness_level_numeric', 'bmi_category_numeric',
        'difficulty_level', 'user_exercise_difficulty_gap',
        'target_muscle_group_encoded', 'equipment_needed_encoded',
        'default_duration_seconds', 'calories_burned_per_minute',
    ]

    if 'difficulty_perceived' in merged.columns:
        if merged['difficulty_perceived'].dtype == 'object':
            diff_map = {'easy': 1, 'too_easy': 1, 'moderate': 2, 'appropriate': 2,
                        'challenging': 3, 'hard': 4, 'too_hard': 4, 'very_hard': 5}
            merged['difficulty_perceived_numeric'] = merged['difficulty_perceived'].map(diff_map).fillna(2)
            feature_cols.append('difficulty_perceived_numeric')

    if 'enjoyment_rating' in merged.columns:
        if merged['enjoyment_rating'].dtype == 'object':
            merged['enjoyment_rating'] = pd.to_numeric(merged['enjoyment_rating'], errors='coerce').fillna(3)
        else:
            merged['enjoyment_rating'] = merged['enjoyment_rating'].fillna(3)
        feature_cols.append('enjoyment_rating')

    X = merged[feature_cols].fillna(0).values
    y = merged['suitability'].values

    logger.info(f"Total samples: {len(X)}, Features: {len(feature_cols)}")

    results = []

    for frac in TRAINING_FRACTIONS:
        accs = []
        precs = []
        recs = []
        f1s = []

        for seed in RF_RANDOM_SEEDS:
            class_counts = dict(zip(*np.unique(y, return_counts=True)))
            can_stratify = all(c >= 2 for c in class_counts.values())

            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y if can_stratify else None
            )

            # Subsample training set
            n = max(10, int(len(X_train_full) * frac))
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(X_train_full), size=n, replace=False)
            X_sub = X_train_full[indices]
            y_sub = y_train_full[indices]

            scaler = StandardScaler()
            X_sub_scaled = scaler.fit_transform(X_sub)
            X_test_scaled = scaler.transform(X_test)

            rf = RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=seed, n_jobs=-1,
            )
            rf.fit(X_sub_scaled, y_sub)
            y_pred = rf.predict(X_test_scaled)

            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recs.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        train_size = int(len(X) * 0.8 * frac)
        results.append({
            'fraction': frac,
            'train_size': train_size,
            'accuracy_mean': np.mean(accs),
            'accuracy_ci': confidence_interval_95(accs),
            'precision_mean': np.mean(precs),
            'precision_ci': confidence_interval_95(precs),
            'recall_mean': np.mean(recs),
            'recall_ci': confidence_interval_95(recs),
            'f1_mean': np.mean(f1s),
            'f1_ci': confidence_interval_95(f1s),
        })

        logger.info(f"  RF {frac*100:.0f}%: Acc={np.mean(accs):.4f} ±{confidence_interval_95(accs):.4f}, "
                     f"F1={np.mean(f1s):.4f} ±{confidence_interval_95(f1s):.4f}")

    return results


# ============================================================
#  2. COLLABORATIVE FILTERING LEARNING CURVE
# ============================================================

def cf_learning_curve(ratings_df, exercises_df):
    """Train CF (SVD) at different training sizes, evaluate on same held-out items."""
    logger.info("=" * 60)
    logger.info("GENERATING COLLABORATIVE FILTERING LEARNING CURVES")
    logger.info("=" * 60)

    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if not eligible_users:
        return None

    # Fixed test set: hold out last positive per user
    train_rows = []
    test_rows = []
    for uid in eligible_users:
        ur = ratings_df[ratings_df['user_id'] == uid].sort_values('timestamp')
        positive = ur[ur['rating'] >= 4.0]
        if len(positive) == 0:
            train_rows.append(ur)
            continue
        test_item = positive.iloc[-1]
        test_rows.append(test_item)
        train_rows.append(ur[ur.index != test_item.name])

    ineligible = ratings_df[~ratings_df['user_id'].isin(eligible_users)]
    train_df_full = pd.concat(train_rows + [ineligible], ignore_index=True)
    test_df = pd.DataFrame(test_rows)

    logger.info(f"Full train: {len(train_df_full)}, Test users: {len(test_df)}")

    all_exercise_ids = sorted(exercises_df['exercise_id'].tolist())

    results = []

    for frac in TRAINING_FRACTIONS:
        logger.info(f"  CF {frac*100:.0f}%: Training SVD...")

        # Subsample training ratings (stratified by user to keep coverage)
        if frac < 1.0:
            sub_dfs = []
            for uid, group in train_df_full.groupby('user_id'):
                n_sample = max(1, int(len(group) * frac))
                sub_dfs.append(group.sample(n=n_sample, random_state=42))
            sub_train = pd.concat(sub_dfs, ignore_index=True)
        else:
            sub_train = train_df_full

        logger.info(f"    Subsampled train: {len(sub_train)} ratings")

        cf_train = sub_train[['user_id', 'exercise_id', 'rating']].rename(
            columns={'exercise_id': 'item_id'}
        )
        cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=30)
        cf_model.fit(ratings_data=cf_train)

        # Evaluate on fixed test set with sampled candidates
        per_user_hr = []
        per_user_ndcg = []
        per_user_mrr = []

        for _, test_row in test_df.iterrows():
            uid = int(test_row['user_id'])
            test_eid = int(test_row['exercise_id'])
            seen = set(sub_train[sub_train['user_id'] == uid]['exercise_id'].tolist())

            candidates = sampled_candidates(test_eid, all_exercise_ids, seen, NUM_NEGATIVES, seed=uid)
            scored = [(eid, cf_model.predict(uid, eid)) for eid in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = [s[0] for s in scored]

            relevant = {test_eid}

            hit = 1.0 if test_eid in ranked[:10] else 0.0
            per_user_hr.append(hit)
            per_user_ndcg.append(ndcg_at_k(ranked, relevant, 10))
            try:
                per_user_mrr.append(1.0 / (ranked.index(test_eid) + 1))
            except ValueError:
                per_user_mrr.append(0.0)

        results.append({
            'fraction': frac,
            'train_size': len(sub_train),
            'hr10_mean': np.mean(per_user_hr),
            'hr10_ci': confidence_interval_95(per_user_hr),
            'ndcg10_mean': np.mean(per_user_ndcg),
            'ndcg10_ci': confidence_interval_95(per_user_ndcg),
            'mrr_mean': np.mean(per_user_mrr),
            'mrr_ci': confidence_interval_95(per_user_mrr),
        })

        logger.info(f"  CF {frac*100:.0f}%: HR@10={np.mean(per_user_hr):.4f} ±{confidence_interval_95(per_user_hr):.4f}, "
                     f"NDCG@10={np.mean(per_user_ndcg):.4f} ±{confidence_interval_95(per_user_ndcg):.4f}")

    return results


# ============================================================
#  3. HYBRID LEARNING CURVE
# ============================================================

def hybrid_learning_curve(ratings_df, exercises_df):
    """Train Hybrid at different training sizes. CB is fixed, CF varies."""
    logger.info("=" * 60)
    logger.info("GENERATING HYBRID RECOMMENDER LEARNING CURVES")
    logger.info("=" * 60)

    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if not eligible_users:
        return None

    # Fixed test set
    train_rows = []
    test_rows = []
    for uid in eligible_users:
        ur = ratings_df[ratings_df['user_id'] == uid].sort_values('timestamp')
        positive = ur[ur['rating'] >= 4.0]
        if len(positive) == 0:
            train_rows.append(ur)
            continue
        test_item = positive.iloc[-1]
        test_rows.append(test_item)
        train_rows.append(ur[ur.index != test_item.name])

    ineligible = ratings_df[~ratings_df['user_id'].isin(eligible_users)]
    train_df_full = pd.concat(train_rows + [ineligible], ignore_index=True)
    test_df = pd.DataFrame(test_rows)

    logger.info(f"Full train: {len(train_df_full)}, Test users: {len(test_df)}")

    all_exercise_ids = sorted(exercises_df['exercise_id'].tolist())

    def min_max_normalize(scores):
        vals = [s for _, s in scores]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [(eid, 0.5) for eid, _ in scores]
        return [(eid, (s - lo) / (hi - lo)) for eid, s in scores]

    results = []

    for frac in TRAINING_FRACTIONS:
        logger.info(f"  Hybrid {frac*100:.0f}%: Training...")

        # Subsample training ratings (stratified by user)
        if frac < 1.0:
            sub_dfs = []
            for uid, group in train_df_full.groupby('user_id'):
                n_sample = max(1, int(len(group) * frac))
                sub_dfs.append(group.sample(n=n_sample, random_state=42))
            sub_train = pd.concat(sub_dfs, ignore_index=True)
        else:
            sub_train = train_df_full

        logger.info(f"    Subsampled train: {len(sub_train)} ratings")

        # Train CF component on subsampled data
        cf_train = sub_train[['user_id', 'exercise_id', 'rating']].rename(
            columns={'exercise_id': 'item_id'}
        )
        cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=30)
        cf_model.fit(ratings_data=cf_train)

        # Build hybrid with subsampled training data
        hybrid = FinalHybridRecommender()
        hybrid.ratings_df = sub_train.copy()
        hybrid.exercises_df = exercises_df.copy()
        hybrid.user_item_matrix = sub_train.pivot_table(
            index='user_id', columns='exercise_id', values='rating', fill_value=np.nan
        )
        hybrid.set_trained_cf_model({'cf_model': cf_model, 'best_method': 'SVD'})
        hybrid.CONTENT_WEIGHT = 0.6
        hybrid.COLLABORATIVE_WEIGHT = 0.4

        def rank_hybrid(candidate_eids, uid, user_context):
            cf_raw = [(eid, hybrid.calculate_cf_score(uid, eid, user_context)) for eid in candidate_eids]
            cb_raw = [(eid, hybrid.calculate_content_score(uid, eid, user_context)) for eid in candidate_eids]
            cf_norm = {eid: s for eid, s in min_max_normalize(cf_raw)}
            cb_norm = {eid: s for eid, s in min_max_normalize(cb_raw)}
            combined = []
            for eid in candidate_eids:
                score = hybrid.CONTENT_WEIGHT * cb_norm[eid] + hybrid.COLLABORATIVE_WEIGHT * cf_norm[eid]
                combined.append((eid, score))
            combined.sort(key=lambda x: x[1], reverse=True)
            return [eid for eid, _ in combined]

        # Evaluate on fixed test set
        per_user_hr = []
        per_user_ndcg = []
        per_user_mrr = []

        for _, test_row in test_df.iterrows():
            uid = int(test_row['user_id'])
            test_eid = int(test_row['exercise_id'])
            seen = set(sub_train[sub_train['user_id'] == uid]['exercise_id'].tolist())

            user_context = hybrid.get_user_context(uid)

            candidates = sampled_candidates(test_eid, all_exercise_ids, seen, NUM_NEGATIVES, seed=uid)
            ranked = rank_hybrid(candidates, uid, user_context)

            relevant = {test_eid}

            hit = 1.0 if test_eid in ranked[:10] else 0.0
            per_user_hr.append(hit)
            per_user_ndcg.append(ndcg_at_k(ranked, relevant, 10))
            try:
                per_user_mrr.append(1.0 / (ranked.index(test_eid) + 1))
            except ValueError:
                per_user_mrr.append(0.0)

        results.append({
            'fraction': frac,
            'train_size': len(sub_train),
            'hr10_mean': np.mean(per_user_hr),
            'hr10_ci': confidence_interval_95(per_user_hr),
            'ndcg10_mean': np.mean(per_user_ndcg),
            'ndcg10_ci': confidence_interval_95(per_user_ndcg),
            'mrr_mean': np.mean(per_user_mrr),
            'mrr_ci': confidence_interval_95(per_user_mrr),
        })

        logger.info(f"  Hybrid {frac*100:.0f}%: HR@10={np.mean(per_user_hr):.4f} ±{confidence_interval_95(per_user_hr):.4f}, "
                     f"NDCG@10={np.mean(per_user_ndcg):.4f} ±{confidence_interval_95(per_user_ndcg):.4f}")

    return results


# ============================================================
#  REPORT
# ============================================================

def print_learning_curve_report(rf_results, cf_results, hybrid_results):
    print("\n")
    print("=" * 80)
    print("   FITNEASE ML MODEL — LEARNING CURVE DATA")
    print("   For Thesis Defense Presentation")
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Protocol: Sampled evaluation (50 candidates, He et al. 2017)")
    print("   CI: 95% Confidence Interval")
    print("=" * 80)

    # Table 1: Random Forest Learning Curve
    print("\n\nTABLE 1: Random Forest Classifier — Learning Curve")
    print("=" * 80)
    print(f"  {'Train %':>8s} | {'N_train':>8s} | {'Accuracy':>10s} | {'±95% CI':>8s} | "
          f"{'Precision':>10s} | {'±95% CI':>8s} | {'Recall':>10s} | {'±95% CI':>8s} | "
          f"{'F1 Score':>10s} | {'±95% CI':>8s}")
    print("  " + "-" * 105)
    if rf_results:
        for r in rf_results:
            print(f"  {r['fraction']*100:>7.0f}% | {r['train_size']:>8d} | "
                  f"{r['accuracy_mean']*100:>9.2f}% | {r['accuracy_ci']*100:>7.2f}% | "
                  f"{r['precision_mean']*100:>9.2f}% | {r['precision_ci']*100:>7.2f}% | "
                  f"{r['recall_mean']*100:>9.2f}% | {r['recall_ci']*100:>7.2f}% | "
                  f"{r['f1_mean']*100:>9.2f}% | {r['f1_ci']*100:>7.2f}%")
    else:
        print("  [SKIPPED — insufficient data]")

    print(f"\n  Note: Each data point averaged over {len(RF_RANDOM_SEEDS)} random seeds "
          f"(seeds: {RF_RANDOM_SEEDS})")
    print("  Test set: Fixed 20% holdout per seed")

    # Table 2: Collaborative Filtering Learning Curve
    print("\n\nTABLE 2: Collaborative Filtering (SVD) — Learning Curve")
    print("=" * 80)
    print(f"  {'Train %':>8s} | {'N_train':>8s} | {'HR@10':>8s} | {'±95% CI':>8s} | "
          f"{'NDCG@10':>8s} | {'±95% CI':>8s} | {'MRR':>8s} | {'±95% CI':>8s}")
    print("  " + "-" * 78)
    if cf_results:
        for r in cf_results:
            print(f"  {r['fraction']*100:>7.0f}% | {r['train_size']:>8d} | "
                  f"{r['hr10_mean']*100:>7.1f}% | {r['hr10_ci']*100:>7.1f}% | "
                  f"{r['ndcg10_mean']:>8.4f} | {r['ndcg10_ci']:>8.4f} | "
                  f"{r['mrr_mean']:>8.4f} | {r['mrr_ci']:>8.4f}")
    else:
        print("  [SKIPPED — insufficient data]")

    print("\n  Note: Training ratings subsampled per-user (stratified) to maintain user coverage")
    print("  Test set: Fixed last-positive-per-user holdout, 50-candidate sampled evaluation")

    # Table 3: Hybrid Learning Curve
    print("\n\nTABLE 3: Hybrid Recommender (60% CB + 40% CF) — Learning Curve")
    print("=" * 80)
    print(f"  {'Train %':>8s} | {'N_train':>8s} | {'HR@10':>8s} | {'±95% CI':>8s} | "
          f"{'NDCG@10':>8s} | {'±95% CI':>8s} | {'MRR':>8s} | {'±95% CI':>8s}")
    print("  " + "-" * 78)
    if hybrid_results:
        for r in hybrid_results:
            print(f"  {r['fraction']*100:>7.0f}% | {r['train_size']:>8d} | "
                  f"{r['hr10_mean']*100:>7.1f}% | {r['hr10_ci']*100:>7.1f}% | "
                  f"{r['ndcg10_mean']:>8.4f} | {r['ndcg10_ci']:>8.4f} | "
                  f"{r['mrr_mean']:>8.4f} | {r['mrr_ci']:>8.4f}")
    else:
        print("  [SKIPPED — insufficient data]")

    print("\n  Note: CB component uses exercise features (fixed). CF component varies with training data.")
    print("  Test set: Fixed last-positive-per-user holdout, 50-candidate sampled evaluation")

    # Table 4: Side-by-side comparison
    print("\n\nTABLE 4: Model Comparison Across Training Sizes — HR@10")
    print("=" * 80)
    if cf_results and hybrid_results:
        print(f"  {'Train %':>8s} | {'N_train':>8s} | {'CF HR@10':>10s} | {'Hybrid HR@10':>13s} | {'Improvement':>12s}")
        print("  " + "-" * 60)
        for i, frac in enumerate(TRAINING_FRACTIONS):
            cf_r = cf_results[i] if i < len(cf_results) else None
            hy_r = hybrid_results[i] if i < len(hybrid_results) else None
            if cf_r and hy_r:
                improvement = hy_r['hr10_mean'] - cf_r['hr10_mean']
                sign = "+" if improvement >= 0 else ""
                print(f"  {frac*100:>7.0f}% | {cf_r['train_size']:>8d} | "
                      f"{cf_r['hr10_mean']*100:>9.1f}% | "
                      f"{hy_r['hr10_mean']*100:>12.1f}% | "
                      f"{sign}{improvement*100:>10.1f}%")
    else:
        print("  [Not available — requires CF and Hybrid results]")

    # Table 5: NDCG@10 comparison
    print("\n\nTABLE 5: Model Comparison Across Training Sizes — NDCG@10")
    print("=" * 80)
    if cf_results and hybrid_results:
        print(f"  {'Train %':>8s} | {'CF NDCG@10':>11s} | {'Hybrid NDCG@10':>15s} | {'Improvement':>12s}")
        print("  " + "-" * 55)
        for i, frac in enumerate(TRAINING_FRACTIONS):
            cf_r = cf_results[i] if i < len(cf_results) else None
            hy_r = hybrid_results[i] if i < len(hybrid_results) else None
            if cf_r and hy_r:
                improvement = hy_r['ndcg10_mean'] - cf_r['ndcg10_mean']
                sign = "+" if improvement >= 0 else ""
                print(f"  {frac*100:>7.0f}% | "
                      f"{cf_r['ndcg10_mean']:>11.4f} | "
                      f"{hy_r['ndcg10_mean']:>15.4f} | "
                      f"{sign}{improvement:>11.4f}")
    else:
        print("  [Not available — requires CF and Hybrid results]")

    print("\n\n" + "=" * 80)
    print("  METHODOLOGY NOTES")
    print("=" * 80)
    print("""
  1. Training fractions: 20%, 40%, 60%, 80%, 100% of training data
  2. Random Forest: 5 random seeds per point, 80/20 train/test split per seed
  3. CF & Hybrid: Per-user stratified subsampling to maintain user coverage
  4. Test set: Fixed across all training sizes (last positive rating per user)
  5. Sampled evaluation: 50 candidates (1 positive + 49 random negatives)
  6. 95% CI computed as 1.96 * standard error of the mean
  7. Content-Based excluded (uses exercise features, not training data)
""")

    print("=" * 80)
    print("  END OF LEARNING CURVE DATA")
    print("=" * 80)
    print()


# ============================================================
#  MAIN
# ============================================================

def main():
    logger.info("Starting FitNEase Learning Curves Generation...")
    logger.info(f"Training fractions: {TRAINING_FRACTIONS}")

    try:
        tracking_conn, content_conn, auth_conn = get_database_connections()
        logger.info("Connected to all databases")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        ratings_df, exercises_df, users_df = load_data(tracking_conn, content_conn, auth_conn)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)
    finally:
        tracking_conn.close()
        content_conn.close()
        auth_conn.close()

    rf_results = rf_learning_curve(ratings_df, exercises_df, users_df)
    cf_results = cf_learning_curve(ratings_df, exercises_df)
    hybrid_results = hybrid_learning_curve(ratings_df, exercises_df)

    print_learning_curve_report(rf_results, cf_results, hybrid_results)

    # Save to file
    import io
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'learning_curves.txt')

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_learning_curve_report(rf_results, cf_results, hybrid_results)
    sys.stdout = old_stdout
    report_text = buf.getvalue()

    with open(output_file, 'w') as f:
        f.write(report_text)

    logger.info(f"Learning curve data saved to: {output_file}")


if __name__ == '__main__':
    main()
