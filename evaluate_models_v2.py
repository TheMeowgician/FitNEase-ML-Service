"""
FitNEase ML Model Evaluation Script v2
=======================================

Evaluates all 4 ML models using both full-catalog and sampled metrics.
Sampled evaluation follows He et al. (2017) protocol: 100 candidates per user
(1 positive + 99 random negatives). This is the standard in RecSys research.

Usage (inside Docker container):
    docker exec -it fitnease-ml python evaluate_models_v2.py
"""

import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import mysql.connector
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models.custom_classes import (
    FitNeaseContentBasedRecommender,
    ProperCollaborativeFiltering,
    FinalHybridRecommender,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_NEGATIVES = 99  # 99 negatives + 1 positive = 100 candidates


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


def precision_at_k(ranked_list, relevant_items, k):
    hits = sum(1 for item in ranked_list[:k] if item in relevant_items)
    return hits / k


def average_precision(ranked_list, relevant_items, k):
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant_items:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant_items), k) if relevant_items else 0.0


def sampled_candidates(test_exercise_id, all_exercise_ids, seen_ids, num_neg, seed):
    """Sample 99 random negatives + 1 positive = 100 candidates."""
    rng = random.Random(seed)
    pool = [eid for eid in all_exercise_ids if eid not in seen_ids and eid != test_exercise_id]
    negatives = rng.sample(pool, min(num_neg, len(pool)))
    return [test_exercise_id] + negatives


# ============================================================
#  ENCODING HELPERS (for RF)
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
#  1. RANDOM FOREST
# ============================================================

def evaluate_random_forest(ratings_df, exercises_df, users_df):
    logger.info("=" * 60)
    logger.info("EVALUATING RANDOM FOREST CLASSIFIER")
    logger.info("=" * 60)

    merged = ratings_df.merge(exercises_df, on='exercise_id', how='inner')
    merged = merged.merge(users_df, on='user_id', how='inner')

    if len(merged) < 50:
        logger.warning(f"Only {len(merged)} samples — skipping RF")
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
            diff_map = {'easy': 1, 'too_easy': 1, 'moderate': 2, 'appropriate': 2, 'challenging': 3, 'hard': 4, 'too_hard': 4, 'very_hard': 5}
            merged['difficulty_perceived_numeric'] = merged['difficulty_perceived'].map(diff_map).fillna(2)
            feature_cols.append('difficulty_perceived_numeric')
        else:
            merged['difficulty_perceived'] = merged['difficulty_perceived'].fillna(merged['difficulty_level'])
            feature_cols.append('difficulty_perceived')

    if 'enjoyment_rating' in merged.columns:
        if merged['enjoyment_rating'].dtype == 'object':
            merged['enjoyment_rating'] = pd.to_numeric(merged['enjoyment_rating'], errors='coerce').fillna(3)
        else:
            merged['enjoyment_rating'] = merged['enjoyment_rating'].fillna(3)
        feature_cols.append('enjoyment_rating')

    X = merged[feature_cols].fillna(0).values
    y = merged['suitability'].values

    logger.info(f"Feature matrix: {X.shape}, Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    can_stratify = all(c >= 2 for c in class_counts.values())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    present_classes = sorted(set(y_test) | set(y_pred))
    class_label_map = {0: 'Unsuitable', 1: 'Suitable-Easy', 2: 'Suitable-Appropriate', 3: 'Suitable-Hard'}
    present_names = [class_label_map[c] for c in present_classes]
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    class_report = classification_report(
        y_test, y_pred, labels=present_classes, target_names=present_names, zero_division=0,
    )

    importances = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)

    logger.info(f"RF Accuracy: {accuracy:.4f}")
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'confusion_matrix': cm, 'class_names': present_names, 'class_report': class_report,
        'feature_importance': importances,
        'train_size': len(X_train), 'test_size': len(X_test),
    }


# ============================================================
#  2. CONTENT-BASED (Full Catalog + Sampled)
# ============================================================

def evaluate_content_based(ratings_df, exercises_df):
    logger.info("=" * 60)
    logger.info("EVALUATING CONTENT-BASED RECOMMENDER")
    logger.info("=" * 60)

    ex_for_model = exercises_df.copy()
    if 'exercise_name' in ex_for_model.columns and 'name' not in ex_for_model.columns:
        ex_for_model['name'] = ex_for_model['exercise_name']
    if 'exercise_id' in ex_for_model.columns and 'id' not in ex_for_model.columns:
        ex_for_model['id'] = ex_for_model['exercise_id']

    cb_model = FitNeaseContentBasedRecommender()
    cb_model.fit(ex_for_model)

    if not cb_model.is_fitted:
        logger.error("Content-based model failed to fit")
        return None

    all_exercise_ids = sorted(exercises_df['exercise_id'].tolist())
    model_exercise_ids = set(cb_model.exercise_ids) if cb_model.exercise_ids else set()

    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if not eligible_users:
        return None

    # Accumulators: full catalog
    fc_hits10 = 0; fc_hits15 = 0; fc_ndcg10 = []; fc_ndcg15 = []
    # Accumulators: sampled
    s_hits10 = 0; s_hits15 = 0; s_ndcg10 = []; s_ndcg15 = []; s_mrr = []; s_prec10 = []
    total = 0; all_rec = set()

    for user_id in eligible_users:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].sort_values('timestamp')
        if len(user_ratings) < 5:
            continue

        test_item = user_ratings.iloc[-1]
        train_items = user_ratings.iloc[:-1]
        test_eid = int(test_item['exercise_id'])

        if test_eid not in model_exercise_ids:
            continue

        liked = train_items[train_items['rating'] >= 3.5]
        if len(liked) == 0:
            liked = train_items.nlargest(3, 'rating')

        # Build score function: max similarity from any liked exercise
        seen_ids = set(train_items['exercise_id'].tolist())
        candidate_scores = defaultdict(float)

        for _, row in liked.iterrows():
            ex_id = int(row['exercise_id'])
            if ex_id not in model_exercise_ids:
                continue
            recs = cb_model.get_recommendations(exercise_id=ex_id, num_recommendations=30)
            for rec in recs:
                rec_id = rec.get('exercise_id')
                if rec_id and rec_id not in seen_ids:
                    sim = rec.get('similarity_score', 0)
                    if sim > candidate_scores[rec_id]:
                        candidate_scores[rec_id] = sim

        if not candidate_scores:
            continue

        total += 1
        relevant = {test_eid}

        # --- Full Catalog ---
        ranked_fc = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)
        all_rec.update(ranked_fc[:15])

        if test_eid in ranked_fc[:10]: fc_hits10 += 1
        if test_eid in ranked_fc[:15]: fc_hits15 += 1
        fc_ndcg10.append(ndcg_at_k(ranked_fc, relevant, 10))
        fc_ndcg15.append(ndcg_at_k(ranked_fc, relevant, 15))

        # --- Sampled (100 candidates) ---
        candidates = sampled_candidates(test_eid, all_exercise_ids, seen_ids, NUM_NEGATIVES, seed=user_id)
        scored = sorted(candidates, key=lambda eid: candidate_scores.get(eid, 0.0), reverse=True)

        if test_eid in scored[:10]: s_hits10 += 1
        if test_eid in scored[:15]: s_hits15 += 1
        s_ndcg10.append(ndcg_at_k(scored, relevant, 10))
        s_ndcg15.append(ndcg_at_k(scored, relevant, 15))
        s_prec10.append(precision_at_k(scored, relevant, 10))
        try:
            rank = scored.index(test_eid) + 1
            s_mrr.append(1.0 / rank)
        except ValueError:
            s_mrr.append(0.0)

    if total == 0:
        return None

    coverage = len(all_rec) / len(all_exercise_ids) if all_exercise_ids else 0

    logger.info(f"CB Full-Catalog HR@10: {fc_hits10/total:.4f}, Sampled HR@10: {s_hits10/total:.4f}")
    return {
        'users_evaluated': total,
        'fc_hit_rate_10': fc_hits10 / total, 'fc_hit_rate_15': fc_hits15 / total,
        'fc_ndcg_10': np.mean(fc_ndcg10), 'fc_ndcg_15': np.mean(fc_ndcg15),
        's_hit_rate_10': s_hits10 / total, 's_hit_rate_15': s_hits15 / total,
        's_ndcg_10': np.mean(s_ndcg10), 's_ndcg_15': np.mean(s_ndcg15),
        's_precision_10': np.mean(s_prec10), 's_mrr': np.mean(s_mrr),
        'coverage': coverage,
    }


# ============================================================
#  3. COLLABORATIVE FILTERING (Full Catalog + Sampled)
# ============================================================

def evaluate_collaborative(ratings_df):
    logger.info("=" * 60)
    logger.info("EVALUATING COLLABORATIVE FILTERING (SVD)")
    logger.info("=" * 60)

    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if not eligible_users:
        return None

    train_rows = []
    test_rows = []
    for uid in eligible_users:
        ur = ratings_df[ratings_df['user_id'] == uid].sort_values('timestamp')
        test_rows.append(ur.iloc[-1])
        train_rows.append(ur.iloc[:-1])

    ineligible = ratings_df[~ratings_df['user_id'].isin(eligible_users)]
    train_df = pd.concat(train_rows + [ineligible], ignore_index=True)
    test_df = pd.DataFrame(test_rows)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    cf_train = train_df[['user_id', 'exercise_id', 'rating']].rename(columns={'exercise_id': 'item_id'})
    cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=50)
    cf_model.fit(ratings_data=cf_train)

    all_exercise_ids = sorted(ratings_df['exercise_id'].unique())

    # Accumulators
    rmse_err = []; mae_err = []
    fc_hits10 = 0; fc_ndcg10 = []; fc_mrr = []
    s_hits10 = 0; s_ndcg10 = []; s_mrr = []; s_prec10 = []
    total = 0

    for _, test_row in test_df.iterrows():
        uid = int(test_row['user_id'])
        test_eid = int(test_row['exercise_id'])
        actual = float(test_row['rating'])

        pred = cf_model.predict(uid, test_eid)
        rmse_err.append((pred - actual) ** 2)
        mae_err.append(abs(pred - actual))

        seen = set(train_df[train_df['user_id'] == uid]['exercise_id'].tolist())
        total += 1
        relevant = {test_eid}

        # --- Full Catalog ---
        fc_candidates = [eid for eid in all_exercise_ids if eid not in seen]
        if test_eid not in fc_candidates:
            fc_candidates.append(test_eid)

        fc_scored = [(eid, cf_model.predict(uid, eid)) for eid in fc_candidates]
        fc_scored.sort(key=lambda x: x[1], reverse=True)
        fc_ranked = [s[0] for s in fc_scored]

        if test_eid in fc_ranked[:10]: fc_hits10 += 1
        fc_ndcg10.append(ndcg_at_k(fc_ranked, relevant, 10))
        try:
            fc_mrr.append(1.0 / (fc_ranked.index(test_eid) + 1))
        except ValueError:
            fc_mrr.append(0.0)

        # --- Sampled (100 candidates) ---
        candidates = sampled_candidates(test_eid, all_exercise_ids, seen, NUM_NEGATIVES, seed=uid)
        s_scored = [(eid, cf_model.predict(uid, eid)) for eid in candidates]
        s_scored.sort(key=lambda x: x[1], reverse=True)
        s_ranked = [s[0] for s in s_scored]

        if test_eid in s_ranked[:10]: s_hits10 += 1
        s_ndcg10.append(ndcg_at_k(s_ranked, relevant, 10))
        s_prec10.append(precision_at_k(s_ranked, relevant, 10))
        try:
            s_mrr.append(1.0 / (s_ranked.index(test_eid) + 1))
        except ValueError:
            s_mrr.append(0.0)

    if total == 0:
        return None

    logger.info(f"CF Full HR@10: {fc_hits10/total:.4f}, Sampled HR@10: {s_hits10/total:.4f}")
    return {
        'users_evaluated': total,
        'rmse': np.sqrt(np.mean(rmse_err)), 'mae': np.mean(mae_err),
        'fc_hit_rate_10': fc_hits10 / total, 'fc_ndcg_10': np.mean(fc_ndcg10), 'fc_mrr': np.mean(fc_mrr),
        's_hit_rate_10': s_hits10 / total, 's_ndcg_10': np.mean(s_ndcg10),
        's_mrr': np.mean(s_mrr), 's_precision_10': np.mean(s_prec10),
    }


# ============================================================
#  4. HYBRID RECOMMENDER (Full Catalog + Sampled)
# ============================================================

def evaluate_hybrid(ratings_df, exercises_df):
    logger.info("=" * 60)
    logger.info("EVALUATING HYBRID RECOMMENDER")
    logger.info("=" * 60)

    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if not eligible_users:
        return None

    train_rows = []; test_rows = []
    for uid in eligible_users:
        ur = ratings_df[ratings_df['user_id'] == uid].sort_values('timestamp')
        test_rows.append(ur.iloc[-1])
        train_rows.append(ur.iloc[:-1])

    ineligible = ratings_df[~ratings_df['user_id'].isin(eligible_users)]
    train_df = pd.concat(train_rows + [ineligible], ignore_index=True)
    test_df = pd.DataFrame(test_rows)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Train CF component
    cf_train = train_df[['user_id', 'exercise_id', 'rating']].rename(columns={'exercise_id': 'item_id'})
    cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=50)
    cf_model.fit(ratings_data=cf_train)

    # Build hybrid
    hybrid = FinalHybridRecommender()
    hybrid.ratings_df = train_df.copy()
    hybrid.exercises_df = exercises_df.copy()
    hybrid.user_item_matrix = train_df.pivot_table(
        index='user_id', columns='exercise_id', values='rating', fill_value=np.nan
    )
    hybrid.set_trained_cf_model({'cf_model': cf_model, 'best_method': 'SVD'})

    all_exercise_ids = sorted(exercises_df['exercise_id'].tolist())
    all_eid_set = set(all_exercise_ids)

    # Accumulators
    fc_hits10 = 0; fc_hits15 = 0; fc_ndcg10 = []
    s_hits10 = 0; s_hits15 = 0; s_ndcg10 = []; s_map10 = []; s_prec10 = []; s_mrr = []
    all_rec = set(); total = 0

    for _, test_row in test_df.iterrows():
        uid = int(test_row['user_id'])
        test_eid = int(test_row['exercise_id'])
        seen = set(train_df[train_df['user_id'] == uid]['exercise_id'].tolist())
        relevant = {test_eid}

        # Pre-compute user context once
        user_context = hybrid.get_user_context(uid)

        def hybrid_score(eid):
            cf_s = hybrid.calculate_cf_score(uid, eid, user_context)
            content_s = hybrid.calculate_content_score(uid, eid, user_context)
            return hybrid.CONTENT_WEIGHT * content_s + hybrid.COLLABORATIVE_WEIGHT * cf_s

        total += 1

        # --- Full Catalog: score all unseen exercises ---
        fc_candidates = [eid for eid in all_exercise_ids if eid not in seen]
        if test_eid not in fc_candidates:
            fc_candidates.append(test_eid)

        fc_scored = [(eid, hybrid_score(eid)) for eid in fc_candidates]
        fc_scored.sort(key=lambda x: x[1], reverse=True)
        fc_ranked = [s[0] for s in fc_scored]
        all_rec.update(fc_ranked[:15])

        if test_eid in fc_ranked[:10]: fc_hits10 += 1
        if test_eid in fc_ranked[:15]: fc_hits15 += 1
        fc_ndcg10.append(ndcg_at_k(fc_ranked, relevant, 10))

        # --- Sampled (100 candidates) ---
        candidates = sampled_candidates(test_eid, all_exercise_ids, seen, NUM_NEGATIVES, seed=uid)
        s_scored = [(eid, hybrid_score(eid)) for eid in candidates]
        s_scored.sort(key=lambda x: x[1], reverse=True)
        s_ranked = [s[0] for s in s_scored]

        if test_eid in s_ranked[:10]: s_hits10 += 1
        if test_eid in s_ranked[:15]: s_hits15 += 1
        s_ndcg10.append(ndcg_at_k(s_ranked, relevant, 10))
        s_map10.append(average_precision(s_ranked, relevant, 10))
        s_prec10.append(precision_at_k(s_ranked, relevant, 10))
        try:
            s_mrr.append(1.0 / (s_ranked.index(test_eid) + 1))
        except ValueError:
            s_mrr.append(0.0)

    if total == 0:
        return None

    coverage = len(all_rec) / len(all_eid_set) if all_eid_set else 0

    # Diversity
    diversity = 0.0
    if all_rec:
        rec_ex = exercises_df[exercises_df['exercise_id'].isin(all_rec)]
        if len(rec_ex) > 1:
            um = rec_ex['target_muscle_group'].nunique()
            ud = rec_ex['difficulty_level'].nunique()
            ue = rec_ex['equipment_needed'].nunique()
            diversity = min((um + ud + ue) / (3 * max(len(rec_ex), 1)), 1.0)

    logger.info(f"Hybrid Full HR@10: {fc_hits10/total:.4f}, Sampled HR@10: {s_hits10/total:.4f}")
    return {
        'users_evaluated': total,
        'fc_hit_rate_10': fc_hits10 / total, 'fc_hit_rate_15': fc_hits15 / total,
        'fc_ndcg_10': np.mean(fc_ndcg10),
        's_hit_rate_10': s_hits10 / total, 's_hit_rate_15': s_hits15 / total,
        's_ndcg_10': np.mean(s_ndcg10), 's_map_10': np.mean(s_map10),
        's_precision_10': np.mean(s_prec10), 's_mrr': np.mean(s_mrr),
        'coverage': coverage, 'diversity': diversity,
    }


# ============================================================
#  REPORT
# ============================================================

def print_report(rf, cb, cf, hy):
    print("\n")
    print("=" * 60)
    print("   FITNEASE ML MODEL EVALUATION REPORT v2")
    print("   Sampled metrics: 100 candidates (He et al. 2017)")
    print("=" * 60)

    # 1. Random Forest
    print("\n1. RANDOM FOREST (Workout Suitability Classifier)")
    print("-" * 50)
    if rf:
        print(f"   Train/Test:            {rf['train_size']} / {rf['test_size']}")
        print(f"   Accuracy:              {rf['accuracy']*100:.1f}%")
        print(f"   Precision (weighted):  {rf['precision']*100:.1f}%")
        print(f"   Recall (weighted):     {rf['recall']*100:.1f}%")
        print(f"   F1 Score (weighted):   {rf['f1']*100:.1f}%")
        print(f"\n   Per-class Report:")
        for line in rf['class_report'].split('\n'):
            if line.strip():
                print(f"   {line}")
        print(f"\n   Confusion Matrix:")
        cm = rf['confusion_matrix']
        labels = rf.get('class_names', [f'C{i}' for i in range(cm.shape[0])])
        short = [l.replace('Suitable-', '') for l in labels]
        print("   {:>14s}".format("") + "".join(f"  {l:>12s}" for l in short))
        for i, row in enumerate(cm):
            print(f"   {short[i]:>14s}" + "".join(f"  {v:>12d}" for v in row))
        print(f"\n   Top Features:")
        for feat, imp in rf['feature_importance'][:5]:
            print(f"     {feat:<40s} {imp:.4f}")
    else:
        print("   [SKIPPED]")

    # 2. Content-Based
    print("\n2. CONTENT-BASED RECOMMENDER")
    print("-" * 50)
    if cb:
        print(f"   Users evaluated:       {cb['users_evaluated']}")
        print(f"\n   --- Full Catalog ---")
        print(f"   Hit Rate@10:           {cb['fc_hit_rate_10']*100:.1f}%")
        print(f"   Hit Rate@15:           {cb['fc_hit_rate_15']*100:.1f}%")
        print(f"   NDCG@10:               {cb['fc_ndcg_10']:.4f}")
        print(f"\n   --- Sampled (100 candidates) ---")
        print(f"   Hit Rate@10:           {cb['s_hit_rate_10']*100:.1f}%")
        print(f"   Hit Rate@15:           {cb['s_hit_rate_15']*100:.1f}%")
        print(f"   NDCG@10:               {cb['s_ndcg_10']:.4f}")
        print(f"   Precision@10:          {cb['s_precision_10']:.4f}")
        print(f"   MRR:                   {cb['s_mrr']:.4f}")
        print(f"   Coverage:              {cb['coverage']*100:.1f}%")
    else:
        print("   [SKIPPED]")

    # 3. Collaborative Filtering
    print("\n3. COLLABORATIVE FILTERING (SVD)")
    print("-" * 50)
    if cf:
        print(f"   Users evaluated:       {cf['users_evaluated']}")
        print(f"   RMSE:                  {cf['rmse']:.4f}")
        print(f"   MAE:                   {cf['mae']:.4f}")
        print(f"\n   --- Full Catalog ---")
        print(f"   Hit Rate@10:           {cf['fc_hit_rate_10']*100:.1f}%")
        print(f"   NDCG@10:               {cf['fc_ndcg_10']:.4f}")
        print(f"   MRR:                   {cf['fc_mrr']:.4f}")
        print(f"\n   --- Sampled (100 candidates) ---")
        print(f"   Hit Rate@10:           {cf['s_hit_rate_10']*100:.1f}%")
        print(f"   NDCG@10:               {cf['s_ndcg_10']:.4f}")
        print(f"   Precision@10:          {cf['s_precision_10']:.4f}")
        print(f"   MRR:                   {cf['s_mrr']:.4f}")
    else:
        print("   [SKIPPED]")

    # 4. Hybrid
    print("\n4. HYBRID RECOMMENDER")
    print("-" * 50)
    if hy:
        print(f"   Users evaluated:       {hy['users_evaluated']}")
        print(f"\n   --- Full Catalog ---")
        print(f"   Hit Rate@10:           {hy['fc_hit_rate_10']*100:.1f}%")
        print(f"   Hit Rate@15:           {hy['fc_hit_rate_15']*100:.1f}%")
        print(f"   NDCG@10:               {hy['fc_ndcg_10']:.4f}")
        print(f"\n   --- Sampled (100 candidates) ---")
        print(f"   Hit Rate@10:           {hy['s_hit_rate_10']*100:.1f}%")
        print(f"   Hit Rate@15:           {hy['s_hit_rate_15']*100:.1f}%")
        print(f"   NDCG@10:               {hy['s_ndcg_10']:.4f}")
        print(f"   MAP@10:                {hy['s_map_10']:.4f}")
        print(f"   Precision@10:          {hy['s_precision_10']:.4f}")
        print(f"   MRR:                   {hy['s_mrr']:.4f}")
        print(f"   Coverage:              {hy['coverage']*100:.1f}%")
        print(f"   Diversity:             {hy['diversity']:.4f}")
    else:
        print("   [SKIPPED]")

    print("\n" + "=" * 60)
    print("  All metrics computed from real data with train/test splits.")
    print("  Sampled metrics use 100 candidates (He et al. 2017).")
    print("=" * 60)
    print()


# ============================================================
#  MAIN
# ============================================================

def main():
    logger.info("Starting FitNEase ML Model Evaluation v2...")

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

    rf = evaluate_random_forest(ratings_df, exercises_df, users_df)
    cb = evaluate_content_based(ratings_df, exercises_df)
    cf = evaluate_collaborative(ratings_df)
    hy = evaluate_hybrid(ratings_df, exercises_df)

    print_report(rf, cb, cf, hy)


if __name__ == '__main__':
    main()
