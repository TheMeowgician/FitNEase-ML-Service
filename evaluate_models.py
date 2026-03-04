"""
FitNEase ML Model Evaluation Script
====================================

Standalone script that evaluates all 4 ML models with proper train/test splits
using real production data. Creates NO changes to existing files or production models.

Usage (inside Docker container):
    docker exec -it fitnease-ml python evaluate_models.py

Models evaluated:
    1. Random Forest - 4-class workout suitability classifier
    2. Content-Based Recommender - exercise similarity recommender
    3. Collaborative Filtering (SVD) - user-item rating prediction
    4. Hybrid Recommender - weighted blend of content + collaborative
"""

import os
import sys
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

# Import project model classes (same as production)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models.custom_classes import (
    FitNeaseContentBasedRecommender,
    ProperCollaborativeFiltering,
    FinalHybridRecommender,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  DATABASE CONNECTIONS (reuses pattern from retrain_hybrid_model.py)
# ============================================================

def get_database_connections():
    """Connect to tracking, content, and auth databases."""
    password = os.getenv('MYSQL_ROOT_PASSWORD', '5mMFUgBvx7xu7rvAI7p0T7rc9ZoHc6yl3zbpIWKV6jU=')

    tracking_conn = mysql.connector.connect(
        host=os.getenv('TRACKING_DB_HOST', 'fitnease-tracking-db'),
        port=int(os.getenv('TRACKING_DB_PORT', 3306)),
        database='fitnease_tracking_db',
        user='root',
        password=password,
    )

    content_conn = mysql.connector.connect(
        host=os.getenv('CONTENT_DB_HOST', 'fitnease-content-db'),
        port=int(os.getenv('CONTENT_DB_PORT', 3306)),
        database='fitnease_content_db',
        user='root',
        password=password,
    )

    auth_conn = mysql.connector.connect(
        host=os.getenv('AUTH_DB_HOST', 'fitnease-auth-db'),
        port=int(os.getenv('AUTH_DB_PORT', 3306)),
        database='fitnease_auth_db',
        user='root',
        password=password,
    )

    return tracking_conn, content_conn, auth_conn


# ============================================================
#  DATA LOADING
# ============================================================

def load_data(tracking_conn, content_conn, auth_conn):
    """Load ratings, exercises, and user profiles from production databases."""

    # --- Ratings ---
    ratings_df = pd.read_sql("""
        SELECT
            user_id,
            exercise_id,
            rating_value AS rating,
            difficulty_perceived,
            enjoyment_rating,
            would_do_again,
            completed,
            rated_at AS timestamp
        FROM workout_exercise_ratings
        WHERE completed = 1
        ORDER BY rated_at
    """, tracking_conn)
    logger.info(f"Loaded {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users")

    # --- Exercises ---
    exercises_df = pd.read_sql("""
        SELECT
            exercise_id,
            exercise_name,
            target_muscle_group,
            difficulty_level,
            equipment_needed,
            default_duration_seconds,
            calories_burned_per_minute,
            instructions
        FROM exercises
    """, content_conn)
    exercises_df['difficulty_level'] = exercises_df['difficulty_level'].fillna(2).astype(int)
    logger.info(f"Loaded {len(exercises_df)} exercises")

    # --- User profiles ---
    users_df = pd.read_sql("""
        SELECT
            id AS user_id,
            age,
            fitness_level,
            height,
            weight
        FROM users
        WHERE age IS NOT NULL
    """, auth_conn)
    # Compute BMI where possible
    users_df['bmi'] = np.where(
        (users_df['height'] > 0) & (users_df['weight'] > 0),
        users_df['weight'] / ((users_df['height'] / 100) ** 2),
        23.0,
    )
    logger.info(f"Loaded {len(users_df)} user profiles")

    return ratings_df, exercises_df, users_df


# ============================================================
#  HELPERS
# ============================================================

def encode_fitness_level(level):
    """Map fitness level string to numeric (matches random_forest_predictor.py)."""
    mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
    if isinstance(level, str):
        return mapping.get(level.lower(), 2)
    return 2


def encode_bmi_category(bmi):
    """Map BMI value to category number."""
    if bmi < 18.5:
        return 1  # underweight
    elif bmi < 25:
        return 2  # normal
    elif bmi < 30:
        return 3  # overweight
    return 4  # obese


def derive_suitability_label(row):
    """Derive 4-class suitability label from rating + difficulty gap.

    Classes:
        0 = Unsuitable  (low rating — bad match)
        1 = Suitable-Easy  (decent rating, exercise easier than user level)
        2 = Suitable-Appropriate  (high rating, difficulty matches user)
        3 = Suitable-Hard  (decent rating, exercise harder than user level)
    """
    rating = row['rating']
    gap = row['difficulty_gap']  # exercise_difficulty - user_fitness_numeric

    if rating < 3:
        return 0  # Unsuitable
    if gap < -0.5:
        return 1  # Suitable-Easy (exercise easier)
    if gap > 0.5:
        return 3  # Suitable-Hard (exercise harder)
    return 2  # Suitable-Appropriate


def ndcg_at_k(ranked_list, relevant_items, k):
    """Compute NDCG@K for a single ranked list."""
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG: all relevant items at the top
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(ranked_list, relevant_items, k):
    """Compute Precision@K."""
    hits = sum(1 for item in ranked_list[:k] if item in relevant_items)
    return hits / k


def average_precision(ranked_list, relevant_items, k):
    """Compute Average Precision for a single ranked list."""
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in relevant_items:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(len(relevant_items), k) if relevant_items else 0.0


# ============================================================
#  1. RANDOM FOREST EVALUATION
# ============================================================

def evaluate_random_forest(ratings_df, exercises_df, users_df):
    """Evaluate Random Forest suitability classifier with 80/20 split."""
    logger.info("=" * 60)
    logger.info("EVALUATING RANDOM FOREST CLASSIFIER")
    logger.info("=" * 60)

    # Merge ratings with exercises and user profiles
    merged = ratings_df.merge(exercises_df, on='exercise_id', how='inner')
    merged = merged.merge(users_df, on='user_id', how='inner')

    if len(merged) < 50:
        logger.warning(f"Only {len(merged)} samples after merge — not enough for RF evaluation")
        return None

    # Build feature matrix (matching random_forest_predictor.py:217-299)
    merged['fitness_level_numeric'] = merged['fitness_level'].apply(encode_fitness_level)
    merged['bmi_category_numeric'] = merged['bmi'].apply(encode_bmi_category)
    merged['difficulty_gap'] = merged['difficulty_level'] - merged['fitness_level_numeric']
    merged['user_exercise_difficulty_gap'] = merged['difficulty_gap'].abs()

    # Derive target label
    merged['suitability'] = merged.apply(derive_suitability_label, axis=1)

    # Encode categoricals
    le_muscle = LabelEncoder()
    merged['target_muscle_group_encoded'] = le_muscle.fit_transform(
        merged['target_muscle_group'].fillna('unknown')
    )
    le_equip = LabelEncoder()
    merged['equipment_needed_encoded'] = le_equip.fit_transform(
        merged['equipment_needed'].fillna('bodyweight')
    )

    # Feature columns (matching the 16 features in random_forest_predictor.py)
    feature_cols = [
        'age',
        'fitness_level_numeric',
        'bmi_category_numeric',
        'difficulty_level',
        'user_exercise_difficulty_gap',
        'target_muscle_group_encoded',
        'equipment_needed_encoded',
        'default_duration_seconds',
        'calories_burned_per_minute',
    ]

    # Add derived features where available
    if 'difficulty_perceived' in merged.columns:
        merged['difficulty_perceived'] = merged['difficulty_perceived'].fillna(
            merged['difficulty_level']
        )
        feature_cols.append('difficulty_perceived')
    if 'enjoyment_rating' in merged.columns:
        merged['enjoyment_rating'] = merged['enjoyment_rating'].fillna(3)
        feature_cols.append('enjoyment_rating')

    X = merged[feature_cols].fillna(0).values
    y = merged['suitability'].values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 80/20 stratified split (fall back to non-stratified if a class has < 2 samples)
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    can_stratify = all(c >= 2 for c in class_counts.values())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    # Use only the classes that actually appear in the data
    present_classes = sorted(set(y_test) | set(y_pred))
    class_label_map = {0: 'Unsuitable', 1: 'Suitable-Easy', 2: 'Suitable-Appropriate', 3: 'Suitable-Hard'}
    present_names = [class_label_map[c] for c in present_classes]
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    class_report = classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    )

    # Feature importance
    importances = dict(zip(feature_cols, rf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_names': present_names,
        'class_report': class_report,
        'feature_importance': sorted_imp,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

    logger.info(f"RF Accuracy: {accuracy:.4f}")
    return results


# ============================================================
#  2. CONTENT-BASED EVALUATION
# ============================================================

def evaluate_content_based(ratings_df, exercises_df):
    """Evaluate content-based recommender with Leave-One-Out."""
    logger.info("=" * 60)
    logger.info("EVALUATING CONTENT-BASED RECOMMENDER")
    logger.info("=" * 60)

    # Fit content-based model on all exercises
    # Need to rename columns to match what the model expects
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

    all_exercise_ids = set(exercises_df['exercise_id'].tolist())
    model_exercise_ids = set(cb_model.exercise_ids) if cb_model.exercise_ids else set()

    # Leave-One-Out: for each user with >= 5 ratings, hold out most recent
    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if len(eligible_users) == 0:
        logger.warning("No eligible users for content-based evaluation")
        return None

    hits_at_10 = 0
    hits_at_15 = 0
    ndcg_scores_10 = []
    ndcg_scores_15 = []
    precision_scores_10 = []
    all_recommended = set()
    total_users = 0

    for user_id in eligible_users:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].sort_values('timestamp')
        if len(user_ratings) < 5:
            continue

        # Hold out the most recent rating
        test_item = user_ratings.iloc[-1]
        train_items = user_ratings.iloc[:-1]

        test_exercise_id = int(test_item['exercise_id'])
        if test_exercise_id not in model_exercise_ids:
            continue

        # Get user's liked exercises from train set (rating >= 3.5)
        liked = train_items[train_items['rating'] >= 3.5]
        if len(liked) == 0:
            liked = train_items.nlargest(3, 'rating')

        # Aggregate recommendations from all liked exercises
        candidate_scores = defaultdict(float)
        seen_exercises = set(train_items['exercise_id'].tolist())

        for _, row in liked.iterrows():
            ex_id = int(row['exercise_id'])
            if ex_id not in model_exercise_ids:
                continue
            recs = cb_model.get_recommendations(exercise_id=ex_id, num_recommendations=20)
            for rec in recs:
                rec_id = rec.get('exercise_id')
                if rec_id and rec_id not in seen_exercises:
                    sim = rec.get('similarity_score', 0)
                    if sim > candidate_scores[rec_id]:
                        candidate_scores[rec_id] = sim

        if not candidate_scores:
            continue

        # Rank candidates by aggregated similarity score
        ranked = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)
        relevant = {test_exercise_id}

        total_users += 1
        all_recommended.update(ranked[:15])

        # Hit Rate
        if test_exercise_id in ranked[:10]:
            hits_at_10 += 1
        if test_exercise_id in ranked[:15]:
            hits_at_15 += 1

        # NDCG
        ndcg_scores_10.append(ndcg_at_k(ranked, relevant, 10))
        ndcg_scores_15.append(ndcg_at_k(ranked, relevant, 15))

        # Precision
        precision_scores_10.append(precision_at_k(ranked, relevant, 10))

    if total_users == 0:
        logger.warning("No users evaluated for content-based")
        return None

    coverage = len(all_recommended) / len(all_exercise_ids) if all_exercise_ids else 0

    results = {
        'hit_rate_10': hits_at_10 / total_users,
        'hit_rate_15': hits_at_15 / total_users,
        'ndcg_10': np.mean(ndcg_scores_10),
        'ndcg_15': np.mean(ndcg_scores_15),
        'precision_10': np.mean(precision_scores_10),
        'coverage': coverage,
        'users_evaluated': total_users,
    }

    logger.info(f"Content-Based Hit Rate@10: {results['hit_rate_10']:.4f}")
    return results


# ============================================================
#  3. COLLABORATIVE FILTERING EVALUATION
# ============================================================

def evaluate_collaborative(ratings_df):
    """Evaluate collaborative filtering (SVD) with Leave-One-Out."""
    logger.info("=" * 60)
    logger.info("EVALUATING COLLABORATIVE FILTERING (SVD)")
    logger.info("=" * 60)

    # Leave-One-Out split
    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if len(eligible_users) == 0:
        logger.warning("No eligible users for collaborative evaluation")
        return None

    # Build train/test sets
    train_rows = []
    test_rows = []
    for user_id in eligible_users:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].sort_values('timestamp')
        test_rows.append(user_ratings.iloc[-1])
        train_rows.append(user_ratings.iloc[:-1])

    # Also include ratings from users with < 5 ratings (train only)
    ineligible_mask = ~ratings_df['user_id'].isin(eligible_users)
    ineligible_ratings = ratings_df[ineligible_mask]

    train_df = pd.concat(train_rows + [ineligible_ratings], ignore_index=True)
    test_df = pd.DataFrame(test_rows)

    logger.info(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")

    # Train CF model (expects 'item_id' column, not 'exercise_id')
    cf_train = train_df[['user_id', 'exercise_id', 'rating']].rename(
        columns={'exercise_id': 'item_id'}
    )
    cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=50)
    cf_model.fit(ratings_data=cf_train)

    # Evaluate
    all_exercise_ids = sorted(ratings_df['exercise_id'].unique())
    rmse_errors = []
    mae_errors = []
    hits_at_10 = 0
    ndcg_scores_10 = []
    reciprocal_ranks = []
    total_users = 0

    for _, test_row in test_df.iterrows():
        user_id = int(test_row['user_id'])
        test_exercise = int(test_row['exercise_id'])
        actual_rating = float(test_row['rating'])

        # Predict rating for held-out item
        predicted_rating = cf_model.predict(user_id, test_exercise)
        rmse_errors.append((predicted_rating - actual_rating) ** 2)
        mae_errors.append(abs(predicted_rating - actual_rating))

        # Rank all unseen exercises by predicted rating
        user_train = train_df[train_df['user_id'] == user_id]
        seen = set(user_train['exercise_id'].tolist())
        candidates = [eid for eid in all_exercise_ids if eid not in seen]

        if test_exercise not in candidates:
            candidates.append(test_exercise)

        scored = []
        for eid in candidates:
            pred = cf_model.predict(user_id, eid)
            scored.append((eid, pred))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [s[0] for s in scored]

        total_users += 1
        relevant = {test_exercise}

        # Hit Rate@10
        if test_exercise in ranked_ids[:10]:
            hits_at_10 += 1

        # NDCG@10
        ndcg_scores_10.append(ndcg_at_k(ranked_ids, relevant, 10))

        # MRR
        try:
            rank = ranked_ids.index(test_exercise) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    if total_users == 0:
        logger.warning("No users evaluated for CF")
        return None

    results = {
        'rmse': np.sqrt(np.mean(rmse_errors)),
        'mae': np.mean(mae_errors),
        'hit_rate_10': hits_at_10 / total_users,
        'ndcg_10': np.mean(ndcg_scores_10),
        'mrr': np.mean(reciprocal_ranks),
        'users_evaluated': total_users,
    }

    logger.info(f"CF RMSE: {results['rmse']:.4f}, Hit Rate@10: {results['hit_rate_10']:.4f}")
    return results


# ============================================================
#  4. HYBRID RECOMMENDER EVALUATION
# ============================================================

def evaluate_hybrid(ratings_df, exercises_df):
    """Evaluate hybrid recommender with Leave-One-Out."""
    logger.info("=" * 60)
    logger.info("EVALUATING HYBRID RECOMMENDER")
    logger.info("=" * 60)

    # Leave-One-Out split
    user_groups = ratings_df.groupby('user_id')
    eligible_users = [uid for uid, grp in user_groups if len(grp) >= 5]
    logger.info(f"Eligible users (>=5 ratings): {len(eligible_users)}")

    if len(eligible_users) == 0:
        logger.warning("No eligible users for hybrid evaluation")
        return None

    # Build train/test
    train_rows = []
    test_rows = []
    for user_id in eligible_users:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id].sort_values('timestamp')
        test_rows.append(user_ratings.iloc[-1])
        train_rows.append(user_ratings.iloc[:-1])

    ineligible_mask = ~ratings_df['user_id'].isin(eligible_users)
    ineligible_ratings = ratings_df[ineligible_mask]
    train_df = pd.concat(train_rows + [ineligible_ratings], ignore_index=True)
    test_df = pd.DataFrame(test_rows)

    logger.info(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")

    # Train collaborative filtering component (expects 'item_id' column)
    cf_train = train_df[['user_id', 'exercise_id', 'rating']].rename(
        columns={'exercise_id': 'item_id'}
    )
    cf_model = ProperCollaborativeFiltering(algorithm='SVD', n_factors=50)
    cf_model.fit(ratings_data=cf_train)

    # Build hybrid recommender
    hybrid = FinalHybridRecommender()
    hybrid.ratings_df = train_df.copy()
    hybrid.exercises_df = exercises_df.copy()

    # Build user-item matrix from train data
    hybrid.user_item_matrix = train_df.pivot_table(
        index='user_id', columns='exercise_id', values='rating', fill_value=np.nan
    )

    # Set the trained CF model
    hybrid.set_trained_cf_model({'cf_model': cf_model, 'best_method': 'SVD'})

    # Evaluate
    all_exercise_ids = set(exercises_df['exercise_id'].tolist())
    hits_at_10 = 0
    hits_at_15 = 0
    ndcg_scores_10 = []
    map_scores_10 = []
    precision_scores_10 = []
    all_recommended = set()
    total_users = 0

    for _, test_row in test_df.iterrows():
        user_id = int(test_row['user_id'])
        test_exercise = int(test_row['exercise_id'])

        # Get hybrid recommendations (excludes already-rated exercises)
        recs = hybrid.get_hybrid_recommendations(user_id, num_recs=15)
        ranked_ids = [r['exercise_id'] for r in recs]

        total_users += 1
        relevant = {test_exercise}
        all_recommended.update(ranked_ids[:15])

        # Hit Rate
        if test_exercise in ranked_ids[:10]:
            hits_at_10 += 1
        if test_exercise in ranked_ids[:15]:
            hits_at_15 += 1

        # NDCG@10
        ndcg_scores_10.append(ndcg_at_k(ranked_ids, relevant, 10))

        # MAP@10
        map_scores_10.append(average_precision(ranked_ids, relevant, 10))

        # Precision@10
        precision_scores_10.append(precision_at_k(ranked_ids, relevant, 10))

    if total_users == 0:
        logger.warning("No users evaluated for hybrid")
        return None

    coverage = len(all_recommended) / len(all_exercise_ids) if all_exercise_ids else 0

    # Diversity: average pairwise distance between recommended exercises
    diversity = 0.0
    if all_recommended:
        rec_exercises = exercises_df[exercises_df['exercise_id'].isin(all_recommended)]
        if len(rec_exercises) > 1:
            unique_muscles = rec_exercises['target_muscle_group'].nunique()
            unique_difficulties = rec_exercises['difficulty_level'].nunique()
            unique_equipment = rec_exercises['equipment_needed'].nunique()
            diversity = (unique_muscles + unique_difficulties + unique_equipment) / (3 * max(len(rec_exercises), 1))
            diversity = min(diversity, 1.0)

    results = {
        'hit_rate_10': hits_at_10 / total_users,
        'hit_rate_15': hits_at_15 / total_users,
        'ndcg_10': np.mean(ndcg_scores_10),
        'map_10': np.mean(map_scores_10),
        'precision_10': np.mean(precision_scores_10),
        'coverage': coverage,
        'diversity': diversity,
        'users_evaluated': total_users,
    }

    logger.info(f"Hybrid Hit Rate@10: {results['hit_rate_10']:.4f}")
    return results


# ============================================================
#  REPORT PRINTER
# ============================================================

def print_report(rf_results, cb_results, cf_results, hybrid_results):
    """Print structured evaluation report."""
    print("\n")
    print("=" * 60)
    print("     FITNEASE ML MODEL EVALUATION REPORT")
    print("=" * 60)

    # --- 1. Random Forest ---
    print("\n1. RANDOM FOREST (Workout Suitability Classifier)")
    print("-" * 50)
    if rf_results:
        print(f"   Train size:            {rf_results['train_size']}")
        print(f"   Test size:             {rf_results['test_size']}")
        print(f"   Accuracy:              {rf_results['accuracy'] * 100:.1f}%")
        print(f"   Precision (weighted):  {rf_results['precision'] * 100:.1f}%")
        print(f"   Recall (weighted):     {rf_results['recall'] * 100:.1f}%")
        print(f"   F1 Score (weighted):   {rf_results['f1'] * 100:.1f}%")
        print(f"\n   Per-class Report:")
        for line in rf_results['class_report'].split('\n'):
            if line.strip():
                print(f"   {line}")
        print(f"\n   Confusion Matrix:")
        cm = rf_results['confusion_matrix']
        active_labels = rf_results.get('class_names', [f'Class{i}' for i in range(cm.shape[0])])
        # Shorten labels for display
        short_labels = [l.replace('Suitable-', '') for l in active_labels]
        header = "   {:>14s}".format("") + "".join(f"  {l:>12s}" for l in short_labels)
        print(header)
        for i, row in enumerate(cm):
            print(f"   {short_labels[i]:>14s}" + "".join(f"  {v:>12d}" for v in row))
        print(f"\n   Top Feature Importances:")
        for feat, imp in rf_results['feature_importance'][:5]:
            print(f"     {feat:<40s} {imp:.4f}")
    else:
        print("   [SKIPPED - insufficient data]")

    # --- 2. Content-Based ---
    print("\n2. CONTENT-BASED RECOMMENDER")
    print("-" * 50)
    if cb_results:
        print(f"   Users evaluated:  {cb_results['users_evaluated']}")
        print(f"   Hit Rate@10:      {cb_results['hit_rate_10'] * 100:.1f}%")
        print(f"   Hit Rate@15:      {cb_results['hit_rate_15'] * 100:.1f}%")
        print(f"   NDCG@10:          {cb_results['ndcg_10']:.4f}")
        print(f"   NDCG@15:          {cb_results['ndcg_15']:.4f}")
        print(f"   Precision@10:     {cb_results['precision_10']:.4f}")
        print(f"   Coverage:         {cb_results['coverage'] * 100:.1f}%")
    else:
        print("   [SKIPPED - insufficient data]")

    # --- 3. Collaborative Filtering ---
    print("\n3. COLLABORATIVE FILTERING (SVD)")
    print("-" * 50)
    if cf_results:
        print(f"   Users evaluated:  {cf_results['users_evaluated']}")
        print(f"   RMSE:             {cf_results['rmse']:.4f}")
        print(f"   MAE:              {cf_results['mae']:.4f}")
        print(f"   Hit Rate@10:      {cf_results['hit_rate_10'] * 100:.1f}%")
        print(f"   NDCG@10:          {cf_results['ndcg_10']:.4f}")
        print(f"   MRR:              {cf_results['mrr']:.4f}")
    else:
        print("   [SKIPPED - insufficient data]")

    # --- 4. Hybrid ---
    print("\n4. HYBRID RECOMMENDER")
    print("-" * 50)
    if hybrid_results:
        print(f"   Users evaluated:  {hybrid_results['users_evaluated']}")
        print(f"   Hit Rate@10:      {hybrid_results['hit_rate_10'] * 100:.1f}%")
        print(f"   Hit Rate@15:      {hybrid_results['hit_rate_15'] * 100:.1f}%")
        print(f"   NDCG@10:          {hybrid_results['ndcg_10']:.4f}")
        print(f"   MAP@10:           {hybrid_results['map_10']:.4f}")
        print(f"   Precision@10:     {hybrid_results['precision_10']:.4f}")
        print(f"   Coverage:         {hybrid_results['coverage'] * 100:.1f}%")
        print(f"   Diversity:        {hybrid_results['diversity']:.4f}")
    else:
        print("   [SKIPPED - insufficient data]")

    print("\n" + "=" * 60)
    print("  Evaluation complete. All metrics computed from real data")
    print("  with proper train/test splits (no hardcoded values).")
    print("=" * 60)
    print()


# ============================================================
#  MAIN
# ============================================================

def main():
    logger.info("Starting FitNEase ML Model Evaluation...")

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

    # Run all evaluations
    rf_results = evaluate_random_forest(ratings_df, exercises_df, users_df)
    cb_results = evaluate_content_based(ratings_df, exercises_df)
    cf_results = evaluate_collaborative(ratings_df)
    hybrid_results = evaluate_hybrid(ratings_df, exercises_df)

    # Print final report
    print_report(rf_results, cb_results, cf_results, hybrid_results)


if __name__ == '__main__':
    main()
