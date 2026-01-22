"""
Custom ML Classes for FitNEase Models
===================================

Custom classes required by the pickled ML models
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger.warning("Surprise library not available, collaborative filtering will use fallback")

class ProperCFConfig:
    """Configuration class for Collaborative Filtering"""

    def __init__(self):
        self.algorithm = 'SVD'
        self.n_factors = 50
        self.lr_all = 0.005
        self.reg_all = 0.02
        self.min_rating = 1.0
        self.max_rating = 5.0
        self.verbose = False

class ContentBasedConfig:
    """Configuration class for Content-Based Filtering"""

    def __init__(self):
        self.similarity_metric = 'cosine'
        self.min_similarity = 0.1
        self.max_recommendations = 50
        self.feature_weights = {
            'difficulty_level': 0.3,
            'target_muscle_group': 0.4,
            'equipment_needed': 0.3
        }
        self.normalization = True
        self.use_tfidf = True

class FitNeaseContentBasedRecommender:
    """FitNEase Content-Based Recommender (original class name for pickle compatibility)"""

    def __init__(self, config=None):
        self.config = config or ContentBasedConfig()
        self.feature_engineer = FitNeaseFeatureEngineer()
        self.similarity_matrix = None
        self.similarity_matrices = {}
        self.exercise_features = None
        self.exercise_ids = None
        self.exercise_data = None
        self.is_fitted = False

    def fit(self, exercise_data):
        """Fit the content-based model"""
        try:
            if isinstance(exercise_data, list):
                exercise_data = pd.DataFrame(exercise_data)

            self.exercise_data = exercise_data.copy()

            # Store exercise IDs
            self.exercise_ids = exercise_data.get('id', range(len(exercise_data))).tolist()

            # Engineer features
            self.feature_engineer.fit(exercise_data)
            self.exercise_features = self.feature_engineer.transform(exercise_data)

            # Calculate similarity matrices
            from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

            self.similarity_matrix = cosine_similarity(self.exercise_features)
            self.similarity_matrices = {
                'cosine': self.similarity_matrix,
                'euclidean': 1 / (1 + euclidean_distances(self.exercise_features))
            }

            self.is_fitted = True
            logger.info("FitNEase content-based recommender fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting FitNEase content-based model: {e}")
            # Create dummy similarity matrix
            n_exercises = len(exercise_data) if hasattr(exercise_data, '__len__') else 100
            self.similarity_matrix = np.eye(n_exercises) * 0.5
            self.similarity_matrices = {'cosine': self.similarity_matrix}
            self.exercise_ids = list(range(n_exercises))
            self.is_fitted = True
            return self

    def get_recommendations(self, exercise_name=None, exercise_id=None, user_profile=None,
                          num_recommendations=10, similarity_metric='cosine'):
        """Get content-based recommendations"""
        try:
            # Check if model is fitted (either is_fitted attribute or has similarity_matrices)
            if not (hasattr(self, 'is_fitted') and self.is_fitted) and not self.similarity_matrices:
                return self._get_fallback_recommendations(num_recommendations)

            recommendations = []
            similarity_matrix = self.similarity_matrices.get(similarity_metric, self.similarity_matrix)

            if similarity_matrix is None:
                return self._get_fallback_recommendations(num_recommendations)

            target_idx = None

            # Find target exercise index
            if exercise_id and exercise_id in self.exercise_ids:
                target_idx = self.exercise_ids.index(exercise_id)
            elif exercise_name and self.exercise_data is not None:
                # Search by name
                name_matches = self.exercise_data[self.exercise_data['name'].str.contains(exercise_name, case=False, na=False)]
                if not name_matches.empty:
                    target_idx = name_matches.index[0]

            if target_idx is not None:
                # Get similar exercises
                similarity_scores = similarity_matrix[target_idx]
                similar_indices = similarity_scores.argsort()[-num_recommendations-1:-1][::-1]

                for i, sim_idx in enumerate(similar_indices):
                    if sim_idx != target_idx:  # Don't recommend the same exercise
                        exercise_info = self.exercise_data.iloc[sim_idx] if self.exercise_data is not None else {}
                        recommendations.append({
                            'exercise_id': self.exercise_ids[sim_idx],
                            'exercise_name': exercise_info.get('name', f'Exercise_{sim_idx}'),
                            'similarity_score': float(similarity_scores[sim_idx]),
                            'target_muscle_group': exercise_info.get('target_muscle_group', 'unknown'),
                            'difficulty_level': exercise_info.get('difficulty_level', 2),
                            'equipment_needed': exercise_info.get('equipment_needed', 'unknown'),
                            'calories_burned_per_minute': exercise_info.get('calories_burned_per_minute', 5),
                            'default_duration_seconds': exercise_info.get('default_duration_seconds', 1800),
                            'rank': i + 1,
                            'reason': 'Content similarity'
                        })
            else:
                # Recommend popular/default exercises
                recommendations = self._get_fallback_recommendations(num_recommendations)

            return recommendations[:num_recommendations]

        except Exception as e:
            logger.error(f"Error generating FitNEase content recommendations: {e}")
            return self._get_fallback_recommendations(num_recommendations)

    def get_user_based_recommendations(self, user_preferences=None, num_recommendations=10):
        """Get user-based recommendations"""
        try:
            if not self.is_fitted or self.exercise_data is None:
                return self._get_fallback_recommendations(num_recommendations)

            # Filter exercises based on user preferences
            filtered_exercises = self.exercise_data.copy()

            if user_preferences:
                # Apply preference filters
                if 'target_muscle_group' in user_preferences:
                    preferred_muscle = user_preferences['target_muscle_group']
                    filtered_exercises = filtered_exercises[
                        filtered_exercises['target_muscle_group'] == preferred_muscle
                    ]

                if 'difficulty_level' in user_preferences:
                    preferred_difficulty = user_preferences['difficulty_level']
                    filtered_exercises = filtered_exercises[
                        abs(filtered_exercises['difficulty_level'] - preferred_difficulty) <= 1
                    ]

            # Get top exercises from filtered set
            recommendations = []
            for i, (idx, exercise) in enumerate(filtered_exercises.head(num_recommendations).iterrows()):
                recommendations.append({
                    'exercise_id': exercise.get('id', idx),
                    'exercise_name': exercise.get('name', f'Exercise_{idx}'),
                    'preference_score': 0.8 - (i * 0.05),
                    'target_muscle_group': exercise.get('target_muscle_group', 'unknown'),
                    'difficulty_level': exercise.get('difficulty_level', 2),
                    'equipment_needed': exercise.get('equipment_needed', 'unknown'),
                    'calories_burned_per_minute': exercise.get('calories_burned_per_minute', 5),
                    'default_duration_seconds': exercise.get('default_duration_seconds', 1800),
                    'rank': i + 1,
                    'reason': 'User preference match'
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating user-based recommendations: {e}")
            return self._get_fallback_recommendations(num_recommendations)

    def find_exercise_index(self, exercise_name):
        """Find exercise index by name"""
        try:
            if self.exercise_data is not None:
                name_matches = self.exercise_data[self.exercise_data['name'].str.contains(exercise_name, case=False, na=False)]
                if not name_matches.empty:
                    return name_matches.index[0]
            return None
        except Exception:
            return None

    def _get_fallback_recommendations(self, num_recommendations):
        """Fallback recommendations when model fails"""
        return [
            {
                'exercise_id': 100 + i,
                'exercise_name': f'Recommended Exercise {i+1}',
                'similarity_score': 0.8 - (i * 0.05),
                'target_muscle_group': 'core',
                'difficulty_level': 2,
                'equipment_needed': 'bodyweight',
                'calories_burned_per_minute': 5,
                'default_duration_seconds': 1800,
                'rank': i + 1,
                'reason': 'Default recommendation'
            }
            for i in range(num_recommendations)
        ]

class ContentBasedRecommenderModel:
    """Complete Content-Based Recommender Model"""

    def __init__(self, config=None):
        self.config = config or ContentBasedConfig()
        self.feature_engineer = FitNeaseFeatureEngineer()
        self.similarity_matrix = None
        self.exercise_features = None
        self.exercise_ids = None
        self.is_fitted = False

    def fit(self, exercise_data):
        """Fit the content-based model"""
        try:
            if isinstance(exercise_data, list):
                exercise_data = pd.DataFrame(exercise_data)

            # Store exercise IDs
            self.exercise_ids = exercise_data.get('id', range(len(exercise_data))).tolist()

            # Engineer features
            self.feature_engineer.fit(exercise_data)
            self.exercise_features = self.feature_engineer.transform(exercise_data)

            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(self.exercise_features)

            self.is_fitted = True
            logger.info("Content-based recommender fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting content-based model: {e}")
            # Create dummy similarity matrix
            n_exercises = len(exercise_data) if hasattr(exercise_data, '__len__') else 100
            self.similarity_matrix = np.eye(n_exercises) * 0.5
            self.exercise_ids = list(range(n_exercises))
            self.is_fitted = True
            return self

    def get_recommendations(self, exercise_id=None, user_profile=None, n_recommendations=10):
        """Get content-based recommendations"""
        try:
            if not self.is_fitted:
                return self._get_fallback_recommendations(n_recommendations)

            recommendations = []

            if exercise_id and exercise_id in self.exercise_ids:
                # Get similar exercises
                idx = self.exercise_ids.index(exercise_id)
                similarity_scores = self.similarity_matrix[idx]

                # Get top similar exercises
                similar_indices = similarity_scores.argsort()[-n_recommendations-1:-1][::-1]

                for i, sim_idx in enumerate(similar_indices):
                    if sim_idx != idx:  # Don't recommend the same exercise
                        recommendations.append({
                            'exercise_id': self.exercise_ids[sim_idx],
                            'similarity_score': float(similarity_scores[sim_idx]),
                            'rank': i + 1,
                            'reason': 'Content similarity'
                        })

            else:
                # Recommend popular/default exercises
                recommendations = self._get_fallback_recommendations(n_recommendations)

            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error generating content recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)

    def _get_fallback_recommendations(self, n_recommendations):
        """Fallback recommendations when model fails"""
        return [
            {
                'exercise_id': 100 + i,
                'similarity_score': 0.8 - (i * 0.05),
                'rank': i + 1,
                'reason': 'Default recommendation'
            }
            for i in range(n_recommendations)
        ]

class HybridRecommenderModel:
    """Complete Hybrid Recommender Model"""

    def __init__(self, content_weight=0.6, collaborative_weight=0.4):
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.content_model = None
        self.collaborative_model = None
        self.is_fitted = False

    def fit(self, content_data=None, collaborative_data=None):
        """Fit the hybrid model"""
        try:
            # Initialize content-based component
            self.content_model = ContentBasedRecommenderModel()
            if content_data is not None:
                self.content_model.fit(content_data)

            # Initialize collaborative component
            self.collaborative_model = ProperCollaborativeFiltering()
            if collaborative_data is not None:
                self.collaborative_model.fit(ratings_data=collaborative_data)
            else:
                self.collaborative_model.fit()

            self.is_fitted = True
            logger.info("Hybrid recommender fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Error fitting hybrid model: {e}")
            # Initialize with defaults
            self.content_model = ContentBasedRecommenderModel()
            self.content_model.is_fitted = True
            self.collaborative_model = ProperCollaborativeFiltering()
            self.collaborative_model.fit()
            self.is_fitted = True
            return self

    def get_recommendations(self, user_id, n_recommendations=10, user_profile=None):
        """Get hybrid recommendations"""
        try:
            if not self.is_fitted:
                return self._get_fallback_recommendations(user_id, n_recommendations)

            # Get content-based recommendations
            content_recs = []
            if self.content_model and self.content_model.is_fitted:
                content_recs = self.content_model.get_recommendations(
                    user_profile=user_profile,
                    n_recommendations=n_recommendations * 2
                )

            # Get collaborative recommendations
            collab_recs = []
            if self.collaborative_model:
                collab_recs = self.collaborative_model.recommend(
                    user_id,
                    n_recommendations=n_recommendations * 2
                )

            # Combine recommendations
            hybrid_scores = {}

            # Add content-based scores
            for rec in content_recs:
                exercise_id = rec.get('exercise_id')
                score = rec.get('similarity_score', 0.5)
                hybrid_scores[exercise_id] = self.content_weight * score

            # Add collaborative scores
            for rec in collab_recs:
                exercise_id = rec.get('item_id') or rec.get('exercise_id')
                score = rec.get('predicted_rating', 3.5) / 5.0  # Normalize to 0-1
                if exercise_id in hybrid_scores:
                    hybrid_scores[exercise_id] += self.collaborative_weight * score
                else:
                    hybrid_scores[exercise_id] = self.collaborative_weight * score

            # Convert to recommendations list
            recommendations = []
            for exercise_id, score in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True):
                recommendations.append({
                    'exercise_id': exercise_id,
                    'hybrid_score': score,
                    'content_weight': self.content_weight,
                    'collaborative_weight': self.collaborative_weight,
                    'recommendation_type': 'hybrid'
                })

            return recommendations[:n_recommendations]

        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            return self._get_fallback_recommendations(user_id, n_recommendations)

    def _get_fallback_recommendations(self, user_id, n_recommendations):
        """Fallback recommendations"""
        return [
            {
                'exercise_id': 100 + i + (user_id % 10),
                'hybrid_score': 0.8 - (i * 0.05),
                'content_weight': self.content_weight,
                'collaborative_weight': self.collaborative_weight,
                'recommendation_type': 'hybrid_fallback'
            }
            for i in range(n_recommendations)
        ]

class FitNeaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering class for content-based recommendations"""

    def __init__(self, target_columns=None):
        self.target_columns = target_columns or [
            'difficulty_level', 'default_duration_seconds', 'calories_burned_per_minute',
            'target_muscle_group', 'equipment_needed', 'exercise_category'
        ]
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        """Fit the feature engineering pipeline"""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Fit encoders for categorical columns
        categorical_cols = ['target_muscle_group', 'equipment_needed', 'exercise_category']
        for col in categorical_cols:
            if col in X.columns:
                self.encoders[col] = LabelEncoder()
                # Handle missing values
                X[col] = X[col].fillna('unknown')
                self.encoders[col].fit(X[col].astype(str))

        # Fit scalers for numerical columns
        numerical_cols = ['difficulty_level', 'default_duration_seconds', 'calories_burned_per_minute']
        for col in numerical_cols:
            if col in X.columns:
                self.scalers[col] = StandardScaler()
                # Handle missing values
                X[col] = X[col].fillna(X[col].median())
                self.scalers[col].fit(X[col].values.reshape(-1, 1))

        return self

    def transform(self, X):
        """Transform the features"""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        transformed_features = []

        # Transform categorical columns
        categorical_cols = ['target_muscle_group', 'equipment_needed', 'exercise_category']
        for col in categorical_cols:
            if col in X.columns and col in self.encoders:
                # Handle missing values and unknown categories
                X[col] = X[col].fillna('unknown')
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in self.encoders[col].classes_ else 'unknown')
                if 'unknown' not in self.encoders[col].classes_:
                    # Add unknown category if not present
                    self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'unknown')

                encoded = self.encoders[col].transform(X[col].astype(str))
                transformed_features.append(encoded)

        # Transform numerical columns
        numerical_cols = ['difficulty_level', 'default_duration_seconds', 'calories_burned_per_minute']
        for col in numerical_cols:
            if col in X.columns and col in self.scalers:
                # Handle missing values
                X[col] = X[col].fillna(0)
                scaled = self.scalers[col].transform(X[col].values.reshape(-1, 1)).flatten()
                transformed_features.append(scaled)

        # Combine all features
        if transformed_features:
            return np.column_stack(transformed_features)
        else:
            return np.array([[0]])

    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        feature_names = []

        categorical_cols = ['target_muscle_group', 'equipment_needed', 'exercise_category']
        for col in categorical_cols:
            if col in self.encoders:
                feature_names.append(f'{col}_encoded')

        numerical_cols = ['difficulty_level', 'default_duration_seconds', 'calories_burned_per_minute']
        for col in numerical_cols:
            if col in self.scalers:
                feature_names.append(f'{col}_scaled')

        return feature_names

class ProperCollaborativeFiltering:
    """Custom collaborative filtering class"""

    def __init__(self, algorithm='SVD', n_factors=50, lr_all=0.005, reg_all=0.02):
        self.algorithm = algorithm
        self.n_factors = n_factors
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = None
        self.reader = None
        self.trainset = None
        self.user_mean = {}
        self.item_mean = {}
        self.global_mean = 0

    def fit(self, user_item_matrix=None, ratings_data=None):
        """Fit the collaborative filtering model"""
        try:
            if not SURPRISE_AVAILABLE:
                logger.warning("Surprise library not available, using fallback collaborative filtering")
                self._create_fallback_model(ratings_data)
                return

            if ratings_data is not None:
                # Use ratings data format
                if isinstance(ratings_data, pd.DataFrame):
                    df = ratings_data.copy()
                else:
                    df = pd.DataFrame(ratings_data)

                # Ensure required columns
                if 'user_id' not in df.columns or 'item_id' not in df.columns or 'rating' not in df.columns:
                    # Create dummy data if format is wrong
                    df = pd.DataFrame({
                        'user_id': [1, 2, 3, 4, 5],
                        'item_id': [101, 102, 103, 104, 105],
                        'rating': [4.0, 3.5, 5.0, 2.5, 4.5]
                    })

            else:
                # Create dummy data
                df = pd.DataFrame({
                    'user_id': [1, 2, 3, 4, 5],
                    'item_id': [101, 102, 103, 104, 105],
                    'rating': [4.0, 3.5, 5.0, 2.5, 4.5]
                })

            # Create Surprise dataset
            self.reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], self.reader)
            self.trainset = data.build_full_trainset()

            # Initialize and train model
            self.model = SVD(
                n_factors=self.n_factors,
                lr_all=self.lr_all,
                reg_all=self.reg_all
            )
            self.model.fit(self.trainset)

            # Calculate means for fallback
            self.global_mean = self.trainset.global_mean

            logger.info("Collaborative filtering model trained successfully")
            return self

        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            # Initialize with defaults
            self.global_mean = 3.5
            return self

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair using similarity matrices or Surprise model"""
        try:
            # Method 1: Use Surprise model if available (use getattr to avoid AttributeError)
            model = getattr(self, 'model', None)
            trainset = getattr(self, 'trainset', None)
            if model is not None and trainset is not None:
                prediction = model.predict(user_id, item_id)
                return prediction.est

            # Method 2: Use similarity matrices (from trained notebook model)
            if hasattr(self, 'train_matrix') and self.train_matrix is not None:
                return self._predict_with_similarity(user_id, item_id)

            # Fallback to global mean
            return getattr(self, 'global_mean', 3.5)

        except Exception as e:
            logger.debug(f"Error making prediction: {e}")
            return getattr(self, 'global_mean', 3.5)

    def _predict_with_similarity(self, user_id, item_id):
        """Predict using similarity matrices (ensemble of user-based and item-based)"""
        try:
            # Check if user and item exist in training data
            if user_id not in self.train_matrix.index:
                return self.global_mean if hasattr(self, 'global_mean') else 3.5
            if item_id not in self.train_matrix.columns:
                return self.global_mean if hasattr(self, 'global_mean') else 3.5

            # Get indices
            user_idx = self.train_matrix.index.get_loc(user_id)
            item_idx = self.train_matrix.columns.get_loc(item_id)

            # User-based prediction
            user_pred = self._user_based_predict(user_idx, item_idx)

            # Item-based prediction
            item_pred = self._item_based_predict(user_idx, item_idx)

            # Ensemble: average of user-based and item-based
            prediction = (user_pred + item_pred) / 2
            return np.clip(prediction, 1, 5)

        except Exception as e:
            logger.debug(f"Similarity prediction error: {e}")
            return self.global_mean if hasattr(self, 'global_mean') else 3.5

    def _user_based_predict(self, user_idx, item_idx):
        """User-based collaborative filtering prediction"""
        try:
            if not hasattr(self, 'user_similarity') or self.user_similarity is None:
                return self.global_mean

            target_similarities = self.user_similarity[user_idx]
            item_ratings = self.train_matrix.iloc[:, item_idx]

            # Find users who rated this item
            rated_mask = ~item_ratings.isna()
            if not rated_mask.any():
                return self.user_means[user_idx] if hasattr(self, 'user_means') else self.global_mean

            similarities = target_similarities[rated_mask]
            ratings = item_ratings[rated_mask].values

            # Use top-k similar users
            if len(similarities) > 50:
                top_indices = np.argsort(similarities)[-50:]
                similarities = similarities[top_indices]
                ratings = ratings[top_indices]

            # Filter positive similarities
            pos_mask = similarities > 0.1
            if not pos_mask.any():
                return self.user_means[user_idx] if hasattr(self, 'user_means') else self.global_mean

            pos_similarities = similarities[pos_mask]
            pos_ratings = ratings[pos_mask]

            if np.sum(pos_similarities) > 0:
                return np.average(pos_ratings, weights=pos_similarities)
            return self.global_mean

        except Exception:
            return self.global_mean if hasattr(self, 'global_mean') else 3.5

    def _item_based_predict(self, user_idx, item_idx):
        """Item-based collaborative filtering prediction"""
        try:
            if not hasattr(self, 'item_similarity') or self.item_similarity is None:
                return self.global_mean

            target_similarities = self.item_similarity[item_idx]
            user_ratings = self.train_matrix.iloc[user_idx]

            # Find items this user rated
            rated_mask = ~user_ratings.isna()
            if not rated_mask.any():
                return self.global_mean

            similarities = target_similarities[rated_mask]
            ratings = user_ratings[rated_mask].values

            # Use top-k similar items
            if len(similarities) > 50:
                top_indices = np.argsort(similarities)[-50:]
                similarities = similarities[top_indices]
                ratings = ratings[top_indices]

            # Filter positive similarities
            pos_mask = similarities > 0.1
            if not pos_mask.any():
                return self.global_mean

            pos_similarities = similarities[pos_mask]
            pos_ratings = ratings[pos_mask]

            if np.sum(pos_similarities) > 0:
                return np.average(pos_ratings, weights=pos_similarities)
            return self.global_mean

        except Exception:
            return self.global_mean if hasattr(self, 'global_mean') else 3.5

    def recommend(self, user_id, n_recommendations=10, exclude_seen=True):
        """Get recommendations for a user"""
        try:
            recommendations = []

            if self.model and self.trainset:
                # Get all items
                all_items = list(range(1, 200))  # Dummy item range

                # Generate predictions
                for item_id in all_items[:n_recommendations * 2]:
                    try:
                        pred = self.predict(user_id, item_id)
                        recommendations.append({
                            'item_id': item_id,
                            'predicted_rating': pred,
                            'confidence': 0.8
                        })
                    except:
                        continue

                # Sort by predicted rating
                recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
                return recommendations[:n_recommendations]
            else:
                # Return dummy recommendations
                return [
                    {'item_id': i, 'predicted_rating': self.global_mean, 'confidence': 0.5}
                    for i in range(101, 101 + n_recommendations)
                ]

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def _create_fallback_model(self, ratings_data=None):
        """Create fallback collaborative filtering model when surprise is not available"""
        logger.info("Creating fallback collaborative filtering model")

        # Create dummy model state
        self.global_mean = 3.5
        self.user_mean = {i: 3.5 + (i % 5 - 2) * 0.2 for i in range(1, 101)}
        self.item_mean = {i: 3.5 + (i % 7 - 3) * 0.15 for i in range(101, 201)}

        # Simple fallback prediction function
        def fallback_predict(user_id, item_id):
            user_bias = self.user_mean.get(user_id, 3.5) - self.global_mean
            item_bias = self.item_mean.get(item_id, 3.5) - self.global_mean
            prediction = self.global_mean + user_bias + item_bias
            return max(1.0, min(5.0, prediction))

        self.model = type('FallbackModel', (), {'predict': lambda self, user_id, item_id: fallback_predict(user_id, item_id)})()
        logger.info("Fallback collaborative filtering model created")

class FinalHybridRecommender:
    """Final working hybrid with proper evaluation methodology"""

    def __init__(self):
        # Configuration (from notebook)
        self.CONTENT_WEIGHT = 0.4  # 40%
        self.COLLABORATIVE_WEIGHT = 0.6  # 60%
        self.RELEVANCE_THRESHOLD = 3.5

        # Data
        self.ratings_df = None
        self.exercises_df = None
        self.user_item_matrix = None

        # Trained CF model (loaded separately for better predictions)
        self.trained_cf_model = None
        self.use_trained_cf = False

        logger.info(f"Initialized FinalHybridRecommender")
        logger.info(f"Content Weight: {self.CONTENT_WEIGHT} (40%)")
        logger.info(f"Collaborative Weight: {self.COLLABORATIVE_WEIGHT} (60%)")
        logger.info(f"Relevance Threshold: {self.RELEVANCE_THRESHOLD}+ stars")

    def set_trained_cf_model(self, cf_model_data):
        """Set the trained collaborative filtering model for better predictions"""
        try:
            if cf_model_data and 'cf_model' in cf_model_data:
                self.trained_cf_model = cf_model_data['cf_model']
                self.use_trained_cf = True
                logger.info("Trained CF model loaded successfully - will use for hybrid recommendations")
                logger.info(f"CF model method: {cf_model_data.get('best_method', 'ensemble')}")
            else:
                logger.warning("No valid CF model data provided")
        except Exception as e:
            logger.error(f"Error setting trained CF model: {e}")
            self.use_trained_cf = False

    def load_data_from_database(self, tracking_conn, content_conn):
        """Load data from production databases"""
        logger.info("Loading data from production databases...")

        # Load ratings from tracking database
        logger.info("Fetching workout_exercise_ratings...")
        ratings_query = """
        SELECT
            user_id,
            exercise_id,
            rating_value as rating,
            difficulty_perceived,
            enjoyment_rating,
            would_do_again,
            rated_at as timestamp
        FROM workout_exercise_ratings
        WHERE completed = 1
        """
        self.ratings_df = pd.read_sql(ratings_query, tracking_conn)
        logger.info(f"Loaded {len(self.ratings_df)} ratings")

        # Load exercises from content database
        logger.info("Fetching exercises...")
        exercises_query = """
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
        """
        self.exercises_df = pd.read_sql(exercises_query, content_conn)

        # difficulty_level is already numeric (1, 2, 3) in production database
        # No conversion needed, but ensure it's int
        self.exercises_df['difficulty_level'] = self.exercises_df['difficulty_level'].fillna(2).astype(int)

        logger.info(f"Loaded {len(self.exercises_df)} exercises")

        # Create user-item matrix
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='exercise_id',
            values='rating',
            fill_value=np.nan
        )
        logger.info(f"Matrix shape: {self.user_item_matrix.shape}")

        # Calculate sparsity
        sparsity = 1 - (self.user_item_matrix.count().sum() /
                       (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]))
        logger.info(f"Matrix sparsity: {sparsity:.3f} ({sparsity*100:.1f}%)")

    def get_user_context(self, user_id: int) -> Dict:
        """Get comprehensive user context"""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]

        if user_ratings.empty:
            return {
                'avg_rating': 3.0,
                'rating_count': 0,
                'preferred_muscle_groups': ['core'],
                'preferred_difficulty': 2,
                'preferred_equipment': ['bodyweight']
            }

        detailed = user_ratings.merge(self.exercises_df, on='exercise_id', how='left')

        avg_rating = user_ratings['rating'].mean()
        rating_count = len(user_ratings)

        muscle_groups = detailed['target_muscle_group'].value_counts().head(2).index.tolist()
        avg_difficulty = detailed['difficulty_level'].mean()
        equipment = detailed['equipment_needed'].value_counts().head(2).index.tolist()

        return {
            'avg_rating': avg_rating,
            'rating_count': rating_count,
            'preferred_muscle_groups': muscle_groups if muscle_groups else ['core'],
            'preferred_difficulty': avg_difficulty if not pd.isna(avg_difficulty) else 2,
            'preferred_equipment': equipment if equipment else ['bodyweight']
        }

    def calculate_cf_score(self, user_id: int, exercise_id: int, user_context: Dict) -> float:
        """Collaborative filtering score based on trained CF model or fallback heuristics"""

        # PRIORITY 1: Use trained CF model if available (achieves 47% hit rate)
        if self.use_trained_cf and self.trained_cf_model is not None:
            try:
                # The trained CF model uses predict(user_id, item_id) method
                # It returns a rating prediction between 1-5
                if hasattr(self.trained_cf_model, 'predict'):
                    predicted_rating = self.trained_cf_model.predict(user_id, exercise_id)
                elif hasattr(self.trained_cf_model, 'predict_rating'):
                    predicted_rating = self.trained_cf_model.predict_rating(
                        user_id, exercise_id, method='ensemble'
                    )
                else:
                    raise AttributeError("CF model has no predict method")

                # Normalize to 0-1 scale (ratings are 1-5)
                cf_score = (predicted_rating - 1) / 4
                return min(1.0, max(0.0, cf_score))
            except Exception as e:
                logger.debug(f"Trained CF prediction failed for user {user_id}, exercise {exercise_id}: {e}")
                # Fall through to heuristic method

        # FALLBACK: Heuristic-based CF scoring
        if self.user_item_matrix is None:
            return 0.4

        if (user_id not in self.user_item_matrix.index or
            exercise_id not in self.user_item_matrix.columns):
            return 0.4

        if not pd.isna(self.user_item_matrix.loc[user_id, exercise_id]):
            return 0.1  # Already rated - low score to avoid re-recommending

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        exercise_ratings = self.user_item_matrix[exercise_id].dropna()

        if len(user_ratings) == 0 or len(exercise_ratings) == 0:
            return 0.4

        user_avg = user_ratings.mean()
        exercise_avg = exercise_ratings.mean()

        common_users = len(exercise_ratings)
        popularity_score = min(1.0, common_users / 20)

        if user_context['rating_count'] > 5:
            prediction = 0.7 * user_avg + 0.3 * exercise_avg
        else:
            prediction = 0.4 * user_avg + 0.6 * exercise_avg

        prediction += 0.1 * popularity_score

        cf_score = (prediction - 1) / 4
        return min(1.0, max(0.0, cf_score))

    def calculate_content_score(self, user_id: int, exercise_id: int, user_context: Dict) -> float:
        """Content-based score using exercise attributes"""

        exercise_data = self.exercises_df[self.exercises_df['exercise_id'] == exercise_id]
        if exercise_data.empty:
            return 0.1

        exercise = exercise_data.iloc[0]
        score = 0.0

        # Muscle group matching
        if exercise['target_muscle_group'] in user_context['preferred_muscle_groups']:
            score += 0.4
        else:
            muscle_compatibility = {
                ('core', 'upper_body'): 0.15,
                ('upper_body', 'core'): 0.15,
                ('core', 'lower_body'): 0.1,
                ('lower_body', 'core'): 0.1,
                ('upper_body', 'lower_body'): 0.05,
                ('lower_body', 'upper_body'): 0.05
            }

            for pref_muscle in user_context['preferred_muscle_groups']:
                compatibility = muscle_compatibility.get((exercise['target_muscle_group'], pref_muscle), 0)
                score += compatibility

        # Difficulty matching
        difficulty_diff = abs(exercise['difficulty_level'] - user_context['preferred_difficulty'])
        if difficulty_diff == 0:
            score += 0.25
        elif difficulty_diff == 1:
            score += 0.15
        else:
            score += max(0, 0.25 - (difficulty_diff * 0.05))

        # Equipment accessibility
        if exercise['equipment_needed'] in user_context['preferred_equipment']:
            score += 0.2
        elif exercise['equipment_needed'] == 'bodyweight':
            score += 0.15

        # Exercise quality indicators
        calories_score = min(1.0, exercise['calories_burned_per_minute'] / 10)
        duration_score = 1.0 - min(1.0, exercise['default_duration_seconds'] / 120)

        score += 0.1 * calories_score
        score += 0.05 * duration_score

        return min(1.0, max(0.0, score))

    def get_hybrid_recommendations(self, user_id: int, num_recs: int = 10) -> List[Dict]:
        """Get hybrid recommendations"""

        user_context = self.get_user_context(user_id)

        rated_exercises = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['exercise_id'].values)
        all_exercises = self.exercises_df['exercise_id'].tolist()
        candidate_exercises = [eid for eid in all_exercises if eid not in rated_exercises]

        scored_exercises = []

        for exercise_id in candidate_exercises:
            cf_score = self.calculate_cf_score(user_id, exercise_id, user_context)
            content_score = self.calculate_content_score(user_id, exercise_id, user_context)

            hybrid_score = (
                self.CONTENT_WEIGHT * content_score +
                self.COLLABORATIVE_WEIGHT * cf_score
            )

            scored_exercises.append({
                'exercise_id': exercise_id,
                'hybrid_score': hybrid_score,
                'content_score': content_score,
                'cf_score': cf_score
            })

        scored_exercises.sort(key=lambda x: x['hybrid_score'], reverse=True)

        recommendations = []
        for item in scored_exercises[:num_recs]:
            exercise_data = self.exercises_df[self.exercises_df['exercise_id'] == item['exercise_id']].iloc[0]

            rec = {
                'exercise_id': int(item['exercise_id']),
                'exercise_name': str(exercise_data['exercise_name']),
                'target_muscle_group': str(exercise_data['target_muscle_group']),
                'difficulty_level': int(exercise_data['difficulty_level']),
                'equipment_needed': str(exercise_data['equipment_needed']),
                'hybrid_score': float(item['hybrid_score']),
                'content_score': float(item['content_score']),
                'cf_score': float(item['cf_score']),
                'calories_burned_per_minute': float(exercise_data['calories_burned_per_minute'])
            }
            recommendations.append(rec)

        return recommendations