"""
Data Preprocessing Utilities
============================

Utility functions for data preprocessing and feature engineering
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing utility class"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}

    def process_behavioral_data(self, data: Dict) -> Dict:
        """Process user behavioral data"""
        try:
            processed_data = {
                'user_id': data.get('user_id'),
                'processed_at': pd.Timestamp.now(),
                'original_data': data
            }

            # Extract workout patterns
            if 'workouts' in data:
                processed_data['workout_patterns'] = self._analyze_workout_patterns(data['workouts'])

            # Extract engagement patterns
            if 'interactions' in data:
                processed_data['engagement_patterns'] = self._analyze_engagement_patterns(data['interactions'])

            # Calculate behavioral scores
            processed_data['behavioral_scores'] = self._calculate_behavioral_scores(data)

            return processed_data

        except Exception as e:
            logger.error(f"Error processing behavioral data: {e}")
            return {}

    def prepare_content_features(self, exercises: List[Dict]) -> pd.DataFrame:
        """Prepare content-based features for exercises"""
        try:
            df = pd.DataFrame(exercises)

            # Handle missing values
            df = self._handle_missing_values(df)

            # Encode categorical features
            categorical_columns = ['target_muscle_group', 'equipment_needed', 'exercise_category']
            for col in categorical_columns:
                if col in df.columns:
                    df[f'{col}_encoded'] = self._encode_categorical(df[col], col)

            # Scale numerical features
            numerical_columns = ['difficulty_level', 'default_duration_seconds', 'calories_burned_per_minute']
            for col in numerical_columns:
                if col in df.columns:
                    df[f'{col}_scaled'] = self._scale_numerical(df[col], col)

            # Create composite features
            df = self._create_composite_features(df)

            return df

        except Exception as e:
            logger.error(f"Error preparing content features: {e}")
            return pd.DataFrame()

    def prepare_user_features(self, user_data: Dict) -> Dict:
        """Prepare user features for ML models"""
        try:
            features = {}

            # Basic demographics
            features['age'] = user_data.get('age', 30)
            features['gender_encoded'] = 1 if user_data.get('gender', 'male') == 'male' else 0
            features['bmi'] = user_data.get('bmi', 23.0)

            # Fitness features
            fitness_level = user_data.get('fitness_level', 'intermediate')
            features['fitness_level_encoded'] = self._encode_fitness_level(fitness_level)
            features['experience_months'] = user_data.get('experience_months', 12)
            features['weekly_frequency'] = user_data.get('weekly_workout_frequency', 3)

            # Preference features
            preferences = user_data.get('preferences', {})
            features['preferred_duration'] = preferences.get('preferred_workout_duration', 30)
            features['preferred_difficulty'] = preferences.get('difficulty_preference', 2)

            # One-hot encode equipment preference
            equipment_pref = preferences.get('equipment_preference', 'bodyweight')
            features['prefers_bodyweight'] = 1 if equipment_pref == 'bodyweight' else 0
            features['prefers_dumbbells'] = 1 if equipment_pref == 'dumbbells' else 0
            features['prefers_barbell'] = 1 if equipment_pref == 'barbell' else 0

            # One-hot encode muscle group preference
            muscle_pref = preferences.get('preferred_muscle_groups', 'core')
            features['prefers_core'] = 1 if muscle_pref == 'core' else 0
            features['prefers_upper'] = 1 if muscle_pref == 'upper_body' else 0
            features['prefers_lower'] = 1 if muscle_pref == 'lower_body' else 0

            return features

        except Exception as e:
            logger.error(f"Error preparing user features: {e}")
            return {}

    def prepare_workout_features(self, workout_data: Dict) -> Dict:
        """Prepare workout features for ML models"""
        try:
            features = {}

            # Basic workout features
            features['difficulty_level'] = workout_data.get('difficulty_level', 2)
            features['duration_minutes'] = workout_data.get('estimated_duration_minutes', 30)
            features['exercise_count'] = workout_data.get('exercise_count', 8)
            features['calories_estimate'] = workout_data.get('calories_burned_estimate', 200)

            # Equipment encoding
            equipment = workout_data.get('equipment_needed', 'bodyweight')
            features['equipment_bodyweight'] = 1 if equipment == 'bodyweight' else 0
            features['equipment_dumbbells'] = 1 if equipment == 'dumbbells' else 0
            features['equipment_barbell'] = 1 if equipment == 'barbell' else 0
            features['equipment_kettlebell'] = 1 if equipment == 'kettlebell' else 0

            # Muscle group encoding
            muscle_groups = workout_data.get('target_muscle_groups', 'core')
            if isinstance(muscle_groups, str):
                muscle_groups = [muscle_groups]

            features['targets_core'] = 1 if 'core' in muscle_groups else 0
            features['targets_upper'] = 1 if 'upper_body' in muscle_groups else 0
            features['targets_lower'] = 1 if 'lower_body' in muscle_groups else 0

            # Derived features
            features['intensity_score'] = self._calculate_intensity_score(workout_data)
            features['complexity_score'] = self._calculate_complexity_score(workout_data)

            return features

        except Exception as e:
            logger.error(f"Error preparing workout features: {e}")
            return {}

    def normalize_user_ratings(self, ratings: List[Dict]) -> List[Dict]:
        """Normalize user rating data"""
        try:
            if not ratings:
                return []

            df = pd.DataFrame(ratings)

            # Handle missing ratings
            df['rating'] = df['rating'].fillna(df['rating'].mean())

            # Normalize ratings to 0-1 scale
            if 'rating' in df.columns:
                min_rating = df['rating'].min()
                max_rating = df['rating'].max()
                if max_rating > min_rating:
                    df['normalized_rating'] = (df['rating'] - min_rating) / (max_rating - min_rating)
                else:
                    df['normalized_rating'] = 0.5

            # Add rating statistics
            user_stats = df.groupby('user_id')['rating'].agg(['mean', 'std', 'count']).reset_index()
            user_stats.columns = ['user_id', 'avg_rating', 'rating_std', 'rating_count']

            # Merge back with original data
            df = df.merge(user_stats, on='user_id', how='left')

            return df.to_dict('records')

        except Exception as e:
            logger.error(f"Error normalizing user ratings: {e}")
            return ratings

    def create_user_item_matrix(self, ratings: List[Dict]) -> pd.DataFrame:
        """Create user-item interaction matrix"""
        try:
            df = pd.DataFrame(ratings)

            if df.empty:
                return pd.DataFrame()

            # Create pivot table
            matrix = df.pivot_table(
                index='user_id',
                columns='exercise_id',
                values='rating',
                fill_value=0
            )

            return matrix

        except Exception as e:
            logger.error(f"Error creating user-item matrix: {e}")
            return pd.DataFrame()

    def _analyze_workout_patterns(self, workouts: List[Dict]) -> Dict:
        """Analyze workout patterns from user data"""
        if not workouts:
            return {}

        df = pd.DataFrame(workouts)

        patterns = {
            'total_workouts': len(workouts),
            'completion_rate': df.get('completed', pd.Series([True])).mean(),
            'avg_duration': df.get('duration_minutes', pd.Series([30])).mean(),
            'preferred_time': self._find_preferred_time(df),
            'preferred_days': self._find_preferred_days(df),
            'difficulty_distribution': df.get('difficulty_level', pd.Series([2])).value_counts().to_dict()
        }

        return patterns

    def _analyze_engagement_patterns(self, interactions: List[Dict]) -> Dict:
        """Analyze user engagement patterns"""
        if not interactions:
            return {}

        df = pd.DataFrame(interactions)

        patterns = {
            'total_interactions': len(interactions),
            'interaction_types': df.get('type', pd.Series(['view'])).value_counts().to_dict(),
            'engagement_score': self._calculate_engagement_score(df),
            'session_frequency': self._calculate_session_frequency(df)
        }

        return patterns

    def _calculate_behavioral_scores(self, data: Dict) -> Dict:
        """Calculate behavioral scoring metrics"""
        scores = {
            'consistency_score': 0.5,
            'engagement_score': 0.5,
            'progress_score': 0.5,
            'preference_alignment': 0.5
        }

        # Calculate consistency based on workout frequency
        if 'workouts' in data:
            workouts = data['workouts']
            if workouts:
                # Simple consistency calculation
                scores['consistency_score'] = min(len(workouts) / 20, 1.0)

        # Calculate engagement based on interactions
        if 'interactions' in data:
            interactions = data['interactions']
            if interactions:
                scores['engagement_score'] = min(len(interactions) / 50, 1.0)

        return scores

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset"""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')

        return df

    def _encode_categorical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Encode categorical variables"""
        if column_name not in self.encoders:
            self.encoders[column_name] = LabelEncoder()
            return pd.Series(self.encoders[column_name].fit_transform(series))
        else:
            return pd.Series(self.encoders[column_name].transform(series))

    def _scale_numerical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Scale numerical variables"""
        if column_name not in self.scalers:
            self.scalers[column_name] = StandardScaler()
            return pd.Series(self.scalers[column_name].fit_transform(series.values.reshape(-1, 1)).flatten())
        else:
            return pd.Series(self.scalers[column_name].transform(series.values.reshape(-1, 1)).flatten())

    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features"""
        # Intensity feature
        if 'difficulty_level' in df.columns and 'calories_burned_per_minute' in df.columns:
            df['intensity_composite'] = df['difficulty_level'] * df['calories_burned_per_minute']

        # Duration efficiency feature
        if 'default_duration_seconds' in df.columns and 'calories_burned_per_minute' in df.columns:
            df['efficiency_score'] = df['calories_burned_per_minute'] / (df['default_duration_seconds'] / 60)

        return df

    def _encode_fitness_level(self, fitness_level: str) -> int:
        """Encode fitness level to numeric"""
        mapping = {
            'beginner': 1,
            'intermediate': 2,
            'expert': 3,
            'advanced': 3
        }
        return mapping.get(fitness_level.lower(), 2)

    def _calculate_intensity_score(self, workout_data: Dict) -> float:
        """Calculate workout intensity score"""
        difficulty = workout_data.get('difficulty_level', 2)
        duration = workout_data.get('estimated_duration_minutes', 30)
        calories = workout_data.get('calories_burned_estimate', 200)

        # Simple intensity calculation
        intensity = (difficulty * calories) / max(duration, 1)
        return min(intensity / 50, 1.0)  # Normalize to 0-1

    def _calculate_complexity_score(self, workout_data: Dict) -> float:
        """Calculate workout complexity score"""
        exercise_count = workout_data.get('exercise_count', 8)
        difficulty = workout_data.get('difficulty_level', 2)

        # Simple complexity calculation
        complexity = (exercise_count * difficulty) / 20
        return min(complexity, 1.0)  # Normalize to 0-1

    def _find_preferred_time(self, df: pd.DataFrame) -> str:
        """Find user's preferred workout time"""
        if 'time_of_day' in df.columns:
            return df['time_of_day'].mode().iloc[0] if not df['time_of_day'].mode().empty else 'evening'
        return 'evening'

    def _find_preferred_days(self, df: pd.DataFrame) -> List[str]:
        """Find user's preferred workout days"""
        if 'day_of_week' in df.columns:
            return df['day_of_week'].value_counts().head(3).index.tolist()
        return ['monday', 'wednesday', 'friday']

    def _calculate_engagement_score(self, df: pd.DataFrame) -> float:
        """Calculate engagement score from interactions"""
        if df.empty:
            return 0.5

        # Simple engagement calculation based on interaction variety and frequency
        unique_types = df['type'].nunique() if 'type' in df.columns else 1
        total_interactions = len(df)

        score = min((unique_types * total_interactions) / 50, 1.0)
        return score

    def _calculate_session_frequency(self, df: pd.DataFrame) -> float:
        """Calculate session frequency"""
        if df.empty or 'timestamp' not in df.columns:
            return 0.5

        # Convert timestamps and calculate frequency
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        days_span = (df['timestamp'].max() - df['timestamp'].min()).days

        if days_span > 0:
            frequency = len(df) / days_span
            return min(frequency, 1.0)

        return 0.5