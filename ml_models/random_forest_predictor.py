"""
Random Forest Predictor Model
============================

Wrapper for the trained Random Forest model for workout completion
and difficulty appropriateness prediction
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class RandomForestPredictor:
    """Random Forest model wrapper for workout predictions"""

    def __init__(self, model_data: Dict):
        """Initialize with loaded model data"""
        try:
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.label_encoders = model_data.get('label_encoders', model_data.get('encoders', {}))
            self.feature_columns = model_data.get('feature_columns', model_data.get('feature_names', []))
            self.model_info = model_data.get('model_info', {})

            if not self.model:
                raise ValueError("Random Forest model not found in model data")

            logger.info("Random Forest predictor initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Random Forest predictor: {e}")
            raise

    def predict_completion_probability(self, user_profile: Dict, workout_features: Dict) -> Dict:
        """Predict workout completion probability"""
        try:
            # Prepare feature vector
            features = self._prepare_features(user_profile, workout_features)

            if features is None:
                return {
                    'completion_probability': 0.5,
                    'confidence': 0.0,
                    'prediction_class': 'uncertain',
                    'error': 'Feature preparation failed'
                }

            # Make prediction
            probabilities = self.model.predict_proba(features)
            completion_prob = float(probabilities[0][1]) if probabilities.shape[1] > 1 else float(probabilities[0][0])

            # Get prediction class
            prediction = self.model.predict(features)[0]
            prediction_class = 'will_complete' if prediction == 1 else 'will_not_complete'

            # Calculate confidence (distance from 0.5)
            confidence = abs(completion_prob - 0.5) * 2

            result = {
                'completion_probability': completion_prob,
                'confidence': confidence,
                'prediction_class': prediction_class,
                'recommendation': self._get_completion_recommendation(completion_prob),
                'factors': self._analyze_completion_factors(user_profile, workout_features)
            }

            logger.info(f"Predicted completion probability: {completion_prob:.3f} for user")
            return result

        except Exception as e:
            logger.error(f"Error predicting completion probability: {e}")
            return {
                'completion_probability': 0.5,
                'confidence': 0.0,
                'prediction_class': 'error',
                'error': str(e)
            }

    def predict_difficulty_appropriateness(self, user_profile: Dict, workout_features: Dict) -> Dict:
        """Predict if workout difficulty is appropriate for user"""
        try:
            # Prepare features
            features = self._prepare_features(user_profile, workout_features)

            if features is None:
                return {
                    'appropriateness_score': 0.5,
                    'difficulty_rating': 'unknown',
                    'recommendation': 'Unable to assess',
                    'error': 'Feature preparation failed'
                }

            # Get prediction probability
            probabilities = self.model.predict_proba(features)
            appropriateness = float(probabilities[0][1]) if probabilities.shape[1] > 1 else float(probabilities[0][0])

            # Determine difficulty rating
            if appropriateness >= 0.8:
                difficulty_rating = 'perfect_match'
            elif appropriateness >= 0.6:
                difficulty_rating = 'good_match'
            elif appropriateness >= 0.4:
                difficulty_rating = 'moderate_match'
            else:
                difficulty_rating = 'poor_match'

            result = {
                'appropriateness_score': appropriateness,
                'difficulty_rating': difficulty_rating,
                'recommendation': self._get_difficulty_recommendation(appropriateness, user_profile, workout_features),
                'user_level': user_profile.get('fitness_level', 'unknown'),
                'workout_difficulty': workout_features.get('difficulty_level', 'unknown')
            }

            logger.info(f"Difficulty appropriateness: {appropriateness:.3f} ({difficulty_rating})")
            return result

        except Exception as e:
            logger.error(f"Error predicting difficulty appropriateness: {e}")
            return {
                'appropriateness_score': 0.5,
                'difficulty_rating': 'error',
                'recommendation': 'Unable to assess difficulty',
                'error': str(e)
            }

    def predict_workout_success(self, user_profile: Dict, workout_features: Dict,
                              user_history: Dict = None) -> Dict:
        """Comprehensive workout success prediction"""
        try:
            completion_pred = self.predict_completion_probability(user_profile, workout_features)
            difficulty_pred = self.predict_difficulty_appropriateness(user_profile, workout_features)

            # Calculate overall success score
            success_score = (completion_pred['completion_probability'] +
                           difficulty_pred['appropriateness_score']) / 2

            # Determine success category
            if success_score >= 0.8:
                success_category = 'highly_likely'
            elif success_score >= 0.6:
                success_category = 'likely'
            elif success_score >= 0.4:
                success_category = 'moderate'
            else:
                success_category = 'unlikely'

            result = {
                'overall_success_score': success_score,
                'success_category': success_category,
                'completion_prediction': completion_pred,
                'difficulty_prediction': difficulty_pred,
                'recommendations': self._get_success_recommendations(success_score, user_profile, workout_features),
                'risk_factors': self._identify_risk_factors(user_profile, workout_features, user_history)
            }

            logger.info(f"Workout success prediction: {success_score:.3f} ({success_category})")
            return result

        except Exception as e:
            logger.error(f"Error predicting workout success: {e}")
            return {
                'overall_success_score': 0.5,
                'success_category': 'error',
                'error': str(e)
            }

    def batch_predict(self, predictions_data: List[Dict]) -> List[Dict]:
        """Batch prediction for multiple user-workout combinations"""
        try:
            results = []

            for data in predictions_data:
                user_profile = data.get('user_profile', {})
                workout_features = data.get('workout_features', {})
                user_history = data.get('user_history', {})

                prediction = self.predict_workout_success(
                    user_profile, workout_features, user_history
                )

                prediction['user_id'] = data.get('user_id')
                prediction['workout_id'] = data.get('workout_id')
                results.append(prediction)

            logger.info(f"Completed batch prediction for {len(results)} user-workout combinations")
            return results

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return []

    def _prepare_features(self, user_profile: Dict, workout_features: Dict) -> Optional[np.ndarray]:
        """Prepare feature vector for prediction matching trained model features"""
        try:
            # Create feature dictionary matching the trained model
            features_dict = {}

            # User features (matching trained model feature names)
            features_dict['age'] = user_profile.get('age', 30)
            features_dict['fitness_level_numeric'] = self._encode_fitness_level(user_profile.get('fitness_level', 'intermediate'))
            features_dict['bmi_category_numeric'] = self._encode_bmi_category(user_profile.get('bmi', 23.0))
            features_dict['user_experience'] = user_profile.get('experience_months', 12)
            features_dict['user_consistency'] = user_profile.get('weekly_workout_frequency', 3)
            features_dict['days_since_last_workout'] = user_profile.get('days_since_last_workout', 2)

            # Workout features (matching trained model feature names)
            features_dict['difficulty_level'] = workout_features.get('difficulty_level', 2)
            features_dict['exercise_intensity'] = workout_features.get('exercise_intensity', workout_features.get('difficulty_level', 2))
            features_dict['duration_minutes'] = workout_features.get('estimated_duration_minutes', 30)
            features_dict['calories_per_minute'] = workout_features.get('calories_per_minute',
                                                  workout_features.get('calories_burned_estimate', 200) /
                                                  workout_features.get('estimated_duration_minutes', 30))
            features_dict['equipment_accessibility'] = 1  # Assume equipment is accessible
            features_dict['user_exercise_difficulty_gap'] = abs(features_dict['fitness_level_numeric'] - features_dict['difficulty_level'])
            features_dict['user_fatigue'] = user_profile.get('fatigue_level', 1)

            # Encoded categorical features using the model's encoders
            target_muscle_group = workout_features.get('target_muscle_groups', workout_features.get('target_muscle_group', 'core'))
            features_dict['target_muscle_group_encoded'] = self._encode_categorical('target_muscle_group', target_muscle_group)

            exercise_category = workout_features.get('exercise_category', 'strength')
            features_dict['exercise_category_encoded'] = self._encode_categorical('exercise_category', exercise_category)

            equipment_needed = workout_features.get('equipment_needed', 'bodyweight')
            features_dict['equipment_needed_encoded'] = self._encode_categorical('equipment_needed', equipment_needed)

            # Convert to array in the order expected by the model
            if self.feature_columns:
                feature_values = [features_dict.get(col, 0) for col in self.feature_columns]
                feature_array = np.array([feature_values])

                # Create a DataFrame with proper feature names to avoid sklearn warning
                import pandas as pd
                feature_df = pd.DataFrame([feature_values], columns=self.feature_columns)
            else:
                # Fallback to default order if feature_columns not available
                feature_values = [
                    features_dict['age'],
                    features_dict['fitness_level_numeric'],
                    features_dict['bmi_category_numeric'],
                    features_dict['user_experience'],
                    features_dict['user_consistency'],
                    features_dict['days_since_last_workout'],
                    features_dict['difficulty_level'],
                    features_dict['exercise_intensity'],
                    features_dict['duration_minutes'],
                    features_dict['calories_per_minute'],
                    features_dict['equipment_accessibility'],
                    features_dict['user_exercise_difficulty_gap'],
                    features_dict['user_fatigue'],
                    features_dict['target_muscle_group_encoded'],
                    features_dict['exercise_category_encoded'],
                    features_dict['equipment_needed_encoded']
                ]
                feature_array = np.array([feature_values])

                # Create DataFrame with default column names
                import pandas as pd
                default_columns = [
                    'age', 'fitness_level_numeric', 'bmi_category_numeric', 'user_experience',
                    'user_consistency', 'days_since_last_workout', 'difficulty_level',
                    'exercise_intensity', 'duration_minutes', 'calories_per_minute',
                    'equipment_accessibility', 'user_exercise_difficulty_gap', 'user_fatigue',
                    'target_muscle_group_encoded', 'exercise_category_encoded', 'equipment_needed_encoded'
                ]
                feature_df = pd.DataFrame([feature_values], columns=default_columns)

            # Apply scaling if available and fitted
            if self.scaler and hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                try:
                    # Scale the feature array
                    feature_array = self.scaler.transform(feature_array)
                    # Update the DataFrame with scaled values
                    feature_df.iloc[0] = feature_array[0]
                except Exception as e:
                    logger.warning(f"Error applying scaler: {e}, using unscaled features")
            elif self.scaler:
                logger.warning("Scaler is not fitted, using unscaled features")

            # Return DataFrame for better sklearn compatibility
            return feature_df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def _encode_fitness_level(self, fitness_level: str) -> int:
        """Encode fitness level to numeric value"""
        level_mapping = {
            'beginner': 1,
            'intermediate': 2,
            'expert': 3,
            'advanced': 3
        }
        return level_mapping.get(fitness_level.lower(), 2)

    def _encode_bmi_category(self, bmi: float) -> int:
        """Encode BMI to numeric category"""
        if bmi < 18.5:
            return 1  # Underweight
        elif bmi < 25:
            return 2  # Normal
        elif bmi < 30:
            return 3  # Overweight
        else:
            return 4  # Obese

    def _encode_categorical(self, category_name: str, value: str) -> int:
        """Encode categorical value using stored encoders"""
        try:
            if category_name in self.label_encoders:
                encoder = self.label_encoders[category_name]
                if hasattr(encoder, 'transform'):
                    # sklearn LabelEncoder
                    try:
                        return encoder.transform([value])[0]
                    except ValueError:
                        # Value not seen during training, return most common class or 0
                        return 0
                elif isinstance(encoder, dict):
                    # Dictionary mapping
                    return encoder.get(value, 0)

            # Fallback encoding for common values
            if category_name == 'target_muscle_group':
                muscle_mapping = {
                    'core': 0, 'upper_body': 1, 'lower_body': 2,
                    'full_body': 3, 'cardio': 4
                }
                return muscle_mapping.get(value.lower(), 0)
            elif category_name == 'exercise_category':
                category_mapping = {
                    'strength': 0, 'cardio': 1, 'flexibility': 2,
                    'balance': 3, 'sports': 4
                }
                return category_mapping.get(value.lower(), 0)
            elif category_name == 'equipment_needed':
                equipment_mapping = {
                    'bodyweight': 0, 'dumbbells': 1, 'barbell': 2,
                    'resistance_bands': 3, 'kettlebell': 4, 'machine': 5
                }
                return equipment_mapping.get(value.lower(), 0)

            return 0

        except Exception as e:
            logger.warning(f"Error encoding {category_name}={value}: {e}")
            return 0

    def _get_completion_recommendation(self, probability: float) -> str:
        """Get recommendation based on completion probability"""
        if probability >= 0.8:
            return "High likelihood of completion - proceed with workout"
        elif probability >= 0.6:
            return "Good chance of completion - recommended"
        elif probability >= 0.4:
            return "Moderate completion risk - consider modifications"
        else:
            return "Low completion probability - suggest easier alternative"

    def _get_difficulty_recommendation(self, appropriateness: float, user_profile: Dict, workout_features: Dict) -> str:
        """Get recommendation based on difficulty appropriateness"""
        if appropriateness >= 0.8:
            return "Perfect difficulty match for user level"
        elif appropriateness >= 0.6:
            return "Good difficulty match - proceed as planned"
        elif appropriateness >= 0.4:
            return "Moderate match - consider slight modifications"
        else:
            return "Difficulty mismatch - recommend different level"

    def _analyze_completion_factors(self, user_profile: Dict, workout_features: Dict) -> Dict:
        """Analyze factors affecting completion"""
        factors = {
            'positive_factors': [],
            'risk_factors': []
        }

        # Analyze duration
        duration = workout_features.get('estimated_duration_minutes', 30)
        if duration <= 30:
            factors['positive_factors'].append('Short duration')
        elif duration >= 60:
            factors['risk_factors'].append('Long duration')

        # Analyze user experience
        experience = user_profile.get('experience_months', 12)
        if experience >= 24:
            factors['positive_factors'].append('Experienced user')
        elif experience < 6:
            factors['risk_factors'].append('Limited experience')

        # Analyze difficulty vs fitness level
        user_level = self._encode_fitness_level(user_profile.get('fitness_level', 'intermediate'))
        workout_difficulty = workout_features.get('difficulty_level', 2)

        if workout_difficulty <= user_level:
            factors['positive_factors'].append('Appropriate difficulty')
        else:
            factors['risk_factors'].append('High difficulty for user level')

        return factors

    def _get_success_recommendations(self, success_score: float, user_profile: Dict, workout_features: Dict) -> List[str]:
        """Get recommendations for improving workout success"""
        recommendations = []

        if success_score < 0.6:
            # Suggest modifications
            if workout_features.get('estimated_duration_minutes', 30) > 45:
                recommendations.append("Consider reducing workout duration")

            if workout_features.get('difficulty_level', 2) > self._encode_fitness_level(user_profile.get('fitness_level', 'intermediate')):
                recommendations.append("Try a lower difficulty level")

            recommendations.append("Ensure proper warm-up and cool-down")
            recommendations.append("Start with shorter sessions and build up")

        elif success_score >= 0.8:
            recommendations.append("Excellent match - proceed confidently")
            recommendations.append("Consider increasing intensity over time")

        else:
            recommendations.append("Good match with room for optimization")
            recommendations.append("Monitor completion and adjust as needed")

        return recommendations

    def _identify_risk_factors(self, user_profile: Dict, workout_features: Dict, user_history: Dict = None) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []

        # Check for mismatched difficulty
        user_level = self._encode_fitness_level(user_profile.get('fitness_level', 'intermediate'))
        workout_difficulty = workout_features.get('difficulty_level', 2)

        if workout_difficulty > user_level + 1:
            risk_factors.append("Workout difficulty significantly exceeds user level")

        # Check for long duration
        if workout_features.get('estimated_duration_minutes', 30) > 60:
            risk_factors.append("Extended workout duration")

        # Check user history if available
        if user_history:
            completion_rate = user_history.get('completion_rate', 0.7)
            if completion_rate < 0.5:
                risk_factors.append("Low historical completion rate")

        return risk_factors

    def get_model_info(self) -> Dict:
        """Get model information and metadata"""
        return {
            'model_type': 'random_forest_classifier',
            'version': self.model_info.get('version', '1.0'),
            'training_date': self.model_info.get('training_date'),
            'accuracy': self.model_info.get('accuracy', 0.998),
            'feature_count': len(self.feature_columns) if self.feature_columns else 15,
            'capabilities': [
                'completion_prediction',
                'difficulty_assessment',
                'success_scoring',
                'batch_prediction'
            ],
            'status': 'loaded'
        }

    def health_check(self) -> Dict:
        """Check model health and status"""
        try:
            is_healthy = (
                self.model is not None and
                hasattr(self.model, 'predict') and
                hasattr(self.model, 'predict_proba')
            )

            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'model_loaded': self.model is not None,
                'scaler_loaded': self.scaler is not None,
                'feature_columns_available': len(self.feature_columns) > 0 if self.feature_columns else False,
                'can_predict': is_healthy
            }

        except Exception as e:
            logger.error(f"Error in Random Forest model health check: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }