"""
API Routes
==========

Main API routes for all ML operations
"""

from flask import Blueprint, request, jsonify
import logging

from controllers.content_based_controller import ContentBasedController
from controllers.collaborative_controller import CollaborativeController
from controllers.hybrid_controller import HybridController
from controllers.random_forest_controller import RandomForestController
from controllers.model_management_controller import ModelManagementController

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Initialize controllers
content_controller = ContentBasedController()
collaborative_controller = CollaborativeController()
hybrid_controller = HybridController()
rf_controller = RandomForestController()
management_controller = ModelManagementController()

# ============================================================================
# CORE ML MODEL ENDPOINTS (4 Models Required)
# ============================================================================

# Model 1: Content-Based Filtering Model
@api_bp.route('/content-similarity', methods=['POST'])
def content_similarity():
    """Calculate exercise similarity scores"""
    try:
        data = request.get_json() or {}
        result = content_controller.calculate_similarity(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Content similarity endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/content-recommendations/<int:user_id>', methods=['GET'])
def content_recommendations(user_id):
    """Content-based recommendations for user"""
    try:
        # Extract Bearer token from request
        auth_token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            auth_token = auth_header[7:]  # Remove 'Bearer ' prefix
            logger.info(f"ML service received Bearer token: {auth_token[:20]}..." if auth_token else "No token received")

        data = request.args.to_dict()
        data['num_recommendations'] = int(data.get('num_recommendations', 10))
        data['auth_token'] = auth_token  # Pass token to controller

        result = content_controller.get_user_recommendations(user_id, data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Content recommendations endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/exercise-similarity', methods=['POST'])
def exercise_similarity():
    """Calculate similarity between two exercises"""
    try:
        data = request.get_json() or {}
        result = content_controller.get_exercise_similarity(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Exercise similarity endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/similar-exercises', methods=['POST'])
def similar_exercises():
    """Get exercises similar to given exercise above threshold"""
    try:
        data = request.get_json() or {}
        result = content_controller.get_similar_exercises(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Similar exercises endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

# Model 2: Collaborative Filtering Model
@api_bp.route('/collaborative-scores', methods=['POST'])
def collaborative_scores():
    """Calculate user similarity scores"""
    try:
        data = request.get_json() or {}
        result = collaborative_controller.calculate_user_similarity(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Collaborative scores endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/collaborative-recommendations/<int:user_id>', methods=['GET'])
def collaborative_recommendations(user_id):
    """Collaborative recommendations for user"""
    try:
        data = request.args.to_dict()
        data['num_recommendations'] = int(data.get('num_recommendations', 10))

        result = collaborative_controller.get_user_recommendations(user_id, data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Collaborative recommendations endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

# Model 3: Hybrid Filtering Model (Main Recommendation Engine)
@api_bp.route('/recommendations/<int:user_id>', methods=['GET'])
def hybrid_recommendations(user_id):
    """Main hybrid recommendations endpoint (PRIMARY ENDPOINT)"""
    try:
        # Extract Bearer token from request
        auth_token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            auth_token = auth_header[7:]  # Remove 'Bearer ' prefix

        data = request.args.to_dict()
        # Convert numeric parameters
        if 'num_recommendations' in data:
            data['num_recommendations'] = int(data['num_recommendations'])
        if 'content_weight' in data:
            data['content_weight'] = float(data['content_weight'])
        if 'collaborative_weight' in data:
            data['collaborative_weight'] = float(data['collaborative_weight'])

        data['auth_token'] = auth_token  # Pass token to controller

        result = hybrid_controller.get_recommendations(user_id, data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Hybrid recommendations endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/hybrid-scores/<int:user_id>', methods=['GET'])
def hybrid_scores(user_id):
    """Detailed hybrid scoring breakdown"""
    try:
        result = hybrid_controller.get_detailed_scores(user_id)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Hybrid scores endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/hybrid-content-based', methods=['POST'])
def hybrid_content_based():
    """Pure content-based recommendations via hybrid model"""
    try:
        data = request.get_json() or {}
        result = hybrid_controller.get_content_based_recommendations(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Hybrid content-based endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/hybrid-collaborative', methods=['POST'])
def hybrid_collaborative():
    """Pure collaborative recommendations via hybrid model"""
    try:
        data = request.get_json() or {}
        result = hybrid_controller.get_collaborative_recommendations(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Hybrid collaborative endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/hybrid-similarity', methods=['POST'])
def hybrid_similarity():
    """Exercise similarity via hybrid model"""
    try:
        data = request.get_json() or {}
        result = hybrid_controller.calculate_exercise_similarity(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Hybrid similarity endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

# Model 4: Random Forest Classifier
@api_bp.route('/predict-difficulty', methods=['POST'])
def predict_difficulty():
    """Workout difficulty prediction"""
    try:
        data = request.get_json() or {}
        result = rf_controller.predict_difficulty(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Difficulty prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict-completion', methods=['POST'])
def predict_completion():
    """Workout completion probability prediction"""
    try:
        data = request.get_json() or {}
        result = rf_controller.predict_completion(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Completion prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict-suitability', methods=['POST'])
def predict_suitability():
    """Overall workout suitability prediction"""
    try:
        data = request.get_json() or {}
        result = rf_controller.predict_suitability(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Suitability prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/batch-prediction', methods=['POST'])
def batch_prediction():
    """Batch prediction for multiple workout scenarios"""
    try:
        data = request.get_json() or {}
        result = rf_controller.batch_predict(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MODEL MANAGEMENT & TRAINING ENDPOINTS
# ============================================================================

@api_bp.route('/train-model', methods=['POST'])
def train_model():
    """Retrain all ML models with new data"""
    try:
        data = request.get_json() or {}
        result = management_controller.train_models(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Model training endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/model-health', methods=['GET'])
def model_health():
    """ML model performance metrics for all 4 models"""
    try:
        result = management_controller.get_model_health()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Model health endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/model-evaluation', methods=['POST'])
def model_evaluation():
    """Evaluate model accuracy and performance"""
    try:
        data = request.get_json() or {}
        result = management_controller.evaluate_models(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Model evaluation endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/model-status', methods=['GET'])
def model_status():
    """Check deployment status of all models"""
    try:
        result = management_controller.get_model_status()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Model status endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/reload-models', methods=['POST'])
def reload_models():
    """Reload all ML models"""
    try:
        result = management_controller.reload_models()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Model reload endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# BEHAVIORAL ANALYSIS ENDPOINTS
# ============================================================================

@api_bp.route('/behavioral-data', methods=['POST'])
def behavioral_data():
    """Receive user behavior updates (async)"""
    try:
        data = request.get_json() or {}
        result = management_controller.process_behavioral_data(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Behavioral data endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/user-patterns/<int:user_id>', methods=['GET'])
def user_patterns(user_id):
    """Get user behavioral patterns"""
    try:
        result = management_controller.get_user_patterns(user_id)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"User patterns endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/update-patterns', methods=['POST'])
def update_patterns():
    """Update behavioral patterns (batch)"""
    try:
        data = request.get_json() or {}
        result = hybrid_controller.update_user_patterns(data)

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Update patterns endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MODEL-SPECIFIC HEALTH CHECKS
# ============================================================================

@api_bp.route('/content-based/health', methods=['GET'])
def content_based_health():
    """Content-based model health check"""
    try:
        result = content_controller.get_model_health()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Content-based health endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/collaborative/health', methods=['GET'])
def collaborative_health():
    """Collaborative model health check"""
    try:
        result = collaborative_controller.get_model_health()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Collaborative health endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/hybrid/health', methods=['GET'])
def hybrid_health():
    """Hybrid model health check"""
    try:
        result = hybrid_controller.get_model_health()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Hybrid health endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/random-forest/health', methods=['GET'])
def random_forest_health():
    """Random Forest model health check"""
    try:
        result = rf_controller.get_model_health()

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Random Forest health endpoint error: {e}")
        return jsonify({'error': str(e)}), 500