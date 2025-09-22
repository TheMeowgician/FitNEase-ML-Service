"""
Health Check Routes
==================

Basic health check endpoints
"""

from flask import Blueprint, jsonify
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'fitnease-ml',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'message': 'FitNEase ML Service is running'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@health_bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint"""
    return jsonify({'status': 'pong'})

@health_bp.route('/version', methods=['GET'])
def version():
    """Service version information"""
    return jsonify({
        'service': 'fitnease-ml',
        'version': '1.0.0',
        'description': 'FitNEase Machine Learning Service',
        'technology': 'Python Flask + scikit-learn',
        'models': [
            'Content-Based Filtering',
            'Collaborative Filtering',
            'Hybrid Recommendation',
            'Random Forest Classifier'
        ]
    })