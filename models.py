"""
Database models for VoteGuard
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='viewer')  # admin, analyst, viewer
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    sessions = db.relationship('SimulationSession', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class SimulationSession(db.Model):
    """Store simulation sessions for historical comparison"""
    __tablename__ = 'simulation_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    name = db.Column(db.String(100), default='Unnamed Session')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Configuration
    num_candidates = db.Column(db.Integer, default=4)
    anomaly_probability = db.Column(db.Float, default=0.15)
    threshold = db.Column(db.Float, default=0.5)
    duration = db.Column(db.Integer, default=60)
    
    # Results
    total_samples = db.Column(db.Integer, default=0)
    total_anomalies = db.Column(db.Integer, default=0)
    winner = db.Column(db.String(50))
    winner_votes = db.Column(db.Integer, default=0)
    
    # Full data stored as JSON
    data_json = db.Column(db.Text)
    
    # ML Metrics
    rl_final_epsilon = db.Column(db.Float)
    rl_avg_reward = db.Column(db.Float)
    rl_accuracy = db.Column(db.Float)
    
    def set_data(self, data):
        """Store simulation data as JSON"""
        self.data_json = json.dumps(data)
    
    def get_data(self):
        """Retrieve simulation data from JSON"""
        if self.data_json:
            return json.loads(self.data_json)
        return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'num_candidates': self.num_candidates,
            'anomaly_probability': self.anomaly_probability,
            'threshold': self.threshold,
            'duration': self.duration,
            'total_samples': self.total_samples,
            'total_anomalies': self.total_anomalies,
            'winner': self.winner,
            'winner_votes': self.winner_votes,
            'rl_final_epsilon': self.rl_final_epsilon,
            'rl_avg_reward': self.rl_avg_reward,
            'rl_accuracy': self.rl_accuracy
        }


class ModelConfig(db.Model):
    """Store ML model configurations"""
    __tablename__ = 'model_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=False)
    
    # Isolation Forest
    iso_contamination = db.Column(db.Float, default=0.1)
    iso_n_estimators = db.Column(db.Integer, default=100)
    iso_random_state = db.Column(db.Integer, default=42)
    
    # LOF
    lof_n_neighbors = db.Column(db.Integer, default=10)
    lof_contamination = db.Column(db.Float, default=0.1)
    
    # One-Class SVM
    ocsvm_kernel = db.Column(db.String(20), default='rbf')
    ocsvm_gamma = db.Column(db.String(20), default='auto')
    ocsvm_nu = db.Column(db.Float, default=0.1)
    
    # RL Agent
    rl_learning_rate = db.Column(db.Float, default=0.001)
    rl_gamma = db.Column(db.Float, default=0.99)
    rl_epsilon_start = db.Column(db.Float, default=1.0)
    rl_epsilon_end = db.Column(db.Float, default=0.01)
    rl_epsilon_decay = db.Column(db.Float, default=0.995)
    rl_batch_size = db.Column(db.Integer, default=32)
    rl_memory_size = db.Column(db.Integer, default=10000)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'is_active': self.is_active,
            'isolation_forest': {
                'contamination': self.iso_contamination,
                'n_estimators': self.iso_n_estimators,
                'random_state': self.iso_random_state
            },
            'lof': {
                'n_neighbors': self.lof_n_neighbors,
                'contamination': self.lof_contamination
            },
            'ocsvm': {
                'kernel': self.ocsvm_kernel,
                'gamma': self.ocsvm_gamma,
                'nu': self.ocsvm_nu
            },
            'rl_agent': {
                'learning_rate': self.rl_learning_rate,
                'gamma': self.rl_gamma,
                'epsilon_start': self.rl_epsilon_start,
                'epsilon_end': self.rl_epsilon_end,
                'epsilon_decay': self.rl_epsilon_decay,
                'batch_size': self.rl_batch_size,
                'memory_size': self.rl_memory_size
            }
        }


class EmailSubscription(db.Model):
    """Email subscription for reports"""
    __tablename__ = 'email_subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Subscription preferences
    on_anomaly = db.Column(db.Boolean, default=True)
    on_completion = db.Column(db.Boolean, default=True)
    daily_digest = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'is_active': self.is_active,
            'on_anomaly': self.on_anomaly,
            'on_completion': self.on_completion,
            'daily_digest': self.daily_digest
        }
