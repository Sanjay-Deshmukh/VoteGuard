"""
Configuration settings for the Voting Anomaly Detection System
"""
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'voteguard-secret-key-2024'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///voteguard.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Email Configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'voteguard@example.com')
    
    # Simulation Defaults
    DEFAULT_CANDIDATES = 4
    DEFAULT_ANOMALY_PROBABILITY = 0.15
    DEFAULT_SIMULATION_DURATION = 60
    
    # ML Model Defaults
    ISO_CONTAMINATION = 0.1
    ISO_RANDOM_STATE = 42
    LOF_NEIGHBORS = 10
    OCSVM_KERNEL = 'rbf'
    OCSVM_GAMMA = 'auto'
    
    # RL Agent Defaults
    RL_STATE_SIZE = 10
    RL_LEARNING_RATE = 0.001
    RL_GAMMA = 0.99
    RL_EPSILON_START = 1.0
    RL_EPSILON_END = 0.01
    RL_EPSILON_DECAY = 0.995


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
