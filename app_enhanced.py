"""
Enhanced Flask Dashboard for Voting Anomaly Detection with RL
Full-featured version with WebSockets, Authentication, Swagger API, and more
"""
from flask import Flask, render_template, jsonify, request, send_file, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from flask_swagger_ui import get_swaggerui_blueprint
import pandas as pd
import numpy as np
import time
import datetime
import joblib
import os
import random
import threading
import json
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import torch

from config import config
from models import db, User, SimulationSession, ModelConfig, EmailSubscription
from rl_anomaly_detector import RLAnomalyDetector

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config['development'])

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

# Swagger UI
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "VoteGuard API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Global configuration
simulation_config = {
    'num_candidates': 4,
    'anomaly_probability': 0.15,
    'duration': 60,
    'overt_anomaly_prob': 0.10,
    'stealth_anomaly_prob': 0.05
}

# ML Model configuration
ml_config = {
    'iso_contamination': 0.1,
    'iso_n_estimators': 100,
    'lof_n_neighbors': 10,
    'ocsvm_kernel': 'rbf',
    'ocsvm_gamma': 'auto'
}

def get_candidates():
    return [f"cand{i}" for i in range(1, simulation_config['num_candidates'] + 1)]

def get_diff_cols():
    return [f"diff_{i}" for i in range(1, simulation_config['num_candidates'] + 1)]

def get_ratio_cols():
    return [f"ratio_{i}" for i in range(1, simulation_config['num_candidates'] + 1)]


# Global data storage
data_store = {
    'df': None,
    'running': False,
    'start_time': None,
    'step': 0,
    'threshold': 0.5,
    'predictions_history': deque(maxlen=100),
    'heatmap_data': [],
    'trend_data': {'ma_5': [], 'ma_10': [], 'ema': []}
}

# Initialize models
iso, lof, ocsvm = None, None, None
rl_agent = None

def reset_dataframe():
    """Reset the dataframe with current configuration"""
    cols = ["time"] + get_candidates() + [
        "iso_score", "lof_score", "ocsvm_score", "anomaly_score", "alert", "tampering_type",
        "total_votes", "vote_diff_sum"
    ] + get_diff_cols() + get_ratio_cols() + [
        "rl_prediction", "rl_reward", "rl_confidence"
    ]
    data_store['df'] = pd.DataFrame(columns=cols)
    data_store['heatmap_data'] = []
    data_store['trend_data'] = {'ma_5': [], 'ma_10': [], 'ema': []}


def get_fpga_vote_data():
    """Simulates vote counts based on configuration."""
    r = random.random()
    num_cands = simulation_config['num_candidates']
    overt_prob = simulation_config['overt_anomaly_prob']
    stealth_prob = simulation_config['stealth_anomaly_prob']
    
    if r < overt_prob:
        # Inject Overt Anomaly (High Spike)
        data = {}
        spike_cand = random.randint(0, num_cands - 1)
        for i in range(num_cands):
            if i == spike_cand:
                data[f"cand{i+1}"] = random.randint(80, 100)
            else:
                data[f"cand{i+1}"] = random.randint(0, 20)
        return data
    elif overt_prob <= r < overt_prob + stealth_prob:
        # Inject Stealth Anomaly (Sudden Drop/Stall)
        return {f"cand{i+1}": random.randint(0, 5) for i in range(num_cands)}
    
    # Normal distribution
    return {f"cand{i+1}": random.randint(40, 60) for i in range(num_cands)}


def extract_features(df):
    candidates = get_candidates()
    diff_cols = get_diff_cols()
    
    df["total_votes"] = df[candidates].sum(axis=1)
    for i in range(1, simulation_config['num_candidates'] + 1):
        df[f"diff_{i}"] = df[f"cand{i}"].diff().fillna(0).copy()
        df[f"ratio_{i}"] = (df[f"cand{i}"]/df["total_votes"]).fillna(0).copy()
    df["vote_diff_sum"] = df[diff_cols].sum(axis=1)
    
    # Calculate trend indicators
    if len(df) >= 5:
        df['ma_5'] = df['anomaly_score'].rolling(window=5, min_periods=1).mean()
    if len(df) >= 10:
        df['ma_10'] = df['anomaly_score'].rolling(window=10, min_periods=1).mean()
    
    return df.fillna(0)


def classify_tampering(row):
    """Classify tampering type based on features."""
    if pd.isna(row.get("alert")) or row["alert"] == "Normal":
        return "Not Applicable"
    
    num_cands = simulation_config['num_candidates']
    diffs = [row[f"diff_{i}"] for i in range(1, num_cands + 1)]
    ratios = [row[f"ratio_{i}"] for i in range(1, num_cands + 1)]
    max_diff = max(diffs)
    min_diff = min(diffs)
    total = row["total_votes"]
    
    if max_diff > 30 and total > 200:
        return "Sudden Spike"
    elif min_diff < -30:
        return "Sudden Drop/Trough"
    elif total > 350:
        return "Total Vote Overload"
    elif total < 10 and row["time"] > 5:
        return "Vote System Stall"
    elif max(ratios) > 0.85:
        return "Dominant Ratio Injection"
    elif all(0 < d < 10 for d in diffs) and max(abs(d) for d in diffs) > 1:
        return "Gradual Drift"
    elif len(set(diffs)) < 2 and max(abs(d) for d in diffs) > 1:
        return "Duplicate Pattern"
    return "Unclassified Anomaly"


def load_models():
    """Load or train models."""
    global iso, lof, ocsvm
    
    # Always retrain with current configuration
    dummy = pd.DataFrame(np.random.rand(50, 10))
    iso = IsolationForest(
        contamination=ml_config['iso_contamination'],
        n_estimators=ml_config.get('iso_n_estimators', 100),
        random_state=42
    )
    lof = LocalOutlierFactor(n_neighbors=ml_config['lof_n_neighbors'], novelty=True)
    ocsvm = OneClassSVM(kernel=ml_config['ocsvm_kernel'], gamma=ml_config['ocsvm_gamma'])
    
    iso.fit(dummy)
    lof.fit(dummy)
    ocsvm.fit(dummy)


def predict_anomaly(df):
    """Predict future anomaly likelihood based on trends."""
    if len(df) < 10:
        return {'probability': 0.0, 'confidence': 0.0, 'trend': 'stable'}
    
    recent_scores = df['anomaly_score'].tail(10).values
    
    # Calculate trend
    if len(recent_scores) >= 5:
        ma_short = np.mean(recent_scores[-5:])
        ma_long = np.mean(recent_scores)
        
        if ma_short > ma_long * 1.2:
            trend = 'increasing'
            probability = min(0.9, ma_short + 0.1)
        elif ma_short < ma_long * 0.8:
            trend = 'decreasing'
            probability = max(0.1, ma_short - 0.1)
        else:
            trend = 'stable'
            probability = ma_short
        
        # Confidence based on consistency
        std = np.std(recent_scores)
        confidence = max(0.1, 1.0 - std)
        
        return {
            'probability': float(probability),
            'confidence': float(confidence),
            'trend': trend,
            'next_5_prediction': [float(probability + (0.05 * i if trend == 'increasing' else -0.05 * i if trend == 'decreasing' else 0)) for i in range(5)]
        }
    
    return {'probability': 0.0, 'confidence': 0.0, 'trend': 'stable'}


def detect_anomaly(df):
    """Detect anomalies using ensemble + RL."""
    global rl_agent
    
    num_cands = simulation_config['num_candidates']
    feature_list = ["total_votes", "vote_diff_sum"] + get_diff_cols() + get_ratio_cols()
    
    # Ensure we have enough features
    while len(feature_list) < 10:
        feature_list.append(feature_list[-1])
    feature_list = feature_list[:10]
    
    features = df[feature_list].iloc[1:].copy()
    
    if features.empty:
        return df
    
    iso_pred = np.where(iso.predict(features) == -1, 1, 0)
    lof_pred = np.where(lof.predict(features) == -1, 1, 0)
    ocsvm_pred = np.where(ocsvm.predict(features) == -1, 1, 0)
    
    for col in ["iso_score", "lof_score", "ocsvm_score", "anomaly_score", "alert", "tampering_type",
                "rl_prediction", "rl_reward", "rl_confidence"]:
        if col not in df.columns:
            if 'score' in col or 'confidence' in col:
                df[col] = 0.0
            elif 'prediction' in col:
                df[col] = 0
            elif 'reward' in col:
                df[col] = 0.0
            else:
                df[col] = 'N/A'
    
    df.iloc[1:, df.columns.get_loc("iso_score")] = iso_pred
    df.iloc[1:, df.columns.get_loc("lof_score")] = lof_pred
    df.iloc[1:, df.columns.get_loc("ocsvm_score")] = ocsvm_pred
    
    ensemble_score = (iso_pred + lof_pred + ocsvm_pred) / 3
    ensemble_pred = np.where(ensemble_score > 0.5, 1, 0)
    df.iloc[1:, df.columns.get_loc("anomaly_score")] = ensemble_score
    
    # RL Agent Integration
    if rl_agent is not None and len(features) > 0:
        latest_idx = len(features) - 1
        latest_row = features.iloc[latest_idx]
        
        state = rl_agent.get_state(latest_row.values)
        ensemble_pred_val = ensemble_pred[latest_idx] if latest_idx < len(ensemble_pred) else 0
        ensemble_score_val = ensemble_score[latest_idx] if latest_idx < len(ensemble_score) else 0
        
        rl_action, reward, loss = rl_agent.learn_online(state, ensemble_pred_val, ensemble_score_val)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(rl_agent.device)
        with torch.no_grad():
            q_values = rl_agent.q_network(state_tensor)
            confidence = torch.softmax(q_values, dim=1)[0][rl_action].item()
        
        df_idx = len(df) - 1
        if df_idx >= 1:
            df.loc[df_idx, "rl_prediction"] = rl_action
            df.loc[df_idx, "rl_reward"] = reward
            df.loc[df_idx, "rl_confidence"] = confidence
        
        # Store prediction history
        data_store['predictions_history'].append({
            'timestamp': time.time(),
            'rl_pred': rl_action,
            'ensemble_pred': ensemble_pred_val,
            'confidence': confidence
        })
    
    threshold = data_store['threshold']
    df["alert"] = np.where(df["anomaly_score"] > threshold, "Anomaly", "Normal")
    df["tampering_type"] = df.apply(classify_tampering, axis=1)
    
    # Update heatmap data
    if len(df) > 0:
        latest = df.iloc[-1]
        data_store['heatmap_data'].append({
            'time': latest['time'],
            'candidates': {f"cand{i}": latest.get(f"diff_{i}", 0) for i in range(1, num_cands + 1)},
            'anomaly_score': latest['anomaly_score']
        })
    
    # Fill NaN only for numeric columns, preserve string columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df



def simulation_worker():
    """Background worker for data simulation with WebSocket updates."""
    global data_store, rl_agent
    
    while data_store['running'] and data_store['step'] < simulation_config['duration']:
        if not data_store['running']:
            break
        
        data = get_fpga_vote_data()
        data["time"] = time.time() - data_store['start_time']
        data_store['df'] = pd.concat([data_store['df'], pd.DataFrame([data])], ignore_index=True)
        df = data_store['df']
        
        for col in get_candidates():
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df = extract_features(df)
        df = detect_anomaly(df)
        data_store['df'] = df
        
        # Emit WebSocket update
        latest = df.iloc[-1].to_dict()
        socketio.emit('data_update', {
            'step': data_store['step'],
            'latest': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in latest.items()},
            'anomaly_detected': latest.get('alert') == 'Anomaly'
        })
        
        # Send email notification for anomalies
        if latest.get('alert') == 'Anomaly':
            socketio.emit('anomaly_alert', {
                'time': latest['time'],
                'type': latest.get('tampering_type', 'Unknown'),
                'score': float(latest.get('anomaly_score', 0))
            })
        
        if data_store['step'] % 10 == 0 and data_store['step'] > 0 and rl_agent:
            rl_agent.save_model()
        
        data_store['step'] += 1
        time.sleep(1)
    
    # Simulation complete
    if data_store['step'] >= simulation_config['duration']:
        data_store['running'] = False
        socketio.emit('simulation_complete', get_final_results())


def get_final_results():
    """Get final simulation results."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return {}
    
    candidates = get_candidates()
    final_votes = df[candidates].sum().astype(int).to_dict()
    winner_name = max(final_votes, key=final_votes.get)
    
    # Count anomalies - try string match first, then score threshold
    anomaly_count = len(df[df['alert'] == 'Anomaly']) if 'alert' in df.columns else 0
    if anomaly_count == 0 and 'anomaly_score' in df.columns:
        threshold = data_store.get('threshold', 0.5)
        anomaly_count = len(df[df['anomaly_score'] > threshold])
    
    return {
        'final_votes': final_votes,
        'winner': winner_name,
        'winner_votes': int(final_votes[winner_name]),
        'total_anomalies': anomaly_count,
        'total_samples': len(df),
        'is_complete': True
    }


# Initialize on startup
with app.app_context():
    db.create_all()
    
    # Create default admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@voteguard.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

load_models()
reset_dataframe()
rl_agent = RLAnomalyDetector(state_size=10, model_path="rl_dqn_model.pth")


# ============= ROUTES =============

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard_enhanced.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.datetime.utcnow()
            db.session.commit()
            return redirect(url_for('index'))
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout."""
    logout_user()
    return redirect(url_for('login'))


@app.route('/api/status')
def get_status():
    """Get current system status."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({
            'running': data_store['running'],
            'step': data_store['step'],
            'latest': None,
            'stats': {'total_samples': 0, 'anomalies': 0, 'normal': 0},
            'config': simulation_config
        })
    
    candidates = get_candidates()
    latest = df.iloc[-1].to_dict()
    anomalies = len(df[df["alert"] == "Anomaly"])
    final_votes = df[candidates].sum().astype(int).to_dict() if len(df) > 0 else {}
    winner_name = max(final_votes, key=final_votes.get) if final_votes else None
    
    return jsonify({
        'running': data_store['running'],
        'step': data_store['step'],
        'latest': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in latest.items()},
        'stats': {
            'total_samples': len(df),
            'anomalies': anomalies,
            'normal': len(df) - anomalies
        },
        'config': simulation_config,
        'final_results': {
            'is_complete': not data_store['running'] and data_store['step'] >= simulation_config['duration'],
            'final_votes': final_votes,
            'winner': winner_name,
            'winner_votes': int(final_votes.get(winner_name, 0)) if winner_name else 0
        }
    })


@app.route('/api/data')
def get_data():
    """Get all data for charts."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({
            'vote_trends': [],
            'anomaly_scores': [],
            'rl_rewards': [],
            'rl_losses': [],
            'model_scores': [],
            'heatmap_data': [],
            'trend_data': {}
        })
    
    candidates = get_candidates()
    vote_trends = {'time': df['time'].tolist()}
    for cand in candidates:
        vote_trends[cand] = df[cand].tolist()
    
    return jsonify({
        'vote_trends': vote_trends,
        'anomaly_scores': {
            'time': df['time'].tolist(),
            'scores': df['anomaly_score'].tolist(),
            'alerts': df['alert'].tolist()
        },
        'rl_rewards': rl_agent.rewards_history[-100:] if rl_agent else [],
        'rl_losses': rl_agent.loss_history[-100:] if rl_agent else [],
        'model_scores': {
            'time': df['time'].tolist(),
            'iso': df['iso_score'].tolist() if 'iso_score' in df.columns else [],
            'lof': df['lof_score'].tolist() if 'lof_score' in df.columns else [],
            'ocsvm': df['ocsvm_score'].tolist() if 'ocsvm_score' in df.columns else [],
            'ensemble': df['anomaly_score'].tolist(),
            'rl_pred': df['rl_prediction'].tolist() if 'rl_prediction' in df.columns else []
        },
        'rl_metrics': rl_agent.get_learning_metrics() if rl_agent else {},
        'heatmap_data': data_store['heatmap_data'][-50:],
        'trend_data': {
            'ma_5': df['ma_5'].tolist() if 'ma_5' in df.columns else [],
            'ma_10': df['ma_10'].tolist() if 'ma_10' in df.columns else []
        },
        'candidate_totals': df[candidates].sum().astype(int).to_dict()
    })


@app.route('/api/prediction')
def get_prediction():
    """Get anomaly prediction."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({'probability': 0.0, 'confidence': 0.0, 'trend': 'stable'})
    
    return jsonify(predict_anomaly(df))


@app.route('/api/confidence')
def get_confidence():
    """Get model confidence levels."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({
            'iso': {'confidence': 0, 'accuracy': 0},
            'lof': {'confidence': 0, 'accuracy': 0},
            'ocsvm': {'confidence': 0, 'accuracy': 0},
            'rl': {'confidence': 0, 'epsilon': 1.0}
        })
    
    # Calculate model agreement as confidence proxy
    if len(df) > 5:
        agreement = (df['iso_score'] == df['lof_score']).mean() if 'iso_score' in df.columns else 0
        
        return jsonify({
            'iso': {'confidence': float(agreement), 'predictions': int(df['iso_score'].sum()) if 'iso_score' in df.columns else 0},
            'lof': {'confidence': float(agreement), 'predictions': int(df['lof_score'].sum()) if 'lof_score' in df.columns else 0},
            'ocsvm': {'confidence': float(agreement), 'predictions': int(df['ocsvm_score'].sum()) if 'ocsvm_score' in df.columns else 0},
            'rl': {
                'confidence': float(df['rl_confidence'].mean()) if 'rl_confidence' in df.columns else 0,
                'epsilon': rl_agent.epsilon if rl_agent else 1.0,
                'predictions': int(df['rl_prediction'].sum()) if 'rl_prediction' in df.columns else 0
            },
            'ensemble': {
                'agreement_rate': float(agreement),
                'avg_score': float(df['anomaly_score'].mean())
            }
        })
    
    return jsonify({})


@app.route('/api/heatmap')
def get_heatmap():
    """Get heatmap data for visualization."""
    return jsonify({'data': data_store['heatmap_data'][-100:]})


@app.route('/api/3d-trajectories')
def get_3d_trajectories():
    """Get 3D trajectory data for vote patterns."""
    df = data_store['df']
    if df is None or len(df) < 3:
        return jsonify({'trajectories': []})
    
    candidates = get_candidates()
    trajectories = []
    
    for idx, row in df.iterrows():
        point = {
            'time': float(row['time']),
            'votes': [int(row[c]) for c in candidates[:3]],  # Use first 3 candidates for 3D
            'anomaly': row.get('alert') == 'Anomaly',
            'score': float(row.get('anomaly_score', 0))
        }
        trajectories.append(point)
    
    return jsonify({'trajectories': trajectories})


@app.route('/api/decision-tree')
def get_decision_tree():
    """Get decision tree visualization data."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({'tree': None})
    
    # Create a simplified decision tree structure for visualization
    tree = {
        'name': 'Ensemble Decision',
        'children': [
            {
                'name': 'Isolation Forest',
                'value': int(df['iso_score'].sum()) if 'iso_score' in df.columns else 0,
                'color': '#ef4444',
                'description': f"Contamination: {ml_config['iso_contamination']}"
            },
            {
                'name': 'LOF',
                'value': int(df['lof_score'].sum()) if 'lof_score' in df.columns else 0,
                'color': '#3b82f6',
                'description': f"Neighbors: {ml_config['lof_n_neighbors']}"
            },
            {
                'name': 'One-Class SVM',
                'value': int(df['ocsvm_score'].sum()) if 'ocsvm_score' in df.columns else 0,
                'color': '#10b981',
                'description': f"Kernel: {ml_config['ocsvm_kernel']}"
            },
            {
                'name': 'RL Agent',
                'value': int(df['rl_prediction'].sum()) if 'rl_prediction' in df.columns else 0,
                'color': '#f59e0b',
                'description': f"Epsilon: {rl_agent.epsilon:.3f}" if rl_agent else "N/A"
            }
        ]
    }
    
    return jsonify({'tree': tree})


@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({'alerts': []})
    
    candidates = get_candidates()
    
    # Try multiple ways to find anomalies
    if 'alert' in df.columns:
        # First try string match
        anomalies = df[df["alert"] == "Anomaly"]
        
        # If no matches found, try based on anomaly_score threshold
        if len(anomalies) == 0 and 'anomaly_score' in df.columns:
            threshold = data_store.get('threshold', 0.5)
            anomalies = df[df["anomaly_score"] > threshold]
    elif 'anomaly_score' in df.columns:
        threshold = data_store.get('threshold', 0.5)
        anomalies = df[df["anomaly_score"] > threshold]
    else:
        return jsonify({'alerts': []})
    
    # Get last 10 anomalies
    anomalies = anomalies.tail(10)
    alerts = []
    
    for idx, row in anomalies.iterrows():
        try:
            alert = {
                'time': float(row.get('time', 0)),
                'type': str(row.get('tampering_type', 'Unknown')),
                'score': float(row.get('anomaly_score', 0)),
                'candidates': {c: int(row.get(c, 0)) for c in candidates},
                'rl_prediction': int(row.get('rl_prediction', 0)) if 'rl_prediction' in row else 0
            }
            alerts.append(alert)
        except Exception as e:
            print(f"Error processing alert row: {e}")
            continue
    
    return jsonify({'alerts': alerts})


@app.route('/api/candidate-anomalies')
def get_candidate_anomalies():
    """Get anomaly attribution by candidate."""
    df = data_store['df']
    candidates = get_candidates()
    diff_cols = get_diff_cols()
    
    if df is None or len(df) == 0:
        return jsonify({'pie_data': {}, 'bar_data': {}})
    
    anomalies = df[df["alert"] == "Anomaly"].copy()
    tampered_counts = {c: 0 for c in candidates}
    candidate_anomaly_type_counts = {c: {} for c in candidates}
    
    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            diff_values = [row.get(col, 0) for col in diff_cols]
            abs_diffs = [abs(d) for d in diff_values]
            
            if any(abs_diffs):
                max_abs_diff_value = max(abs_diffs)
                max_diff_index = abs_diffs.index(max_abs_diff_value)
                cand_key = candidates[max_diff_index]
                tampered_counts[cand_key] += 1
                
                tampering_type = row.get('tampering_type', 'Unknown')
                candidate_anomaly_type_counts[cand_key][tampering_type] = \
                    candidate_anomaly_type_counts[cand_key].get(tampering_type, 0) + 1
    
    pie_data = {k.upper(): v for k, v in tampered_counts.items() if v > 0}
    bar_data = {cand.upper(): counts for cand, counts in candidate_anomaly_type_counts.items() if counts}
    
    return jsonify({'pie_data': pie_data, 'bar_data': bar_data})


@app.route('/api/control', methods=['POST'])
def control():
    """Control simulation start/stop."""
    global rl_agent
    action = request.json.get('action')
    
    if action == 'start':
        if not data_store['running']:
            reset_dataframe()
            load_models()  # Reload models with current config
            data_store['running'] = True
            data_store['start_time'] = time.time()
            data_store['step'] = 0
            thread = threading.Thread(target=simulation_worker, daemon=True)
            thread.start()
            socketio.emit('simulation_started', {'config': simulation_config})
        return jsonify({'status': 'started'})
    
    elif action == 'stop':
        data_store['running'] = False
        return jsonify({'status': 'stopped'})
    
    elif action == 'threshold':
        threshold = request.json.get('threshold', 0.5)
        data_store['threshold'] = threshold
        return jsonify({'status': 'updated', 'threshold': threshold})
    
    return jsonify({'status': 'error'})


@app.route('/api/config', methods=['GET', 'POST'])
def config_api():
    """Get or update simulation configuration."""
    global simulation_config, ml_config
    
    if request.method == 'POST':
        data = request.json
        
        # Update simulation config
        if 'simulation' in data:
            sim = data['simulation']
            simulation_config['num_candidates'] = sim.get('num_candidates', 4)
            simulation_config['anomaly_probability'] = sim.get('anomaly_probability', 0.15)
            simulation_config['duration'] = sim.get('duration', 60)
            simulation_config['overt_anomaly_prob'] = sim.get('overt_anomaly_prob', 0.10)
            simulation_config['stealth_anomaly_prob'] = sim.get('stealth_anomaly_prob', 0.05)
        
        # Update ML config
        if 'ml' in data:
            ml = data['ml']
            ml_config['iso_contamination'] = ml.get('iso_contamination', 0.1)
            ml_config['iso_n_estimators'] = ml.get('iso_n_estimators', 100)
            ml_config['lof_n_neighbors'] = ml.get('lof_n_neighbors', 10)
            ml_config['ocsvm_kernel'] = ml.get('ocsvm_kernel', 'rbf')
            ml_config['ocsvm_gamma'] = ml.get('ocsvm_gamma', 'auto')
        
        return jsonify({'status': 'updated', 'simulation': simulation_config, 'ml': ml_config})
    
    return jsonify({'simulation': simulation_config, 'ml': ml_config})


@app.route('/api/history')
def get_history():
    """Get simulation history for comparison."""
    sessions = SimulationSession.query.order_by(SimulationSession.created_at.desc()).limit(10).all()
    return jsonify({'sessions': [s.to_dict() for s in sessions]})


@app.route('/api/history/<int:session_id>')
def get_session(session_id):
    """Get specific session data."""
    session = SimulationSession.query.get_or_404(session_id)
    return jsonify({
        'session': session.to_dict(),
        'data': session.get_data()
    })


@app.route('/api/email/subscribe', methods=['POST'])
def subscribe_email():
    """Subscribe to email notifications."""
    data = request.json
    email = data.get('email')
    name = data.get('name', '')
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    existing = EmailSubscription.query.filter_by(email=email).first()
    if existing:
        existing.is_active = True
        existing.on_anomaly = data.get('on_anomaly', True)
        existing.on_completion = data.get('on_completion', True)
    else:
        sub = EmailSubscription(
            email=email,
            name=name,
            on_anomaly=data.get('on_anomaly', True),
            on_completion=data.get('on_completion', True)
        )
        db.session.add(sub)
    
    db.session.commit()
    return jsonify({'status': 'subscribed'})


@app.route('/api/export/pdf')
def export_pdf():
    """Generate and download PDF report."""
    df = data_store['df']
    if df is None or len(df) == 0:
        return jsonify({'error': 'No data available'}), 400
    
    filename = generate_pdf_report(df)
    return send_file(
        filename,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"VoteGuard_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )


def generate_pdf_report(df):
    """Generate comprehensive PDF report matching the reference design."""
    candidates = get_candidates()
    diff_cols = get_diff_cols()
    
    filename = f"VoteGuard_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # Colors for candidates
    cand_colors = [
        (0.9, 0.3, 0.3),   # Red
        (0.3, 0.3, 0.9),   # Blue
        (0.3, 0.7, 0.3),   # Green
        (0.9, 0.6, 0.2),   # Orange
        (0.6, 0.3, 0.7),   # Purple
        (0.9, 0.4, 0.6),   # Pink
        (0.3, 0.7, 0.7),   # Teal
        (0.5, 0.5, 0.5),   # Gray
    ]
    
    # Calculate stats
    final_votes = df[candidates].sum().astype(int)
    winner_name = final_votes.idxmax()
    winner_votes = int(final_votes.max())
    
    # Get anomalies
    anomalies = df[df["anomaly_score"] > data_store.get('threshold', 0.5)].copy()
    if len(anomalies) == 0:
        anomalies = df[df["alert"] == "Anomaly"].copy()
    total_anomalies = len(anomalies)
    
    # Calculate anomaly attribution per candidate
    tampered_counts = {c: 0 for c in candidates}
    for _, row in anomalies.iterrows():
        diff_values = [abs(row.get(col, 0)) for col in diff_cols]
        if any(diff_values):
            max_idx = diff_values.index(max(diff_values))
            if max_idx < len(candidates):
                tampered_counts[candidates[max_idx]] += 1
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-50, "AI Voting Anomaly Detection Report")
    
    # Metadata
    y = height - 100
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, y - 18, f"Total Simulation Time: {df['time'].iloc[-1]:.2f} seconds")
    
    # Winner
    y -= 55
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Election Winner: {winner_name.upper()} with {winner_votes} Total Votes")
    
    # Anomaly Attribution (right side)
    attr_x = 350
    attr_y = height - 100
    c.setFont("Helvetica-Bold", 12)
    c.drawString(attr_x, attr_y, "Anomaly Attribution by Candidate")
    attr_y -= 20
    c.setFont("Helvetica", 10)
    for i, cand in enumerate(candidates):
        count = tampered_counts.get(cand, 0)
        pct = (count / total_anomalies * 100) if total_anomalies > 0 else 0
        # Draw color box
        c.setFillColorRGB(*cand_colors[i % len(cand_colors)])
        c.rect(attr_x, attr_y - 3, 10, 10, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(attr_x + 15, attr_y, f"{cand.upper()}: {count} instances ({pct:.1f}%)")
        attr_y -= 15
    
    # Candidate Summary
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Candidate Summary")
    y -= 20
    c.setFont("Helvetica", 11)
    for i, cand in enumerate(candidates):
        votes = int(final_votes[cand])
        anomaly_count = tampered_counts.get(cand, 0)
        c.setFillColorRGB(*cand_colors[i % len(cand_colors)])
        c.drawString(60, y, f"- {cand.upper()}: {votes} Votes | Suspected Instances: {anomaly_count}")
        c.setFillColorRGB(0, 0, 0)
        y -= 16
    
    # Total Anomalies
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Total Anomalies Detected: {total_anomalies}")
    
    # Separator line
    y -= 15
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.line(50, y, width - 50, y)
    
    # Anomaly Log Header
    y -= 25
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0, 0, 0)
    
    # Anomaly details table
    for idx, (_, row) in enumerate(anomalies.head(40).iterrows()):
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Anomaly Log (continued)")
            y -= 25
            c.setFont("Helvetica", 9)
        
        time_val = row.get('time', 0)
        anomaly_type = row.get('tampering_type', 'Unclassified Anomaly')
        score = row.get('anomaly_score', 1.0)
        votes = [int(row.get(cand, 0)) for cand in candidates]
        
        c.setFont("Helvetica", 9)
        line = f"[{idx+1} | {time_val:.2f}s] Type: {anomaly_type} | Score: {score:.2f} | Votes: {votes}"
        c.drawString(50, y, line)
        y -= 12
    
    c.save()
    return filename



# WebSocket events
@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'Connected to VoteGuard'})


@socketio.on('disconnect')
def handle_disconnect():
    pass


@socketio.on('request_update')
def handle_request_update():
    """Handle manual update request."""
    df = data_store['df']
    if df is not None and len(df) > 0:
        latest = df.iloc[-1].to_dict()
        emit('data_update', {
            'step': data_store['step'],
            'latest': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in latest.items()},
            'anomaly_detected': latest.get('alert') == 'Anomaly'
        })


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
