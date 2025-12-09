"""
Flask Dashboard for Voting Anomaly Detection with RL
Dark-themed modern dashboard inspired by Bhacemp template
"""
from flask import Flask, render_template, jsonify, request, send_file, Response
import pandas as pd
import numpy as np
import time
import datetime
import joblib
import os
import random
import threading
import io
import json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import torch
from rl_anomaly_detector import RLAnomalyDetector

app = Flask(__name__)

# Global configuration
CANDIDATES = [f"cand{i}" for i in range(1, 5)]
DIFF_COLS = [f"diff_{i}" for i in range(1, 5)]
RATIO_COLS = [f"ratio_{i}" for i in range(1, 5)]

# Global data storage
data_store = {
    'df': pd.DataFrame(columns=[
        "time", "cand1", "cand2", "cand3", "cand4",
        "iso_score", "lof_score", "ocsvm_score", "anomaly_score", "alert", "tampering_type",
        "total_votes", "vote_diff_sum",
        "diff_1", "diff_2", "diff_3", "diff_4",
        "ratio_1", "ratio_2", "ratio_3", "ratio_4",
        "rl_prediction", "rl_reward", "rl_confidence"
    ]),
    'running': False,
    'start_time': None,
    'step': 0,
    'threshold': 0.5
}

# Initialize models
iso, lof, ocsvm = None, None, None
rl_agent = None

def get_fpga_vote_data():
    """Simulates vote counts (per instance)."""
    r = random.random()
    if r < 0.10:
        # Inject Overt Anomaly (High Spike)
        return {
            "cand1": random.randint(80, 100),
            "cand2": random.randint(0, 20),
            "cand3": random.randint(0, 10),
            "cand4": random.randint(0, 5)
        }
    elif 0.10 <= r < 0.15:
        # Inject Stealth Anomaly (Sudden Drop/Stall)
        return {
            "cand1": random.randint(0, 5),
            "cand2": random.randint(0, 5),
            "cand3": random.randint(0, 5),
            "cand4": random.randint(0, 5)
        }
    # Normal distribution
    return {
        "cand1": random.randint(40, 60),
        "cand2": random.randint(40, 60),
        "cand3": random.randint(40, 60),
        "cand4": random.randint(40, 60)
    }

def extract_features(df):
    df["total_votes"] = df[CANDIDATES].sum(axis=1)
    for i in range(1, 5):
        df[f"diff_{i}"] = df[f"cand{i}"].diff().fillna(0).copy()
        df[f"ratio_{i}"] = (df[f"cand{i}"]/df["total_votes"]).fillna(0).copy()
    df["vote_diff_sum"] = df[DIFF_COLS].sum(axis=1)
    return df.fillna(0)

def classify_tampering(row):
    """Classify tampering type based on features."""
    if pd.isna(row.get("alert")) or row["alert"] == "Normal":
        return "Not Applicable"
    
    diffs = [row[f"diff_{i}"] for i in range(1, 5)]
    ratios = [row[f"ratio_{i}"] for i in range(1, 5)]
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
    if all(os.path.exists(x) for x in ["iso.pkl", "lof.pkl", "ocsvm.pkl"]):
        iso = joblib.load("iso.pkl")
        lof = joblib.load("lof.pkl")
        ocsvm = joblib.load("ocsvm.pkl")
    else:
        dummy = pd.DataFrame(np.random.rand(50, 10))
        iso = IsolationForest(contamination=0.1, random_state=42)
        lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
        ocsvm = OneClassSVM(kernel='rbf', gamma='auto')
        iso.fit(dummy)
        lof.fit(dummy)
        ocsvm.fit(dummy)
        joblib.dump(iso, "iso.pkl")
        joblib.dump(lof, "lof.pkl")
        joblib.dump(ocsvm, "ocsvm.pkl")

def detect_anomaly(df):
    """Detect anomalies using ensemble + RL."""
    global rl_agent
    
    feature_list = ["total_votes", "vote_diff_sum"] + DIFF_COLS + RATIO_COLS
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
    
    df.loc[1:, "iso_score"] = iso_pred
    df.loc[1:, "lof_score"] = lof_pred
    df.loc[1:, "ocsvm_score"] = ocsvm_pred
    
    ensemble_score = (iso_pred + lof_pred + ocsvm_pred) / 3
    ensemble_pred = np.where(ensemble_score > 0.5, 1, 0)
    df.loc[1:, "anomaly_score"] = ensemble_score
    
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
    
    threshold = data_store['threshold']
    df["alert"] = np.where(df["anomaly_score"] > threshold, "Anomaly", "Normal")
    df["tampering_type"] = df.apply(classify_tampering, axis=1)
    
    return df.fillna(0)

def simulation_worker():
    """Background worker for data simulation."""
    global data_store, rl_agent
    
    while data_store['running'] and data_store['step'] < 60:
        if not data_store['running']:
            break
        
        data = get_fpga_vote_data()
        data["time"] = time.time() - data_store['start_time']
        data_store['df'] = pd.concat([data_store['df'], pd.DataFrame([data])], ignore_index=True)
        df = data_store['df']
        
        for col in CANDIDATES:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df = extract_features(df)
        df = detect_anomaly(df)
        data_store['df'] = df
        
        if data_store['step'] % 10 == 0 and data_store['step'] > 0 and rl_agent:
            rl_agent.save_model()
        
        data_store['step'] += 1
        time.sleep(1)

# Initialize on startup
load_models()
rl_agent = RLAnomalyDetector(state_size=10, model_path="rl_dqn_model.pth")

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status."""
    df = data_store['df']
    if len(df) == 0:
        return jsonify({
            'running': data_store['running'],
            'step': data_store['step'],
            'latest': None,
            'stats': {
                'total_samples': 0,
                'anomalies': 0,
                'normal': 0
            }
        })
    
    latest = df.iloc[-1].to_dict()
    anomalies = len(df[df["alert"] == "Anomaly"])
    
    # Calculate final vote totals (cumulative)
    final_votes = df[CANDIDATES].sum().astype(int).to_dict() if len(df) > 0 else {}
    winner_name = max(final_votes, key=final_votes.get) if final_votes else None
    winner_votes = final_votes.get(winner_name, 0) if winner_name else 0
    
    return jsonify({
        'running': data_store['running'],
        'step': data_store['step'],
        'latest': latest,
        'stats': {
            'total_samples': len(df),
            'anomalies': anomalies,
            'normal': len(df) - anomalies
        },
        'final_results': {
            'is_complete': not data_store['running'] and data_store['step'] >= 60,
            'final_votes': final_votes,
            'winner': winner_name,
            'winner_votes': int(winner_votes)
        }
    })

@app.route('/api/data')
def get_data():
    """Get all data for charts."""
    df = data_store['df']
    if len(df) == 0:
        return jsonify({
            'vote_trends': [],
            'anomaly_scores': [],
            'rl_rewards': [],
            'rl_losses': [],
            'model_scores': []
        })
    
    return jsonify({
        'vote_trends': {
            'time': df['time'].tolist(),
            'cand1': df['cand1'].tolist(),
            'cand2': df['cand2'].tolist(),
            'cand3': df['cand3'].tolist(),
            'cand4': df['cand4'].tolist()
        },
        'anomaly_scores': {
            'time': df['time'].tolist(),
            'scores': df['anomaly_score'].tolist(),
            'alerts': df['alert'].tolist()
        },
        'rl_rewards': rl_agent.rewards_history[-100:] if rl_agent else [],
        'rl_losses': rl_agent.loss_history[-100:] if rl_agent else [],
        'model_scores': {
            'time': df['time'].tolist(),
            'ensemble': df['anomaly_score'].tolist(),  # Ensemble = average of iso, lof, ocsvm
            'rl_pred': df['rl_prediction'].tolist() if 'rl_prediction' in df.columns else []
        },
        'rl_metrics': rl_agent.get_learning_metrics() if rl_agent else {}
    })

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts."""
    df = data_store['df']
    if len(df) == 0:
        return jsonify({'alerts': []})
    
    anomalies = df[df["alert"] == "Anomaly"].tail(10)
    alerts = []
    for _, row in anomalies.iterrows():
        alerts.append({
            'time': row['time'],
            'type': row['tampering_type'],
            'score': float(row['anomaly_score']),
            'candidates': {
                'cand1': int(row['cand1']),
                'cand2': int(row['cand2']),
                'cand3': int(row['cand3']),
                'cand4': int(row['cand4'])
            }
        })
    
    return jsonify({'alerts': alerts})

@app.route('/api/candidate-anomalies')
def get_candidate_anomalies():
    """Get anomaly attribution by candidate."""
    df = data_store['df']
    if len(df) == 0:
        return jsonify({
            'pie_data': {},
            'bar_data': {}
        })
    
    anomalies = df[df["alert"] == "Anomaly"].copy()
    
    # Initialize counts
    tampered_counts = {c: 0 for c in CANDIDATES}
    candidate_anomaly_type_counts = {c: {} for c in CANDIDATES}
    
    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            # Attribute anomaly to candidate with largest absolute change
            diff_values = [row[col] for col in DIFF_COLS]
            abs_diffs = [abs(d) for d in diff_values]
            
            if any(abs_diffs):
                max_abs_diff_value = max(abs_diffs)
                max_diff_index_0_based = abs_diffs.index(max_abs_diff_value)
                cand_key = CANDIDATES[max_diff_index_0_based]
                tampered_counts[cand_key] += 1
                
                # Update anomaly type breakdown
                tampering_type = row['tampering_type']
                candidate_anomaly_type_counts[cand_key][tampering_type] = \
                    candidate_anomaly_type_counts[cand_key].get(tampering_type, 0) + 1
    
    # Prepare pie chart data (only candidates with anomalies)
    pie_data = {k.upper(): v for k, v in tampered_counts.items() if v > 0}
    
    # Prepare stacked bar chart data
    bar_data = {}
    for cand in CANDIDATES:
        if candidate_anomaly_type_counts[cand]:
            bar_data[cand.upper()] = candidate_anomaly_type_counts[cand]
    
    return jsonify({
        'pie_data': pie_data,
        'bar_data': bar_data
    })

@app.route('/api/control', methods=['POST'])
def control():
    """Control simulation start/stop."""
    action = request.json.get('action')
    
    if action == 'start':
        if not data_store['running']:
            data_store['running'] = True
            data_store['start_time'] = time.time()
            data_store['step'] = 0
            thread = threading.Thread(target=simulation_worker, daemon=True)
            thread.start()
        return jsonify({'status': 'started'})
    
    elif action == 'stop':
        data_store['running'] = False
        return jsonify({'status': 'stopped'})
    
    elif action == 'threshold':
        threshold = request.json.get('threshold', 0.5)
        data_store['threshold'] = threshold
        return jsonify({'status': 'updated', 'threshold': threshold})
    
    return jsonify({'status': 'error'})

@app.route('/api/export/pdf')
def export_pdf():
    """Generate and download PDF report."""
    df = data_store['df']
    if len(df) == 0:
        return jsonify({'error': 'No data available'}), 400
    
    # Generate PDF
    filename = generate_pdf_report(df)
    
    return send_file(
        filename,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"Voting_Anomaly_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

def generate_pdf_report(df):
    """Generate PDF report similar to Streamlit version."""
    filename = f"Voting_Anomaly_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, height-50, "AI Voting Anomaly Detection Report")
    
    # Calculate Final Vote Totals and Winner
    final_votes = df[CANDIDATES].sum().astype(int)
    winner_name = final_votes.idxmax()
    winner_votes = final_votes.max()
    
    # Check for Anomaly Data & Calculate Attribution
    anomalies = df[df["alert"] == "Anomaly"].copy()
    total_anomalies = len(anomalies)
    
    # Calculate Suspected Tampered Instances
    tampered_counts = {c: 0 for c in CANDIDATES}
    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            diff_values = [row[col] for col in DIFF_COLS]
            abs_diffs = [abs(d) for d in diff_values]
            if any(abs_diffs):
                max_abs_diff_value = max(abs_diffs)
                max_diff_index_0_based = abs_diffs.index(max_abs_diff_value)
                cand_key = CANDIDATES[max_diff_index_0_based]
                tampered_counts[cand_key] += 1
    
    pie_data = {k.upper(): v for k, v in tampered_counts.items() if v > 0}
    
    # Summary Section
    y_current = height - 80
    c.setFont("Helvetica", 12)
    c.drawString(50, y_current, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, y_current - 20, f"Total Simulation Time: {df['time'].iloc[-1]:.2f} seconds")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_current - 50, f"Election Winner: {winner_name.upper()} with {winner_votes} Total Votes")
    
    # Anomaly Attribution
    y_current_for_pie_start = height - 80
    if pie_data:
        total = sum(pie_data.values())
        pie_colors = [colors.red, colors.blue, colors.green, colors.orange, colors.purple, colors.yellow]
        c.setFont("Helvetica-Bold", 12)
        c.drawString(320, y_current_for_pie_start, "Anomaly Attribution by Candidate")
        c.setFont("Helvetica", 10)
        y_legend = y_current_for_pie_start - 15
        for i, (label, value) in enumerate(pie_data.items()):
            c.setFillColor(pie_colors[i % len(pie_colors)])
            c.rect(320, y_legend - 10, 10, 10, fill=1)
            c.setFillColor(colors.black)
            c.drawString(335, y_legend, f"{label}: {value} instances ({value/total:.1%})")
            y_legend -= 15
        y_current = y_legend - 15
    else:
        y_current = y_current_for_pie_start - 50
    
    # Candidate Totals
    y_current -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_current, "Candidate Summary")
    c.setFont("Helvetica", 11)
    y_current -= 20
    for cand_name, votes in final_votes.items():
        tampered = tampered_counts.get(cand_name, 0)
        text = f"- {cand_name.upper()}: {votes} Votes"
        if tampered > 0:
            text += f" | Suspected Instances: {tampered}"
            c.setFillColor(colors.red)
            c.drawString(70, y_current, text)
            c.setFillColor(colors.black)
        else:
            c.drawString(70, y_current, text)
        y_current -= 15
    
    y_current -= 15
    c.setFont("Helvetica", 10)
    if total_anomalies > 0:
        c.drawString(50, y_current, f"Total Anomalies Detected: {total_anomalies}")
    
    y_current -= 20
    c.drawString(50, y_current, "-" * 100)
    y = y_current - 20
    
    # Detailed Anomaly Log
    for _, row in anomalies.iterrows():
        txt = f"[{int(row.name)} | {row['time']:.2f}s] Type: {row['tampering_type']} | Score: {row['anomaly_score']:.2f} | Votes: [{int(row['cand1'])},{int(row['cand2'])},{int(row['cand3'])},{int(row['cand4'])}]"
        c.drawString(50, y, txt)
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 80
            c.setFont("Helvetica", 10)
    
    c.save()
    return filename

@app.route('/api/export/chart/<chart_name>')
def export_chart(chart_name):
    """Export chart as image (placeholder - requires client-side implementation)."""
    # This endpoint is a placeholder
    # Actual chart export should be done client-side using Chart.js toBase64Image()
    return jsonify({
        'message': 'Chart export should be done client-side',
        'chart_name': chart_name,
        'instructions': 'Use chart.toBase64Image() in JavaScript'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

