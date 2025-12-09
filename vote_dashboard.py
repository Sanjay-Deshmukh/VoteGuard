# ==========================================================
# AI Voting Anomaly Dashboard (FINAL VERSION with Robust Attribution + RL)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import time, datetime, joblib, os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors # Required for coloring in the PDF report
import random
import torch
from rl_anomaly_detector import RLAnomalyDetector

# --- Define Candidate Names Globally for Consistency ---
CANDIDATES = [f"cand{i}" for i in range(1, 5)] # ['cand1', 'cand2', 'cand3', 'cand4']
DIFF_COLS = [f"diff_{i}" for i in range(1, 5)]   # ['diff_1', 'diff_2', 'diff_3', 'diff_4']
RATIO_COLS = [f"ratio_{i}" for i in range(1, 5)] # ['ratio_1', 'ratio_2', 'ratio_3', 'ratio_4']


# ==========================================================
# Simulated FPGA Data Stream
# ==========================================================
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

# ==========================================================
# Feature Engineering (CORRECTED)
# ==========================================================
def extract_features(df):
    df["total_votes"] = df[CANDIDATES].sum(axis=1)
    
    # Iterate using the correct candidate index (1 to 4)
    for i in range(1, 5): 
        # FIX: Use cand{i} and diff_{i}
        df[f"diff_{i}"] = df[f"cand{i}"].diff().fillna(0).copy()
        df[f"ratio_{i}"] = (df[f"cand{i}"]/df["total_votes"]).fillna(0).copy()
        
    df["vote_diff_sum"] = df[DIFF_COLS].sum(axis=1) # Use corrected list
    return df.fillna(0)

# ==========================================================
# Tampering Type Classifier (CORRECTED)
# ==========================================================
def classify_tampering(row):
    """Uses rules to suggest the anomaly type based on features."""
    if row["alert"] == "✅ Normal":
        return "Not Applicable"

    # FIX: Use the corrected diff and ratio columns
    diffs = [row[f"diff_{i}"] for i in range(1, 5)] 
    ratios = [row[f"ratio_{i}"] for i in range(1, 5)]
    max_diff = max(diffs)
    min_diff = min(diffs)
    total = row["total_votes"]

    if max_diff > 30 and total > 200: 
        return " Sudden Spike"
    elif min_diff < -30: 
        return " Sudden Drop/Trough"
    elif total > 350: 
        return " Total Vote Overload"
    elif total < 10 and row["time"] > 5:
        return " Vote System Stall"
    elif max(ratios) > 0.85:
        return " Dominant Ratio Injection"
    elif all(0 < d < 10 for d in diffs) and max(abs(d) for d in diffs) > 1:
        return "↗ Gradual Drift"
    elif len(set(diffs)) < 2 and max(abs(d) for d in diffs) > 1:
        return " Duplicate Pattern"
        
    return " Unclassified Anomaly"

# ==========================================================
# AI Models
# ==========================================================
def train_models(normal_data):
    """Train and save the ensemble models."""
    iso = IsolationForest(contamination=0.1, random_state=42)
    lof = LocalOutlierFactor(n_neighbors=10, novelty=True) 
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto')
    
    iso.fit(normal_data)
    lof.fit(normal_data)
    ocsvm.fit(normal_data)
    
    joblib.dump(iso, "iso.pkl")
    joblib.dump(lof, "lof.pkl")
    joblib.dump(ocsvm, "ocsvm.pkl")
    return iso, lof, ocsvm

def load_models():
    """Load models or train them if not found."""
    if all(os.path.exists(x) for x in ["iso.pkl","lof.pkl","ocsvm.pkl"]):
        st.sidebar.success(" AI Models Loaded.")
        return joblib.load("iso.pkl"), joblib.load("lof.pkl"), joblib.load("ocsvm.pkl")
    else:
        st.sidebar.info(" Training New AI Models (10 features)...")
        # Ensure dummy data size matches expected feature count (10)
        dummy = pd.DataFrame(np.random.rand(50, 10)) 
        return train_models(dummy)

# ==========================================================
# Anomaly Detection (CORRECTED + RL Integration)
# ==========================================================
def detect_anomaly(df, iso, lof, ocsvm, rl_agent=None):
    """Applies ensemble models + RL to detect anomalies and individual scores."""
    
    # FIX: Use corrected diff and ratio columns in feature list
    feature_list = ["total_votes", "vote_diff_sum"] + DIFF_COLS + RATIO_COLS
    features = df[feature_list].iloc[1:].copy()
    
    if features.empty:
        return df 

    iso_pred = np.where(iso.predict(features) == -1, 1, 0)
    lof_pred = np.where(lof.predict(features) == -1, 1, 0)
    ocsvm_pred = np.where(ocsvm.predict(features) == -1, 1, 0)
    
    # Ensure scores columns exist and are initialized to 0
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
    
    # Ensemble score
    ensemble_score = (iso_pred + lof_pred + ocsvm_pred) / 3
    ensemble_pred = np.where(ensemble_score > 0.5, 1, 0)
    df.loc[1:, "anomaly_score"] = ensemble_score
    
    # RL Agent Integration (Online Learning) - Process only latest row for real-time learning
    if rl_agent is not None and len(features) > 0:
        # Only process the latest row for online learning
        latest_idx = len(features) - 1
        latest_row = features.iloc[latest_idx]
        
        state = rl_agent.get_state(latest_row.values)
        ensemble_pred_val = ensemble_pred[latest_idx] if latest_idx < len(ensemble_pred) else 0
        ensemble_score_val = ensemble_score[latest_idx] if latest_idx < len(ensemble_score) else 0
        
        # RL learns online
        rl_action, reward, loss = rl_agent.learn_online(state, ensemble_pred_val, ensemble_score_val)
        
        # Get Q-value for confidence
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(rl_agent.device)
        with torch.no_grad():
            q_values = rl_agent.q_network(state_tensor)
            confidence = torch.softmax(q_values, dim=1)[0][rl_action].item()
        
        # Update only the latest row
        df_idx = len(df) - 1
        if df_idx >= 1:  # Ensure we have at least 2 rows (since features start from index 1)
            df.loc[df_idx, "rl_prediction"] = rl_action
            df.loc[df_idx, "rl_reward"] = reward
            df.loc[df_idx, "rl_confidence"] = confidence
    
    threshold = st.session_state.get('threshold', 0.5) 
    df["alert"] = np.where(df["anomaly_score"] > threshold, "⚠ Anomaly", "✅ Normal")
    df["tampering_type"] = df.apply(classify_tampering, axis=1)
    
    return df.fillna(0) 

# ==========================================================
# PDF Report Generator (UPDATED LOGIC with Pie Chart) (CORRECTED)
# ==========================================================
def generate_pdf_report(df):
    """Creates a PDF report of all detected anomalies, winner, and candidate totals, including an Anomaly Attribution Pie Chart."""
    filename = "Voting_Anomaly_Report.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, height-50, "AI Voting Anomaly Detection Report")
    
    # Calculate Final Vote Totals and Winner
    final_votes = df[CANDIDATES].sum().astype(int)
    winner_name = final_votes.idxmax()
    winner_votes = final_votes.max()
    
    # --- Check for Anomaly Data & Calculate Attribution ---
    anomalies = df[df["alert"]=="⚠ Anomaly"].copy()
    total_anomalies = len(anomalies)
    
    # 1. Calculate Suspected Tampered Instances (Attribution based on max absolute change)
    tampered_counts = {c: 0 for c in CANDIDATES}
    if not anomalies.empty:
        
        for index, row in anomalies.iterrows():
            # FIX: Use the corrected diff column names
            diff_values = [row[col] for col in DIFF_COLS]
            abs_diffs = [abs(d) for d in diff_values]
            
            if any(abs_diffs):
                max_abs_diff_value = max(abs_diffs)
                
                # Attribute anomaly to the candidate with the largest change (gain or loss)
                max_diff_index_0_based = abs_diffs.index(max_abs_diff_value) 
                
                # FIX: Map 0-based index to 1-based candidate name
                # If index 0 is max, it corresponds to diff_1, which corresponds to cand1.
                cand_key = CANDIDATES[max_diff_index_0_based] 
                tampered_counts[cand_key] += 1
    
    # Filter out candidates with zero anomalies for the chart
    pie_data = {k.upper(): v for k, v in tampered_counts.items() if v > 0}
    
    # --- Summary Section (Top Left) ---
    y_current = height - 80
    c.setFont("Helvetica", 12)
    c.drawString(50, y_current, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, y_current - 20, f"Total Simulation Time: {df['time'].iloc[-1]:.2f} seconds")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_current - 50, f" Election Winner: {winner_name.upper()} with {winner_votes} Total Votes")
    
    # --- Anomaly Attribution Pie Chart Representation (Top Right) ---
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
            c.rect(320, y_legend - 10, 10, 10, fill=1) # Color box
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 10)
            c.drawString(335, y_legend, f"{label}: {value} instances ({value/total:.1%})")
            y_legend -= 15
        
        y_current = y_legend - 15 # Move the main document flow below the legend
        
    else:
        # No anomalies detected
        c.setFont("Helvetica", 12)
        c.drawString(320, y_current_for_pie_start, " ")
        y_current = y_current_for_pie_start - 50 # Adjust starting position for the next section


    # --- Anomaly Data Per Candidate (Continues below the summary) ---
    y_current -= 10 # Spacer
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_current, " ")
    
    c.setFont("Helvetica", 11)
    y_current -= 20
    
    # 2. Print Candidate Totals and Tampered Counts
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
        
    # --- Anomaly Log Summary ---
    c.setFont("Helvetica", 10)
    if total_anomalies == 0:
        c.drawString(50, y_current, " ")
    else:
        c.drawString(50, y_current, f"Total Anomalies Detected: {total_anomalies}")

    y_current -= 20
    c.drawString(50, y_current, "-----------------------------------------------------------------------------------------------------------------------------------------------------") # FIX: Restore title
    
    y = y_current - 20
    
    # --- Detailed Anomaly Log (Row by Row) ---
    for _, row in anomalies.iterrows():
        # FIX: Use correct cand columns in the log printout
        txt = f"[{int(row.name)} | {row['time']:.2f}s] Type: {row['tampering_type']} | Score: {row['anomaly_score']:.2f} | Votes: [{int(row['cand1'])},{int(row['cand2'])},{int(row['cand3'])},{int(row['cand4'])}]"
        c.drawString(50, y, txt)
        y -= 15
        if y < 80:
            c.showPage(); y = height - 80
            c.setFont("Helvetica", 10)
            
    c.save()
    return filename

# ==========================================================
# STREAMLIT DASHBOARD (Main Execution)
# ==========================================================
st.set_page_config(page_title="AI Voting Anomaly Dashboard", layout="wide")

st.title(" AI-Powered Voting Anomaly Detection System")
st.markdown("Real-time FPGA Simulation • AI Anomaly Detection • Live Dashboard")

# --- INITIALIZE SESSION STATE (CORRECTED + RL) ---
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        "time","cand1","cand2","cand3","cand4", 
        "iso_score", "lof_score", "ocsvm_score", "anomaly_score", "alert", "tampering_type", 
        "total_votes", "vote_diff_sum", 
        # FIX: Initialize with correct diff/ratio names
        "diff_1", "diff_2", "diff_3", "diff_4", 
        "ratio_1", "ratio_2", "ratio_3", "ratio_4",
        # RL columns
        "rl_prediction", "rl_reward", "rl_confidence"
    ])
    st.session_state.start_time = time.time()
    st.session_state.running = False 
    st.session_state.step = 0 
    st.session_state.threshold = 0.5 

# Initialize models and data structures
iso, lof, ocsvm = load_models()

# Initialize RL Agent
if 'rl_agent' not in st.session_state:
    st.session_state.rl_agent = RLAnomalyDetector(state_size=10, model_path="rl_dqn_model.pth")
    st.sidebar.success("🤖 RL Agent Initialized (Online Learning Enabled)")

rl_agent = st.session_state.rl_agent
df = st.session_state.df

# --- SIDEBAR CONTROLS (Pause/Resume, Threshold) ---
st.sidebar.subheader("Simulation Controls")
col_start, col_stop = st.sidebar.columns(2)

if st.session_state.running:
    if col_stop.button("🛑 Pause Simulation"):
        st.session_state.running = False
else:
    if col_start.button("▶ Start/Resume"):
        st.session_state.running = True
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Anomaly Sensitivity")
st.session_state.threshold = st.sidebar.slider(
    'Alert Threshold (Score >)',
    min_value=0.0, max_value=1.0, value=st.session_state.threshold, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 RL Agent Controls")
if st.sidebar.button("💾 Save RL Model"):
    rl_agent.save_model()
    st.sidebar.success("RL Model Saved!")

# Display RL metrics
rl_metrics = rl_agent.get_learning_metrics()
if len(rl_metrics['rewards']) > 0:
    st.sidebar.metric("RL Learning Steps", rl_metrics['step_count'])
    st.sidebar.metric("Avg Reward (Last 100)", f"{rl_metrics['avg_reward']:.3f}")
    st.sidebar.metric("Exploration Rate (ε)", f"{rl_agent.epsilon:.3f}")

# --- MAIN LAYOUT ---
placeholder = st.empty() 
col1, col2 = st.columns([2,1])
col3, col4 = st.columns(2)
# Define columns for the new charts
col5, col6 = st.columns(2)
# RL visualization columns
col7, col8 = st.columns(2)


# Check if simulation is running and hasn't hit max steps (20 seconds)
if st.session_state.running and st.session_state.step < 50:
    
    # ----------------------------------
    # LIVE SIMULATION STEP
    # ----------------------------------
    
    data = get_fpga_vote_data()
    data["time"] = time.time() - st.session_state.start_time
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([data])], ignore_index=True)
    df = st.session_state.df
    
    for col in CANDIDATES:
        # Ensure conversion handles potential mixed types before arithmetic
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df = extract_features(df)
    df = detect_anomaly(df, iso, lof, ocsvm, rl_agent)
    
    # Save RL model periodically
    if st.session_state.step % 10 == 0 and st.session_state.step > 0:
        rl_agent.save_model()
    
    st.session_state.step += 1
    
    latest = df.iloc[-1]
    alert_color = "red" if latest["alert"]=="⚠ Anomaly" else "green"

    # --- TOP LEFT (Col 1: Vote Trends) ---
    with col1:
        st.subheader(" Live Vote Count Trends")
        fig_trend, ax_trend = plt.subplots(figsize=(7, 4))
        ax_trend.plot(df['time'], df['cand1'], label='Candidate 1')
        ax_trend.plot(df['time'], df['cand2'], label='Candidate 2')
        ax_trend.plot(df['time'], df['cand3'], label='Candidate 3')
        ax_trend.plot(df['time'], df['cand4'], label='Candidate 4')
        ax_trend.set_ylim(0, 100)
        ax_trend.set_xlabel("Time (seconds)")
        ax_trend.set_ylabel("Vote Count (per instance)")
        ax_trend.legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        st.pyplot(fig_trend)

    # --- TOP RIGHT (Col 2: Alerts & Summary) ---
    with col2:
        st.subheader(" Current Status")
        st.markdown(f"<h2 style='color:{alert_color};'>{latest['alert']}</h2>", unsafe_allow_html=True)
        st.write(f"*Type:* {latest['tampering_type']}")
        st.write(f"*Anomaly Score:* {latest['anomaly_score']:.2f}")
        st.write(f"*Total Votes (per instance):* {int(latest['total_votes'])}")
        st.write(f"*Time:* {latest['time']:.2f}s")
        st.markdown("---")
        
        # RL Prediction Display
        if 'rl_prediction' in latest and not pd.isna(latest['rl_prediction']):
            rl_pred_text = "⚠️ Anomaly" if latest['rl_prediction'] == 1 else "✅ Normal"
            rl_color = "red" if latest['rl_prediction'] == 1 else "green"
            rl_confidence = latest.get('rl_confidence', 0)
            st.markdown(f"**🤖 RL Prediction:** <span style='color:{rl_color};'>{rl_pred_text}</span>", unsafe_allow_html=True)
            st.write(f"*RL Confidence:* {rl_confidence:.2f}")
            if 'rl_reward' in latest and not pd.isna(latest['rl_reward']):
                reward_color = "green" if latest['rl_reward'] > 0 else "red"
                st.write(f"*RL Reward:* <span style='color:{reward_color};'>{latest['rl_reward']:.2f}</span>", unsafe_allow_html=True)
            st.markdown("---")
        
        anomaly_log = df[df["alert"] == "⚠ Anomaly"][
            ["time", "tampering_type", "anomaly_score", "cand1", "cand2", "cand3", "cand4"]
        ].rename(columns={'time': 'Time (s)', 'tampering_type': 'Type', 'anomaly_score': 'Score'})
        
        if not anomaly_log.empty:
            st.subheader("🚩 Anomaly Log")
            st.dataframe(anomaly_log.iloc[::-1],
                         hide_index=True, use_container_width=True)

    # --- BOTTOM LEFT (Col 3: Anomaly Count) ---
    with col3:
        st.subheader("Time-Series Anomaly Count")
        df['anomaly_flag'] = np.where(df['alert'] == '⚠ Anomaly', 1, 0)
        df['cumulative_anomalies'] = df['anomaly_flag'].cumsum()
        
        st.line_chart(df[['cumulative_anomalies']], use_container_width=True)

    # --- BOTTOM RIGHT (Col 4: Model Confidence) ---
    with col4:
        st.subheader(" Model Confidence Plot")
        fig_conf, ax_conf = plt.subplots(figsize=(7, 4))
        ax_conf.plot(df['time'], df['iso_score'], label='Isolation Forest', marker='.')
        ax_conf.plot(df['time'], df['lof_score'], label='LOF', marker='.')
        ax_conf.plot(df['time'], df['ocsvm_score'], label='One-Class SVM', marker='.')
        ax_conf.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Strong Alert')
        ax_conf.set_xlabel("Time (seconds)")
        ax_conf.set_ylabel("Individual Model Flag (0/1)")
        ax_conf.set_ylim(-0.1, 1.1)
        ax_conf.legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        st.pyplot(fig_conf)
    
    # ----------------------------------
    # NEW: ANOMALY ATTRIBUTION CHARTS (Col 5 and Col 6)
    # ----------------------------------
    anomalies_df = df[df["alert"] == "⚠ Anomaly"].copy()
    tampered_counts = {c: 0 for c in CANDIDATES}
    
    # Structure for the second chart: Candidate by Tampering Type
    candidate_anomaly_type_counts = {c: {} for c in CANDIDATES}

    if not anomalies_df.empty:
        
        for _, row in anomalies_df.iterrows():
            diff_values = [row[col] for col in DIFF_COLS]
            abs_diffs = [abs(d) for d in diff_values]
            
            if any(abs_diffs):
                max_abs_diff_value = max(abs_diffs)
                
                # Attribute anomaly to the candidate with the largest change (gain or loss)
                max_diff_index_0_based = abs_diffs.index(max_abs_diff_value)
                cand_key = CANDIDATES[max_diff_index_0_based] 
                tampered_counts[cand_key] += 1
                
                # Update the Candidate by Tampering Type matrix
                tampering_type = row['tampering_type']
                candidate_anomaly_type_counts[cand_key][tampering_type] = \
                    candidate_anomaly_type_counts[cand_key].get(tampering_type, 0) + 1

    st.markdown("---")
    st.subheader(" ")

    # 1. PIE CHART: Total Anomalies Per Candidate
    with col5:
        st.subheader("Anomalies Per Candidate")
        
        # Filter out candidates with 0 attributed anomalies for cleaner pie chart
        pie_data = {k.upper(): v for k, v in tampered_counts.items() if v > 0}
        
        if pie_data:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            ax_pie.pie(
                pie_data.values(), 
                labels=pie_data.keys(), 
                autopct='%1.1f%%', 
                startangle=90, 
                wedgeprops={'edgecolor': 'black'}
            )
            ax_pie.axis('equal') 
            plt.title('Total Attributed Anomalies per Candidate')
            st.pyplot(fig_pie)
        else:
            st.info("No anomalies detected to attribute.")

    # 2. STACKED BAR CHART: Type of Anomalies Per Candidate
    with col6:
        st.subheader("Anomaly Types per Candidate")
        
        # Convert the dictionary of dictionaries into a DataFrame for plotting
        chart_df = pd.DataFrame(candidate_anomaly_type_counts).T.fillna(0)
        chart_df.index.name = 'Candidate'
        
        if not chart_df.empty and chart_df.sum().sum() > 0:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 6))
            
            # Stacked bar plot
            chart_df.plot(kind='bar', stacked=True, ax=ax_bar)
            
            ax_bar.set_ylabel("Number of Attributed Anomalies")
            ax_bar.set_title("Anomaly Type Breakdown Per Candidate")
            ax_bar.tick_params(axis='x', rotation=0) 
            ax_bar.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            st.pyplot(fig_bar)
        else:
            st.info("No anomalies detected to break down by type.")
    
    # ----------------------------------
    # RL VISUALIZATIONS (Col 7 and Col 8)
    # ----------------------------------
    st.markdown("---")
    st.subheader("🤖 Reinforcement Learning (Online Learning)")
    
    # Get RL metrics
    rl_metrics = rl_agent.get_learning_metrics()
    
    # RL Learning Curve (Rewards over time)
    with col7:
        st.subheader("RL Learning Curve (Rewards)")
        if len(rl_metrics['rewards']) > 0:
            fig_rl_reward, ax_rl_reward = plt.subplots(figsize=(7, 4))
            
            # Plot rewards
            rewards_array = np.array(rl_metrics['rewards'])
            if len(rewards_array) > 0:
                # Moving average for smoother visualization
                window = min(10, len(rewards_array))
                if window > 1:
                    moving_avg = pd.Series(rewards_array).rolling(window=window, min_periods=1).mean()
                    ax_rl_reward.plot(moving_avg.values, label=f'Reward (MA-{window})', color='blue', alpha=0.7)
                    ax_rl_reward.plot(rewards_array, label='Reward (Raw)', color='lightblue', alpha=0.3, linewidth=0.5)
                else:
                    ax_rl_reward.plot(rewards_array, label='Reward', color='blue')
                
                ax_rl_reward.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax_rl_reward.set_xlabel("Learning Steps")
                ax_rl_reward.set_ylabel("Reward")
                ax_rl_reward.set_title("RL Reward Over Time (Improving Decisions)")
                ax_rl_reward.legend()
                ax_rl_reward.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_rl_reward)
                
                # Display metrics
                st.metric("Average Reward (Last 100)", f"{rl_metrics['avg_reward']:.3f}")
                st.metric("Current Epsilon (Exploration)", f"{rl_agent.epsilon:.3f}")
            else:
                st.info("Waiting for RL learning data...")
        else:
            st.info("RL agent is learning...")
    
    # RL Loss and Decision Improvement
    with col8:
        st.subheader("RL Training Loss & Decision Quality")
        if len(rl_metrics['losses']) > 0:
            fig_rl_loss, ax_rl_loss = plt.subplots(figsize=(7, 4))
            
            losses_array = np.array(rl_metrics['losses'])
            if len(losses_array) > 0:
                # Moving average for loss
                window = min(10, len(losses_array))
                if window > 1:
                    moving_avg_loss = pd.Series(losses_array).rolling(window=window, min_periods=1).mean()
                    ax_rl_loss.plot(moving_avg_loss.values, label=f'Loss (MA-{window})', color='red', alpha=0.7)
                    ax_rl_loss.plot(losses_array, label='Loss (Raw)', color='lightcoral', alpha=0.3, linewidth=0.5)
                else:
                    ax_rl_loss.plot(losses_array, label='Loss', color='red')
                
                ax_rl_loss.set_xlabel("Training Steps")
                ax_rl_loss.set_ylabel("Loss")
                ax_rl_loss.set_title("RL Training Loss (Lower = Better)")
                ax_rl_loss.legend()
                ax_rl_loss.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_rl_loss)
                
                # Display metrics
                st.metric("Average Loss (Last 100)", f"{rl_metrics['avg_loss']:.4f}")
                st.metric("Total Learning Steps", f"{rl_metrics['step_count']}")
                
                # Decision improvement metric (positive rewards percentage)
                if len(rl_metrics['rewards']) > 0:
                    positive_rewards = sum(1 for r in rl_metrics['rewards'][-50:] if r > 0)
                    total_recent = min(50, len(rl_metrics['rewards']))
                    improvement_pct = (positive_rewards / total_recent * 100) if total_recent > 0 else 0
                    st.metric("Decision Quality (Last 50)", f"{improvement_pct:.1f}% Correct")
            else:
                st.info("Waiting for RL training data...")
        else:
            st.info("RL agent is training...")
    
    # RL Prediction vs Ensemble Comparison
    if len(df) > 1 and 'rl_prediction' in df.columns:
        st.markdown("---")
        st.subheader("RL vs Ensemble Model Comparison")
        comparison_df = df[df['rl_prediction'].notna()].copy()
        if not comparison_df.empty:
            comparison_df['rl_alert'] = comparison_df['rl_prediction'].apply(lambda x: "⚠️ Anomaly" if x == 1 else "✅ Normal")
            comparison_df['agreement'] = (comparison_df['rl_prediction'] == (comparison_df['anomaly_score'] > st.session_state.threshold).astype(int))
            
            fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
            ax_comp.plot(comparison_df['time'], comparison_df['anomaly_score'], label='Ensemble Score', color='blue', alpha=0.7)
            ax_comp.plot(comparison_df['time'], comparison_df['rl_prediction'], label='RL Prediction', color='red', alpha=0.7, marker='o', markersize=3)
            ax_comp.axhline(y=st.session_state.threshold, color='gray', linestyle='--', label='Threshold')
            ax_comp.set_xlabel("Time (seconds)")
            ax_comp.set_ylabel("Score / Prediction")
            ax_comp.set_title("RL Learning to Match Ensemble (Adapting to Patterns)")
            ax_comp.legend()
            ax_comp.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_comp)
            
            # Agreement statistics
            if len(comparison_df) > 0:
                agreement_rate = comparison_df['agreement'].sum() / len(comparison_df) * 100
                st.metric("RL-Ensemble Agreement Rate", f"{agreement_rate:.1f}%")
        
    time.sleep(1)
    st.rerun()

elif st.session_state.step >= 50:
    # --- END OF SIMULATION UI ---
    st.subheader(" Simulation Complete!")
    
    # FIX: Use correct cand columns for final votes
    final_votes = df[CANDIDATES].sum().astype(int) 
    winner_name = final_votes.idxmax()
    winner_votes = final_votes.max()

    st.markdown(f"🏆 Final Winner:  <span style='color:green; font-size: 20px;'>{winner_name.upper()}</span> with *{winner_votes}* Total Cumulative Votes.", unsafe_allow_html=True)
    st.write(final_votes)
    
    # ----------------------------------
    # REPORT GENERATION
    # ----------------------------------
    st.sidebar.markdown("---")
    st.sidebar.success("Simulation Complete. Final Report Generation:")

    if st.sidebar.button(" Generate PDF Report"): 
        file = generate_pdf_report(df)
        with open(file, 'rb') as f:
            st.sidebar.download_button(
                label=f"Download {file}",
                data=f.read(),
                file_name=file,
                mime='application/pdf'
            )
        st.sidebar.success(f"Report saved and ready to download.")