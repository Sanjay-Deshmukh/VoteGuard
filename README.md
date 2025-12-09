# AI Voting Anomaly Detection with RL

Real-time voting anomaly detection system with Deep Q-Network (DQN) reinforcement learning that learns and adapts to new tampering patterns.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn torch reportlab joblib
```

### 2. Run the Dashboard

```bash
streamlit run vote_dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 📋 What You'll See

### Dashboard Features:
- **Live Vote Trends**: Real-time vote counts for 4 candidates
- **Anomaly Detection**: Ensemble models (Isolation Forest, LOF, One-Class SVM) + RL Agent
- **RL Predictions**: Reinforcement learning agent making real-time predictions
- **RL Learning Curve**: Visualizes how the RL agent improves over time
- **RL Training Metrics**: Loss, rewards, and decision quality
- **Anomaly Attribution**: Which candidates are affected by anomalies

### Controls (Sidebar):
- **▶ Start/Resume**: Begin the simulation
- **🛑 Pause**: Pause the simulation
- **Alert Threshold**: Adjust sensitivity (0.0 - 1.0)
- **💾 Save RL Model**: Manually save the RL model

## 🤖 RL Agent Features

- **Online Learning**: Learns from each new data point in real-time
- **Pattern Adaptation**: Adapts to new tampering patterns automatically
- **Model Persistence**: Auto-saves every 10 steps, manual save available
- **Visual Feedback**: See rewards, losses, and learning progress

## 📁 Files

- `vote_dashboard.py` - Main Streamlit dashboard
- `rl_anomaly_detector.py` - DQN-based RL agent
- `requirements.txt` - Python dependencies
- `rl_dqn_model.pth` - Saved RL model (created after first run)
- `iso.pkl`, `lof.pkl`, `ocsvm.pkl` - Ensemble models

## 🎯 How It Works

1. **Data Simulation**: Simulates voting data with normal and anomalous patterns
2. **Feature Extraction**: Extracts 10 features (total votes, differences, ratios)
3. **Ensemble Detection**: Three ML models vote on anomalies
4. **RL Learning**: DQN agent learns to predict anomalies, rewarded by ensemble agreement
5. **Real-time Updates**: Dashboard updates every second with new predictions

## 💡 Tips

- Watch the RL Learning Curve to see the agent improve over time
- The RL agent starts with high exploration (ε=1.0) and gradually becomes more confident
- Check the "RL vs Ensemble Comparison" to see agreement rates
- The model saves automatically, so you can stop and resume learning

## 🔧 Troubleshooting

**Import Error**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Port Already in Use**: Streamlit will try a different port automatically, or specify one:
```bash
streamlit run vote_dashboard.py --server.port 8502
```

**RL Model Not Loading**: The model will be created on first run. If you see errors, delete `rl_dqn_model.pth` and restart.

