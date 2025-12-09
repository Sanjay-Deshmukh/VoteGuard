# Flask Dashboard - Voting Anomaly Detection with RL

Modern dark-themed Flask dashboard inspired by Bhacemp template, featuring real-time RL learning and anomaly detection.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Flask Application

```bash
python app.py
```

The dashboard will be available at `http://localhost:5000`

## 📋 Features

### Dashboard Components:
- **Dark Theme UI**: Modern, professional design matching Bhacemp template
- **Sidebar Navigation**: Quick access to different sections
- **Real-time Statistics**: Live KPIs with mini charts
- **Live Vote Trends**: Real-time visualization of candidate votes
- **Anomaly Detection**: Ensemble models + RL agent predictions
- **RL Learning Curves**: Visualize rewards and training loss
- **Model Comparison**: Compare all detection models
- **Anomaly Log Table**: Detailed log of detected anomalies
- **Alert System**: Color-coded alerts in sidebar

### RL Agent Features:
- **Online Learning**: Learns from each data point in real-time
- **Pattern Adaptation**: Adapts to new tampering patterns
- **Visual Feedback**: See learning progress through charts
- **Model Persistence**: Auto-saves every 10 steps

## 🎨 Design Features

- **Dark Theme**: Professional dark color scheme
- **Responsive Layout**: Works on different screen sizes
- **Real-time Updates**: Auto-refreshes every second
- **Interactive Charts**: Chart.js powered visualizations
- **Color-coded Alerts**: Visual alert system in sidebar

## 📁 Project Structure

```
e:\model\
├── app.py                 # Flask application
├── templates/
│   └── dashboard.html     # Main dashboard template
├── static/
│   ├── style.css         # Dark theme styling
│   └── script.js         # Real-time updates & charts
├── rl_anomaly_detector.py # RL agent implementation
└── requirements.txt       # Dependencies
```

## 🔧 API Endpoints

- `GET /` - Main dashboard page
- `GET /api/status` - Get current system status
- `GET /api/data` - Get all data for charts
- `GET /api/alerts` - Get recent alerts
- `POST /api/control` - Control simulation (start/stop/threshold)

## 💡 Usage

1. **Start the Application**: Run `python app.py`
2. **Open Browser**: Navigate to `http://localhost:5000`
3. **Start Simulation**: Click "▶ Start Simulation" button
4. **Monitor**: Watch real-time updates, charts, and RL learning
5. **Adjust Threshold**: Use the slider to adjust anomaly sensitivity
6. **View Alerts**: Check sidebar for active alerts
7. **Review Log**: Scroll to see detailed anomaly log table

## 🎯 Key Differences from Streamlit Version

- **Flask-based**: More control over UI/UX
- **Dark Theme**: Professional Bhacemp-inspired design
- **Better Performance**: More efficient for production
- **Customizable**: Easy to extend and modify
- **RESTful API**: Clean API endpoints for data access

## 🔧 Configuration

Edit `app.py` to customize:
- Port number (default: 5000)
- Update interval (default: 1 second)
- Max simulation steps (default: 200)
- Model paths

## 📊 Dashboard Sections

1. **Statistics Overview**: 4 KPI cards with mini charts
2. **Vote Trends**: Line chart showing all candidates
3. **Anomaly Scores**: Detection scores over time
4. **RL Learning**: Rewards and loss curves
5. **Model Comparison**: All models side-by-side
6. **Anomaly Log**: Detailed table of all anomalies

Enjoy your modern anomaly detection dashboard! 🎉
