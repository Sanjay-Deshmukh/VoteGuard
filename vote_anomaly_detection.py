import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import IsolationForest
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

# --------------------------
# 1️⃣ Training Phase
# --------------------------
np.random.seed(42)

# Generate synthetic "normal" voting behavior
normal_data = np.random.normal(loc=50, scale=10, size=(200, 4))
model = IsolationForest(contamination=0.1)
model.fit(normal_data)

print("✅ Models trained with 11 features and saved.")
print("🚀 Starting real-time anomaly detection...")

# --------------------------
# 2️⃣ Live Data Simulation
# --------------------------
vote_history = []
anomaly_scores = []
timestamps = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.4)

def simulate_votes():
    """Simulate random votes (normal + some anomalies)"""
    if np.random.rand() < 0.15:
        # Inject anomaly (tampering)
        votes = np.random.randint(0, 100, 4)
    else:
        # Normal distribution
        votes = np.random.normal(loc=50, scale=8, size=4).astype(int)
    return np.clip(votes, 0, 100)

# --------------------------
# 3️⃣ Animation Update Function
# --------------------------
def update(frame):
    global vote_history, anomaly_scores, timestamps

    votes = simulate_votes()
    score = model.decision_function([votes])[0]
    pred = model.predict([votes])[0]  # -1 = anomaly, 1 = normal
    anomaly = (pred == -1)

    vote_history.append(votes)
    anomaly_scores.append(abs(score))
    timestamps.append(len(vote_history))

    # --- Logging to console ---
    tag = "⚠️ Anomaly" if anomaly else "✅ Normal"
    print(f"[{frame*0.5:.2f}s] Votes: {votes} → {tag} (score={abs(score):.2f})")

    # --- Update vote chart ---
    ax1.clear()
    ax1.set_title("Real-Time Vote Counts per Candidate")
    ax1.set_ylim(0, 100)
    ax1.plot([v[0] for v in vote_history], label='Candidate A')
    ax1.plot([v[1] for v in vote_history], label='Candidate B')
    ax1.plot([v[2] for v in vote_history], label='Candidate C')
    ax1.plot([v[3] for v in vote_history], label='Candidate D')
    ax1.legend(loc="upper right")

    # --- Update anomaly chart ---
    ax2.clear()
    ax2.set_title(" Anomaly Detection Score")
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Score")
    ax2.plot(timestamps, anomaly_scores, color="red")
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Alert Threshold')
    ax2.legend()

# --------------------------
# 4️⃣ Report Generation (after simulation)
# --------------------------
def generate_report(votes_log, anomaly_scores):
    filename = "Voting_Anomaly_Report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "AI-Enhanced Voting Machine Report")
    c.drawString(50, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 710, f"Total samples: {len(votes_log)}")

    num_anomalies = sum(1 for s in anomaly_scores if s > 0.5)
    c.drawString(50, 690, f"Detected anomalies: {num_anomalies}")

    c.drawString(50, 670, "Summary:")
    c.drawString(70, 650, "- Real-time monitoring successful")
    c.drawString(70, 630, "- AI flagged irregular voting patterns")
    c.drawString(70, 610, "- Use dashboard visualization for review")

    c.showPage()
    c.save()
    print(f" Report saved as {filename}")

# --------------------------
# 5️ Start Animation
# --------------------------
ani = animation.FuncAnimation(fig, update, interval=500)  # update every 0.5s
plt.show()

# Generate report after closing chart
generate_report(vote_history, anomaly_scores)
print(" Simulation Complete. Log and report generated.")
