// Dashboard JavaScript for Real-time Updates
let charts = {};
let updateInterval;

// Register datalabels plugin globally if available
if (typeof ChartDataLabels !== 'undefined') {
    Chart.register(ChartDataLabels);
}

// Initialize Charts
function initCharts() {
    // Samples Chart
    const samplesCtx = document.getElementById('samples-chart');
    if (samplesCtx) {
        charts.samples = new Chart(samplesCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Samples',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { display: false },
                    x: { display: false }
                }
            }
        });
    }

    // Anomalies Chart
    const anomaliesCtx = document.getElementById('anomalies-chart');
    if (anomaliesCtx) {
        charts.anomalies = new Chart(anomaliesCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Anomalies',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { display: false },
                    x: { display: false }
                }
            }
        });
    }

    // Reward Chart
    const rewardCtx = document.getElementById('reward-chart');
    if (rewardCtx) {
        charts.reward = new Chart(rewardCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Reward',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { display: false },
                    x: { display: false }
                }
            }
        });
    }

    // Vote Trends Chart
    const voteTrendsCtx = document.getElementById('vote-trends-chart');
    if (voteTrendsCtx) {
        charts.voteTrends = new Chart(voteTrendsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Candidate 1',
                        data: [],
                        borderColor: '#14b8a6',
                        backgroundColor: 'rgba(20, 184, 166, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Candidate 2',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Candidate 3',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Candidate 4',
                        data: [],
                        borderColor: '#ec4899',
                        backgroundColor: 'rgba(236, 72, 153, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' },
                    datalabels: { display: false },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // Anomaly Scores Chart
    const anomalyScoresCtx = document.getElementById('anomaly-scores-chart');
    if (anomalyScoresCtx) {
        charts.anomalyScores = new Chart(anomalyScoresCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Anomaly Score',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    datalabels: { display: false },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // RL Rewards Chart
    const rlRewardsCtx = document.getElementById('rl-rewards-chart');
    if (rlRewardsCtx) {
        charts.rlRewards = new Chart(rlRewardsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Reward',
                    data: [],
                    borderColor: '#14b8a6',
                    backgroundColor: 'rgba(20, 184, 166, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    datalabels: { display: false },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // RL Loss Chart
    const rlLossCtx = document.getElementById('rl-loss-chart');
    if (rlLossCtx) {
        charts.rlLoss = new Chart(rlLossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    datalabels: { display: false },
                    tooltip: {
                        enabled: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });
    }

    // Model Scores Chart - RL vs Ensemble Comparison
    const modelScoresCtx = document.getElementById('model-scores-chart');
    if (modelScoresCtx) {
        charts.modelScores = new Chart(modelScoresCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Ensemble Model (ISO + LOF + OCSVM)',
                        data: [],
                        borderColor: '#14b8a6',
                        backgroundColor: 'rgba(20, 184, 166, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'RL Prediction',
                        data: [],
                        borderColor: '#ec4899',
                        backgroundColor: 'rgba(236, 72, 153, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        title: {
                            display: true,
                            text: 'Anomaly Score'
                        }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });
    }

    // Candidate Anomalies Pie Chart
    const candidatePieCtx = document.getElementById('candidate-pie-chart');
    if (candidatePieCtx) {
        charts.candidatePie = new Chart(candidatePieCtx, {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#3b82f6', // Blue
                        '#f59e0b', // Orange
                        '#10b981', // Green
                        '#ef4444'  // Red
                    ],
                    borderColor: '#1e293b',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                label += context.parsed + ' (' + percentage + '%)';
                                return label;
                            }
                        }
                    },
                    datalabels: {
                        color: '#ffffff',
                        font: {
                            weight: 'bold',
                            size: 14
                        },
                        formatter: function(value, context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            if (total === 0) return '';
                            const percentage = ((value / total) * 100).toFixed(1);
                            return percentage + '%';
                        },
                        display: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            if (total === 0) return false;
                            const percentage = ((context.dataset.data[context.dataIndex] / total) * 100);
                            // Only show percentage if segment is large enough (> 5%)
                            return percentage > 5;
                        }
                    }
                }
            }
        });
    }

    // Candidate Anomaly Types Stacked Bar Chart
    const candidateBarCtx = document.getElementById('candidate-bar-chart');
    if (candidateBarCtx) {
        charts.candidateBar = new Chart(candidateBarCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        title: {
                            display: true,
                            text: 'Number of Attributed Anomalies'
                        }
                    }
                }
            }
        });
    }
}

// Update Dashboard Data
async function updateDashboard() {
    try {
        // Get status
        const statusRes = await fetch('/api/status');
        const status = await statusRes.json();
        
        // Update stats
        document.getElementById('stat-samples').textContent = status.stats.total_samples;
        document.getElementById('stat-anomalies').textContent = status.stats.anomalies;
        
        // Check if simulation is complete
        if (status.final_results && status.final_results.is_complete) {
            showSimulationComplete(status.final_results);
        } else {
            hideSimulationComplete();
        }
        
        // Get data
        const dataRes = await fetch('/api/data');
        const data = await dataRes.json();
        
        // Update RL metrics
        if (data.rl_metrics) {
            document.getElementById('stat-rl-steps').textContent = data.rl_metrics.step_count || 0;
            document.getElementById('rl-epsilon').textContent = `ε: ${(data.rl_metrics.epsilon || [1.0])[data.rl_metrics.epsilon?.length - 1]?.toFixed(3) || '1.000'}`;
            document.getElementById('stat-rl-reward').textContent = (data.rl_metrics.avg_reward || 0).toFixed(3);
        }
        
        // Update charts
        if (data.vote_trends && charts.voteTrends) {
            charts.voteTrends.data.labels = data.vote_trends.time.map(t => t.toFixed(1));
            charts.voteTrends.data.datasets[0].data = data.vote_trends.cand1;
            charts.voteTrends.data.datasets[1].data = data.vote_trends.cand2;
            charts.voteTrends.data.datasets[2].data = data.vote_trends.cand3;
            charts.voteTrends.data.datasets[3].data = data.vote_trends.cand4;
            charts.voteTrends.update('none');
        }
        
        if (data.anomaly_scores && charts.anomalyScores) {
            charts.anomalyScores.data.labels = data.anomaly_scores.time.map(t => t.toFixed(1));
            charts.anomalyScores.data.datasets[0].data = data.anomaly_scores.scores;
            charts.anomalyScores.update('none');
        }
        
        if (data.rl_rewards && charts.rlRewards) {
            charts.rlRewards.data.labels = data.rl_rewards.map((_, i) => i);
            charts.rlRewards.data.datasets[0].data = data.rl_rewards;
            charts.rlRewards.update('none');
        }
        
        if (data.rl_losses && charts.rlLoss) {
            charts.rlLoss.data.labels = data.rl_losses.map((_, i) => i);
            charts.rlLoss.data.datasets[0].data = data.rl_losses;
            charts.rlLoss.update('none');
        }
        
        if (data.model_scores && charts.modelScores) {
            charts.modelScores.data.labels = data.model_scores.time.map(t => t.toFixed(1));
            charts.modelScores.data.datasets[0].data = data.model_scores.ensemble; // Ensemble Model
            charts.modelScores.data.datasets[1].data = data.model_scores.rl_pred; // RL Prediction
            charts.modelScores.update('none');
        }
        
        // Update mini charts
        if (data.vote_trends && charts.samples) {
            const samples = data.vote_trends.time.length;
            charts.samples.data.labels = Array(samples).fill('');
            charts.samples.data.datasets[0].data = Array(samples).fill(samples);
            charts.samples.update('none');
        }
        
        if (data.anomaly_scores && charts.anomalies) {
            const anomalyCount = data.anomaly_scores.alerts.filter(a => a === 'Anomaly').length;
            charts.anomalies.data.labels = Array(data.anomaly_scores.time.length).fill('');
            charts.anomalies.data.datasets[0].data = Array(data.anomaly_scores.time.length).fill(anomalyCount);
            charts.anomalies.update('none');
        }
        
        // Update alerts
        updateAlerts();
        
        // Update anomaly table
        updateAnomalyTable();
        
        // Update candidate anomaly charts
        updateCandidateCharts();
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

// Update Candidate Anomaly Charts
async function updateCandidateCharts() {
    try {
        const res = await fetch('/api/candidate-anomalies');
        const data = await res.json();
        
        // Update Pie Chart
        if (charts.candidatePie && data.pie_data && Object.keys(data.pie_data).length > 0) {
            charts.candidatePie.data.labels = Object.keys(data.pie_data);
            charts.candidatePie.data.datasets[0].data = Object.values(data.pie_data);
            charts.candidatePie.update('none');
        } else if (charts.candidatePie) {
            charts.candidatePie.data.labels = [];
            charts.candidatePie.data.datasets[0].data = [];
            charts.candidatePie.update('none');
        }
        
        // Update Stacked Bar Chart
        if (charts.candidateBar && data.bar_data && Object.keys(data.bar_data).length > 0) {
            // Get all unique anomaly types
            const allTypes = new Set();
            Object.values(data.bar_data).forEach(types => {
                Object.keys(types).forEach(type => allTypes.add(type));
            });
            
            const typeColors = {
                'Sudden Spike': '#ef4444',
                'Sudden Drop/Trough': '#f59e0b',
                'Total Vote Overload': '#ec4899',
                'Vote System Stall': '#8b5cf6',
                'Dominant Ratio Injection': '#10b981',
                'Gradual Drift': '#14b8a6',
                'Duplicate Pattern': '#3b82f6',
                'Unclassified Anomaly': '#64748b',
                'Not Applicable': '#94a3b8'
            };
            
            // Create datasets for each anomaly type
            const datasets = Array.from(allTypes).map(type => {
                return {
                    label: type,
                    data: Object.keys(data.bar_data).map(cand => data.bar_data[cand][type] || 0),
                    backgroundColor: typeColors[type] || '#64748b'
                };
            });
            
            charts.candidateBar.data.labels = Object.keys(data.bar_data);
            charts.candidateBar.data.datasets = datasets;
            charts.candidateBar.update('none');
        } else if (charts.candidateBar) {
            charts.candidateBar.data.labels = [];
            charts.candidateBar.data.datasets = [];
            charts.candidateBar.update('none');
        }
    } catch (error) {
        console.error('Error updating candidate charts:', error);
    }
}

// Update Alerts
async function updateAlerts() {
    try {
        const res = await fetch('/api/alerts');
        const data = await res.json();
        
        const container = document.getElementById('alerts-container');
        if (!container) return;
        
        if (data.alerts.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); font-size: 12px;">No active alerts</p>';
            return;
        }
        
        container.innerHTML = data.alerts.slice(-3).map(alert => {
            let typeClass = 'exploit';
            if (alert.type.includes('Spike') || alert.type.includes('Overload')) {
                typeClass = 'malware';
            } else if (alert.type.includes('Drop') || alert.type.includes('Stall')) {
                typeClass = 'phishing';
            }
            
            return `
                <div class="alert-card ${typeClass}">
                    <span class="alert-icon">⚠️</span>
                    <div class="alert-info">
                        <div class="alert-type">${alert.type}</div>
                        <div class="alert-id">Time: ${alert.time.toFixed(1)}s</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error updating alerts:', error);
    }
}

// Update Anomaly Table
async function updateAnomalyTable() {
    try {
        const res = await fetch('/api/alerts');
        const data = await res.json();
        
        const tbody = document.getElementById('anomaly-table-body');
        if (!tbody) return;
        
        if (data.alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No anomalies detected yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = data.alerts.slice(-10).reverse().map(alert => {
            return `
                <tr>
                    <td>${alert.time.toFixed(2)}s</td>
                    <td>${alert.type}</td>
                    <td>${alert.score.toFixed(3)}</td>
                    <td>${alert.candidates.cand1}</td>
                    <td>${alert.candidates.cand2}</td>
                    <td>${alert.candidates.cand3}</td>
                    <td>${alert.candidates.cand4}</td>
                    <td><span class="status-badge ${alert.score > 0.5 ? 'anomaly' : 'normal'}">${alert.score > 0.5 ? 'Anomaly' : 'Normal'}</span></td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error updating table:', error);
    }
}

// Control Handlers
document.getElementById('start-btn')?.addEventListener('click', async () => {
    await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start' })
    });
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
});

document.getElementById('stop-btn')?.addEventListener('click', async () => {
    await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' })
    });
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
});

document.getElementById('threshold-slider')?.addEventListener('input', async (e) => {
    const value = parseFloat(e.target.value);
    document.getElementById('threshold-value').textContent = value.toFixed(2);
    await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'threshold', threshold: value })
    });
});

// Export Functions
document.getElementById('export-pdf-btn')?.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/export/pdf');
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Voting_Anomaly_Report_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            showNotification('PDF report generated successfully!', 'success');
        } else {
            const error = await response.json();
            showNotification(error.error || 'Failed to generate PDF', 'error');
        }
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Error generating PDF', 'error');
    }
});

document.getElementById('export-charts-btn')?.addEventListener('click', () => {
    try {
        const chartMap = {
            'vote-trends-chart': 'voteTrends',
            'anomaly-scores-chart': 'anomalyScores',
            'rl-rewards-chart': 'rlRewards',
            'rl-loss-chart': 'rlLoss',
            'model-scores-chart': 'modelScores',
            'candidate-pie-chart': 'candidatePie',
            'candidate-bar-chart': 'candidateBar'
        };
        
        let exported = 0;
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        
        Object.entries(chartMap).forEach(([chartId, chartKey]) => {
            const canvas = document.getElementById(chartId);
            if (canvas && charts[chartKey]) {
                try {
                    const url = charts[chartKey].toBase64Image('image/png', 1.0);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${chartId}_${timestamp}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    exported++;
                } catch (e) {
                    console.warn(`Failed to export ${chartId}:`, e);
                }
            }
        });
        
        if (exported > 0) {
            showNotification(`Exported ${exported} chart(s) successfully!`, 'success');
        } else {
            showNotification('No charts available to export', 'error');
        }
    } catch (error) {
        console.error('Chart export error:', error);
        showNotification('Error exporting charts', 'error');
    }
});

// Simulation Complete Display
function showSimulationComplete(finalResults) {
    const section = document.getElementById('simulation-complete-section');
    if (!section) return;
    
    section.style.display = 'block';
    
    // Update winner
    if (finalResults.winner) {
        document.getElementById('winner-name').textContent = finalResults.winner.toUpperCase();
        document.getElementById('winner-votes').textContent = finalResults.winner_votes;
    }
    
    // Update results table
    const tbody = document.getElementById('results-table-body');
    if (tbody && finalResults.final_votes) {
        // Sort by votes (descending)
        const sortedCandidates = Object.entries(finalResults.final_votes)
            .sort((a, b) => b[1] - a[1]);
        
        tbody.innerHTML = sortedCandidates.map(([cand, votes]) => {
            const isWinner = cand === finalResults.winner;
            return `
                <tr ${isWinner ? 'style="background: rgba(16, 185, 129, 0.1);"' : ''}>
                    <td>${cand.toUpperCase()}</td>
                    <td>${votes}</td>
                </tr>
            `;
        }).join('');
    }
}

function hideSimulationComplete() {
    const section = document.getElementById('simulation-complete-section');
    if (section) {
        section.style.display = 'none';
    }
}

// Notification System
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    updateDashboard();
    updateInterval = setInterval(updateDashboard, 1000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});

