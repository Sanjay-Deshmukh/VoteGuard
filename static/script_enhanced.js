/**
 * VoteGuard Enhanced Dashboard - JavaScript
 * With WebSockets, 3D Visualization, Heatmaps, Animations, and More
 */

// ============= GLOBAL STATE =============
const state = {
    charts: {},
    socket: null,
    is3DInitialized: false,
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    particles: [],
    anomalyCount: 0,
    autoRotate: true,
    fullscreenChart: null,
    theme: 'dark',
    notificationCount: 0
};

// Chart.js global defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.1)';
Chart.defaults.font.family = 'Inter, sans-serif';

// Color palette
const colors = {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    teal: '#14b8a6',
    blue: '#3b82f6',
    green: '#10b981',
    orange: '#f59e0b',
    pink: '#ec4899',
    red: '#ef4444',
    purple: '#a855f7',
    candidates: ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6']
};

// ============= INITIALIZATION =============
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSocket();
    initCharts();
    initEventListeners();
    // init3DVisualization(); // Removed - 3D section disabled
    initHeatmap();
    initDecisionTree();
    startDataPolling();

    // Initial data fetch
    updateDashboard();
    updateCandidateCharts();
    updateConfidenceGauges();
    updatePrediction();
});

// ============= THEME MANAGEMENT =============
function initTheme() {
    const savedTheme = localStorage.getItem('voteguard-theme') || 'dark';
    setTheme(savedTheme);

    document.getElementById('theme-toggle')?.addEventListener('click', () => {
        const newTheme = state.theme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
        localStorage.setItem('voteguard-theme', newTheme);
    });
}

function setTheme(theme) {
    state.theme = theme;
    document.body.classList.remove('dark-mode', 'light-mode');
    document.body.classList.add(`${theme}-mode`);

    // Update chart colors
    const textColor = theme === 'dark' ? '#f8fafc' : '#1e293b';
    const gridColor = theme === 'dark' ? 'rgba(99, 102, 241, 0.1)' : 'rgba(99, 102, 241, 0.15)';

    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = gridColor;

    // Update existing charts
    Object.values(state.charts).forEach(chart => {
        if (chart) chart.update();
    });
}

// ============= WEBSOCKET =============
function initSocket() {
    try {
        state.socket = io();

        state.socket.on('connect', () => {
            console.log('🔌 WebSocket connected');
            updateLiveIndicator(true);
        });

        state.socket.on('disconnect', () => {
            console.log('❌ WebSocket disconnected');
            updateLiveIndicator(false);
        });

        state.socket.on('data_update', (data) => {
            handleDataUpdate(data);
        });

        state.socket.on('anomaly_alert', (data) => {
            handleAnomalyAlert(data);
        });

        state.socket.on('simulation_started', (data) => {
            // Notification removed - simulation start is visible in UI
            console.log('Simulation started:', data);
        });

        state.socket.on('simulation_complete', (data) => {
            handleSimulationComplete(data);
        });
    } catch (err) {
        console.error('WebSocket initialization failed:', err);
    }
}

function updateLiveIndicator(isLive) {
    const indicator = document.getElementById('live-indicator');
    const statusText = indicator?.querySelector('.status-text');

    if (indicator) {
        indicator.classList.toggle('active', isLive);
    }
    if (statusText) {
        statusText.textContent = isLive ? 'Live' : 'Offline';
    }
}

function handleDataUpdate(data) {
    // Update stats
    if (data.step !== undefined) {
        document.getElementById('stat-steps').textContent = data.step;
    }

    if (data.latest) {
        // Check for new anomaly
        if (data.anomaly_detected) {
            state.anomalyCount++;
            updateNotificationBadge();
        }
    }
}

function handleAnomalyAlert(data) {
    // Add to alerts sidebar only (no popup notification)
    addAlertToSidebar(data);
}

function handleSimulationComplete(data) {
    showSimulationComplete(data);
}

// ============= CHARTS INITIALIZATION =============
function initCharts() {
    initVoteTrendsChart();
    initAnomalyScoresChart();
    initRLChartsGroup();
    initTrendChart();
    initModelComparisonChart();
    initCandidatePieChart();
    initCandidateBarChart();
    initSparklines();
}

function initVoteTrendsChart() {
    const ctx = document.getElementById('vote-trends-chart')?.getContext('2d');
    if (!ctx) return;

    state.charts.voteTrends = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Cand 1', data: [], borderColor: colors.candidates[0], backgroundColor: `${colors.candidates[0]}20`, tension: 0.4, fill: true },
                { label: 'Cand 2', data: [], borderColor: colors.candidates[1], backgroundColor: `${colors.candidates[1]}20`, tension: 0.4, fill: true },
                { label: 'Cand 3', data: [], borderColor: colors.candidates[2], backgroundColor: `${colors.candidates[2]}20`, tension: 0.4, fill: true },
                { label: 'Cand 4', data: [], borderColor: colors.candidates[3], backgroundColor: `${colors.candidates[3]}20`, tension: 0.4, fill: true }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, padding: 20 } },
                tooltip: { backgroundColor: 'rgba(17, 24, 39, 0.9)', titleColor: '#fff', bodyColor: '#94a3b8', borderColor: colors.primary, borderWidth: 1 }
            },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, grid: { color: 'rgba(99, 102, 241, 0.1)' } }
            }
        }
    });
}

function initAnomalyScoresChart() {
    const ctx = document.getElementById('anomaly-scores-chart')?.getContext('2d');
    if (!ctx) return;

    state.charts.anomalyScores = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Anomaly Score',
                data: [],
                borderColor: colors.red,
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            plugins: {
                legend: { display: false },
                annotation: { annotations: {} }
            },
            scales: {
                x: { grid: { display: false } },
                y: { min: 0, max: 1, grid: { color: 'rgba(99, 102, 241, 0.1)' } }
            }
        }
    });
}

function initRLChartsGroup() {
    // RL Rewards Chart
    const rewardsCtx = document.getElementById('rl-rewards-chart')?.getContext('2d');
    if (rewardsCtx) {
        state.charts.rlRewards = new Chart(rewardsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Reward',
                    data: [],
                    borderColor: colors.green,
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
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(99, 102, 241, 0.1)' } }
                }
            }
        });
    }

    // RL Loss Chart
    const lossCtx = document.getElementById('rl-loss-chart')?.getContext('2d');
    if (lossCtx) {
        state.charts.rlLoss = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: colors.orange,
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(99, 102, 241, 0.1)' } }
                }
            }
        });
    }
}

function initTrendChart() {
    const ctx = document.getElementById('trend-chart')?.getContext('2d');
    if (!ctx) return;

    state.charts.trend = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Raw Score', data: [], borderColor: colors.orange, backgroundColor: 'rgba(245, 158, 11, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true },
                { label: 'MA-5', data: [], borderColor: colors.teal, borderWidth: 3, pointRadius: 0, tension: 0.4 },
                { label: 'MA-10', data: [], borderColor: colors.purple, borderWidth: 3, pointRadius: 0, tension: 0.4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
                y: { beginAtZero: true, grid: { color: 'rgba(99, 102, 241, 0.1)' } }
            }
        }
    });
}

function initModelComparisonChart() {
    const ctx = document.getElementById('model-comparison-chart')?.getContext('2d');
    if (!ctx) return;

    state.charts.modelComparison = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Isolation Forest', data: [], borderColor: colors.red, borderWidth: 2, pointRadius: 0 },
                { label: 'LOF', data: [], borderColor: colors.blue, borderWidth: 2, pointRadius: 0 },
                { label: 'One-Class SVM', data: [], borderColor: colors.green, borderWidth: 2, pointRadius: 0 },
                { label: 'RL Agent', data: [], borderColor: colors.orange, borderWidth: 2, pointRadius: 0, borderDash: [5, 5] }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { usePointStyle: true } } },
            scales: {
                x: { grid: { display: false } },
                y: { min: 0, max: 1, grid: { color: 'rgba(99, 102, 241, 0.1)' } }
            }
        }
    });
}

function initCandidatePieChart() {
    const ctx = document.getElementById('candidate-pie-chart')?.getContext('2d');
    if (!ctx) return;

    state.charts.candidatePie = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: colors.candidates,
                borderWidth: 2,
                borderColor: 'rgba(0, 0, 0, 0.1)',
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '50%',
            plugins: {
                legend: { position: 'right', labels: { usePointStyle: true, padding: 20 } },
                datalabels: {
                    color: '#fff',
                    font: {
                        weight: 'bold',
                        size: 14
                    },
                    formatter: (value, context) => {
                        const dataset = context.chart.data.datasets[0];
                        const total = dataset.data.reduce((acc, val) => acc + val, 0);
                        if (total === 0) return '';
                        const percentage = ((value / total) * 100).toFixed(1);
                        return percentage + '%';
                    },
                    display: (context) => {
                        const dataset = context.chart.data.datasets[0];
                        const total = dataset.data.reduce((acc, val) => acc + val, 0);
                        const value = dataset.data[context.dataIndex];
                        return (value / total) > 0.05; // Only show if > 5%
                    }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

function initCandidateBarChart() {
    const ctx = document.getElementById('candidate-bar-chart')?.getContext('2d');
    if (!ctx) return;

    state.charts.candidateBar = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { usePointStyle: true } } },
            scales: {
                x: { stacked: true, grid: { display: false } },
                y: { stacked: true, grid: { color: 'rgba(99, 102, 241, 0.1)' } }
            }
        }
    });
}

function initSparklines() {
    // Time sparkline
    const timeCtx = document.getElementById('time-sparkline')?.getContext('2d');
    if (timeCtx) {
        state.charts.timeSparkline = new Chart(timeCtx, {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: colors.teal, borderWidth: 2, pointRadius: 0, tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } }
        });
    }

    // Reward sparkline
    const rewardCtx = document.getElementById('reward-sparkline')?.getContext('2d');
    if (rewardCtx) {
        state.charts.rewardSparkline = new Chart(rewardCtx, {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: colors.orange, borderWidth: 2, pointRadius: 0, tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } }
        });
    }
}

// ============= 3D VISUALIZATION =============
function init3DVisualization() {
    const container = document.getElementById('three-d-container');
    if (!container) return;

    // Scene setup
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x0a0e1a);

    // Camera
    state.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    state.camera.position.set(100, 100, 100);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(state.renderer.domElement);

    // Controls (manual implementation since OrbitControls might not load)
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    container.addEventListener('mousedown', () => { isDragging = true; });
    container.addEventListener('mouseup', () => { isDragging = false; });
    container.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const deltaMove = { x: e.offsetX - previousMousePosition.x, y: e.offsetY - previousMousePosition.y };
            state.camera.position.x += deltaMove.x * 0.5;
            state.camera.position.y -= deltaMove.y * 0.5;
            state.camera.lookAt(0, 0, 0);
        }
        previousMousePosition = { x: e.offsetX, y: e.offsetY };
    });

    // Grid helper
    const gridHelper = new THREE.GridHelper(200, 20, 0x6366f1, 0x1e293b);
    state.scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(50);
    state.scene.add(axesHelper);

    // Add labels
    addAxisLabels();

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    state.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(100, 100, 50);
    state.scene.add(directionalLight);

    state.is3DInitialized = true;
    animate3D();

    // Resize handler
    window.addEventListener('resize', () => {
        if (state.camera && state.renderer && container) {
            state.camera.aspect = container.clientWidth / container.clientHeight;
            state.camera.updateProjectionMatrix();
            state.renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });
}

function addAxisLabels() {
    // Create text sprites for axis labels (simplified)
    const labels = ['Cand1 Votes', 'Cand2 Votes', 'Time'];
    const positions = [[110, 0, 0], [0, 110, 0], [0, 0, 110]];

    // Using simple line indicators instead of text for simplicity
    positions.forEach((pos, i) => {
        const geometry = new THREE.SphereGeometry(3, 16, 16);
        const material = new THREE.MeshBasicMaterial({ color: colors.candidates[i] });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(...pos);
        state.scene.add(sphere);
    });
}

function animate3D() {
    requestAnimationFrame(animate3D);

    if (state.autoRotate && state.camera) {
        state.camera.position.x = 150 * Math.sin(Date.now() * 0.0003);
        state.camera.position.z = 150 * Math.cos(Date.now() * 0.0003);
        state.camera.lookAt(0, 0, 0);
    }

    // Animate particles
    state.particles.forEach(particle => {
        if (particle.userData.isAnomaly) {
            particle.scale.setScalar(1 + 0.1 * Math.sin(Date.now() * 0.005));
        }
    });

    if (state.renderer && state.scene && state.camera) {
        state.renderer.render(state.scene, state.camera);
    }
}

async function update3DTrajectories() {
    if (!state.is3DInitialized) return;

    try {
        const response = await fetch('/api/3d-trajectories');
        const data = await response.json();

        // Clear old particles
        state.particles.forEach(p => state.scene.remove(p));
        state.particles = [];

        // Add new particles
        data.trajectories?.forEach((point, i) => {
            const geometry = new THREE.SphereGeometry(point.anomaly ? 4 : 2, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: point.anomaly ? 0xef4444 : 0x10b981,
                emissive: point.anomaly ? 0xef4444 : 0x000000,
                emissiveIntensity: point.anomaly ? 0.5 : 0
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(
                (point.votes[0] || 0) - 50,
                (point.votes[1] || 0) - 50,
                point.time * 2
            );
            sphere.userData.isAnomaly = point.anomaly;

            state.scene.add(sphere);
            state.particles.push(sphere);
        });

        // Draw connecting lines
        if (data.trajectories?.length > 1) {
            const points = data.trajectories.map(p =>
                new THREE.Vector3((p.votes[0] || 0) - 50, (p.votes[1] || 0) - 50, p.time * 2)
            );
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0x6366f1, opacity: 0.5, transparent: true });
            const line = new THREE.Line(lineGeometry, lineMaterial);
            state.scene.add(line);
            state.particles.push(line);
        }
    } catch (err) {
        console.error('Error updating 3D trajectories:', err);
    }
}

// ============= HEATMAP =============
function initHeatmap() {
    updateHeatmap();
}

async function updateHeatmap() {
    const container = document.getElementById('heatmap');
    if (!container) return;

    try {
        const response = await fetch('/api/heatmap');
        const data = await response.json();

        // Create canvas-based heatmap
        let canvas = container.querySelector('canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            container.appendChild(canvas);
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!data.data || data.data.length === 0) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = '14px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('No data available', canvas.width / 2, canvas.height / 2);
            return;
        }

        const cellWidth = canvas.width / Math.min(data.data.length, 50);
        const cellHeight = canvas.height / 4; // 4 candidates

        data.data.slice(-50).forEach((point, timeIdx) => {
            Object.entries(point.candidates || {}).forEach(([cand, value], candIdx) => {
                const intensity = Math.min(Math.abs(value) / 50, 1);
                const hue = value > 0 ? 0 : 120; // Red for positive, green for negative
                ctx.fillStyle = `hsla(${hue}, 80%, 50%, ${intensity})`;
                ctx.fillRect(timeIdx * cellWidth, candIdx * cellHeight, cellWidth, cellHeight);
            });
        });

        // Add time axis
        ctx.fillStyle = '#64748b';
        ctx.font = '10px Inter';
        ctx.fillText('Time →', canvas.width - 40, canvas.height - 5);

    } catch (err) {
        console.error('Error updating heatmap:', err);
    }
}

// ============= DECISION TREE =============
async function initDecisionTree() {
    updateDecisionTree();
}

async function updateDecisionTree() {
    const container = document.getElementById('decision-tree');
    if (!container) return;

    try {
        const response = await fetch('/api/decision-tree');
        const data = await response.json();

        if (!data.tree) {
            container.innerHTML = '<p class="empty-state">No decision data available</p>';
            return;
        }

        // Clear container
        container.innerHTML = '';

        // Create tree structure using D3
        const width = container.clientWidth;
        const height = 280;

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', `translate(${width / 2}, 40)`);

        // Draw root
        g.append('circle')
            .attr('r', 30)
            .attr('fill', colors.primary)
            .attr('class', 'tree-node');

        g.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.3em')
            .attr('fill', 'white')
            .attr('font-size', '10px')
            .text('Ensemble');

        // Draw children
        const childWidth = width / (data.tree.children?.length || 4);
        data.tree.children?.forEach((child, i) => {
            const x = (i - (data.tree.children.length - 1) / 2) * childWidth;
            const y = 100;

            // Line to child
            g.append('line')
                .attr('x1', 0).attr('y1', 30)
                .attr('x2', x).attr('y2', y - 25)
                .attr('stroke', child.color)
                .attr('stroke-width', 2)
                .attr('opacity', 0.6);

            // Child node
            g.append('circle')
                .attr('cx', x).attr('cy', y)
                .attr('r', 25)
                .attr('fill', child.color);

            // Child label
            g.append('text')
                .attr('x', x).attr('y', y)
                .attr('text-anchor', 'middle')
                .attr('dy', '0.3em')
                .attr('fill', 'white')
                .attr('font-size', '9px')
                .text(child.value);

            // Child name
            g.append('text')
                .attr('x', x).attr('y', y + 45)
                .attr('text-anchor', 'middle')
                .attr('fill', '#94a3b8')
                .attr('font-size', '11px')
                .text(child.name);

            // Description
            g.append('text')
                .attr('x', x).attr('y', y + 60)
                .attr('text-anchor', 'middle')
                .attr('fill', '#64748b')
                .attr('font-size', '9px')
                .text(child.description);
        });

    } catch (err) {
        console.error('Error updating decision tree:', err);
    }
}

// ============= CONFIDENCE GAUGES =============
async function updateConfidenceGauges() {
    try {
        const response = await fetch('/api/confidence');
        const data = await response.json();

        updateGauge('iso-gauge', data.iso?.confidence || 0, colors.red);
        updateGauge('lof-gauge', data.lof?.confidence || 0, colors.blue);
        updateGauge('ocsvm-gauge', data.ocsvm?.confidence || 0, colors.green);
        updateGauge('rl-gauge', data.rl?.confidence || 0, colors.orange);

    } catch (err) {
        console.error('Error updating confidence:', err);
    }
}

function updateGauge(gaugeId, value, color) {
    const gauge = document.getElementById(gaugeId);
    if (!gauge) return;

    const circle = gauge.querySelector('.gauge-fill');
    const valueEl = gauge.querySelector('.gauge-value');

    if (circle) {
        const circumference = 2 * Math.PI * 50;
        const dashArray = circumference * Math.min(value, 1);
        circle.style.strokeDasharray = `${dashArray} ${circumference}`;
        circle.style.stroke = color;
    }

    if (valueEl) {
        valueEl.textContent = `${(value * 100).toFixed(0)}%`;
    }
}

// ============= ANOMALY PREDICTION =============
async function updatePrediction() {
    try {
        const response = await fetch('/api/prediction');
        const data = await response.json();

        // Update prediction arc
        const arc = document.getElementById('prediction-arc');
        if (arc) {
            const probability = data.probability || 0;
            const dashArray = 110 * probability;
            arc.style.strokeDasharray = `${dashArray}, 110`;
        }

        // Update value
        const valueEl = document.getElementById('prediction-value');
        if (valueEl) {
            valueEl.textContent = `${((data.probability || 0) * 100).toFixed(0)}%`;
        }

        // Update trend
        const trendEl = document.getElementById('prediction-trend');
        if (trendEl) {
            const trendIcon = data.trend === 'increasing' ? '📈' : data.trend === 'decreasing' ? '📉' : '➡️';
            trendEl.textContent = `${trendIcon} Trend: ${data.trend?.charAt(0).toUpperCase() + data.trend?.slice(1) || 'Stable'}`;
        }

    } catch (err) {
        console.error('Error updating prediction:', err);
    }
}

// ============= MAIN DATA UPDATE =============
async function updateDashboard() {
    try {
        const [statusRes, dataRes] = await Promise.all([
            fetch('/api/status'),
            fetch('/api/data')
        ]);

        const status = await statusRes.json();
        const data = await dataRes.json();

        // Update status displays
        updateStatusDisplays(status);

        // Update charts
        updateCharts(data);

        // Update table
        // updateAnomalyTable(); // Removed

        // Update alerts
        updateAlerts();

        // Check for completion
        if (status.final_results?.is_complete) {
            showSimulationComplete(status.final_results);
        }

    } catch (err) {
        console.error('Error updating dashboard:', err);
    }
}

function updateStatusDisplays(status) {
    // Update stat cards
    const timeEl = document.getElementById('stat-time');
    if (timeEl && status.latest) {
        timeEl.textContent = `${parseFloat(status.latest.time || 0).toFixed(2)}s`;
    }

    const anomaliesEl = document.getElementById('stat-anomalies');
    if (anomaliesEl) {
        anomaliesEl.textContent = status.stats?.anomalies || 0;
    }

    const stepsEl = document.getElementById('stat-steps');
    if (stepsEl) {
        stepsEl.textContent = status.step || 0;
    }

    // Anomaly rate
    const rateEl = document.getElementById('anomaly-rate');
    if (rateEl && status.stats) {
        const rate = status.stats.total_samples > 0
            ? ((status.stats.anomalies / status.stats.total_samples) * 100).toFixed(1)
            : 0;
        rateEl.textContent = `${rate}% rate`;
    }

    // Progress bar
    const progressEl = document.getElementById('simulation-progress');
    const progressText = document.getElementById('progress-text');
    const duration = status.config?.duration || 60;
    if (progressEl) {
        const progress = (status.step / duration) * 100;
        progressEl.style.width = `${progress}%`;
    }
    if (progressText) {
        progressText.textContent = `${status.step || 0} / ${duration}s`;
    }

    // Button states
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    if (startBtn) startBtn.disabled = status.running;
    if (stopBtn) stopBtn.disabled = !status.running;

    // Live indicator
    updateLiveIndicator(status.running);
}

function updateCharts(data) {
    // Vote trends
    if (state.charts.voteTrends && data.vote_trends?.time) {
        const labels = data.vote_trends.time.map(t => t.toFixed(1));
        state.charts.voteTrends.data.labels = labels;

        ['cand1', 'cand2', 'cand3', 'cand4'].forEach((cand, i) => {
            if (data.vote_trends[cand]) {
                state.charts.voteTrends.data.datasets[i].data = data.vote_trends[cand];
            }
        });
        state.charts.voteTrends.update('none');
    }

    // Anomaly scores
    if (state.charts.anomalyScores && data.anomaly_scores?.time) {
        state.charts.anomalyScores.data.labels = data.anomaly_scores.time.map(t => t.toFixed(1));
        state.charts.anomalyScores.data.datasets[0].data = data.anomaly_scores.scores;
        state.charts.anomalyScores.update('none');
    }

    // Trend chart
    if (state.charts.trend && data.anomaly_scores?.time) {
        const labels = data.anomaly_scores.time.map(t => t.toFixed(1));
        state.charts.trend.data.labels = labels;
        state.charts.trend.data.datasets[0].data = data.anomaly_scores.scores;
        state.charts.trend.data.datasets[1].data = data.trend_data?.ma_5 || [];
        state.charts.trend.data.datasets[2].data = data.trend_data?.ma_10 || [];
        state.charts.trend.update('none');
    }

    // RL rewards
    if (state.charts.rlRewards && data.rl_rewards) {
        state.charts.rlRewards.data.labels = data.rl_rewards.map((_, i) => i);
        state.charts.rlRewards.data.datasets[0].data = data.rl_rewards;
        state.charts.rlRewards.update('none');

        // Update stat
        const avgReward = data.rl_rewards.length > 0
            ? (data.rl_rewards.reduce((a, b) => a + b, 0) / data.rl_rewards.length).toFixed(3)
            : '0.000';
        const rewardEl = document.getElementById('stat-reward');
        if (rewardEl) rewardEl.textContent = avgReward;
    }

    // RL loss
    if (state.charts.rlLoss && data.rl_losses) {
        state.charts.rlLoss.data.labels = data.rl_losses.map((_, i) => i);
        state.charts.rlLoss.data.datasets[0].data = data.rl_losses;
        state.charts.rlLoss.update('none');
    }

    // Model comparison
    if (state.charts.modelComparison && data.model_scores?.time) {
        const labels = data.model_scores.time.map(t => t.toFixed(1));
        state.charts.modelComparison.data.labels = labels;
        state.charts.modelComparison.data.datasets[0].data = data.model_scores.iso || [];
        state.charts.modelComparison.data.datasets[1].data = data.model_scores.lof || [];
        state.charts.modelComparison.data.datasets[2].data = data.model_scores.ocsvm || [];
        state.charts.modelComparison.data.datasets[3].data = data.model_scores.rl_pred || [];
        state.charts.modelComparison.update('none');
    }

    // RL epsilon
    if (data.rl_metrics?.epsilon !== undefined) {
        const epsilonEl = document.getElementById('rl-epsilon');
        if (epsilonEl) epsilonEl.textContent = `ε: ${data.rl_metrics.epsilon.toFixed(3)}`;
    }

    // Sparklines
    if (state.charts.rewardSparkline && data.rl_rewards) {
        state.charts.rewardSparkline.data.labels = data.rl_rewards.slice(-20).map((_, i) => i);
        state.charts.rewardSparkline.data.datasets[0].data = data.rl_rewards.slice(-20);
        state.charts.rewardSparkline.update('none');
    }
}

async function updateCandidateCharts() {
    try {
        const response = await fetch('/api/candidate-anomalies');
        const data = await response.json();

        // Pie chart
        if (state.charts.candidatePie && data.pie_data) {
            state.charts.candidatePie.data.labels = Object.keys(data.pie_data);
            state.charts.candidatePie.data.datasets[0].data = Object.values(data.pie_data);
            state.charts.candidatePie.update();
        }

        // Bar chart
        if (state.charts.candidateBar && data.bar_data) {
            const candidates = Object.keys(data.bar_data);
            const allTypes = new Set();
            Object.values(data.bar_data).forEach(types => Object.keys(types).forEach(t => allTypes.add(t)));

            state.charts.candidateBar.data.labels = candidates;
            state.charts.candidateBar.data.datasets = Array.from(allTypes).map((type, i) => ({
                label: type,
                data: candidates.map(c => data.bar_data[c]?.[type] || 0),
                backgroundColor: colors.candidates[i % colors.candidates.length]
            }));
            state.charts.candidateBar.update();
        }

    } catch (err) {
        console.error('Error updating candidate charts:', err);
    }
}

async function updateAlerts() {
    try {
        const response = await fetch('/api/alerts');
        const data = await response.json();

        const container = document.getElementById('alerts-container');
        if (!container) return;

        if (!data.alerts || data.alerts.length === 0) {
            container.innerHTML = `
                <div class="alert-card waiting">
                    <span class="alert-icon">📡</span>
                    <div class="alert-info">
                        <p class="alert-type">Waiting for data...</p>
                        <p class="alert-id">Start simulation to detect anomalies</p>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = data.alerts.slice(0, 5).map(alert => `
            <div class="alert-card danger">
                <span class="alert-icon">⚠️</span>
                <div class="alert-info">
                    <p class="alert-type">${alert.type}</p>
                    <p class="alert-id">Score: ${(alert.score * 100).toFixed(1)}%</p>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Error updating alerts:', err);
    }
}

async function updateAnomalyTable() {
    try {
        const response = await fetch('/api/alerts');
        const data = await response.json();

        const tbody = document.getElementById('anomaly-table-body');
        if (!tbody) return;

        if (!data.alerts || data.alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No anomalies detected yet</td></tr>';
            return;
        }

        tbody.innerHTML = data.alerts.map(alert => {
            const cands = alert.candidates || {};
            const c1 = cands.cand1 || 0;
            const c2 = cands.cand2 || 0;
            const c3 = cands.cand3 || 0;
            const c4 = cands.cand4 || 0;
            
            return `
            <tr>
                <td>${parseFloat(alert.time || 0).toFixed(2)}s</td>
                <td><span class="status-badge anomaly">${alert.type || 'Unknown'}</span></td>
                <td>${((alert.score || 0) * 100).toFixed(1)}%</td>
                <td>${c1}</td>
                <td>${c2}</td>
                <td>${c3}</td>
                <td>${c4}</td>
                <td>${alert.rl_prediction || 0}</td>
            </tr>
        `}).join('');

    } catch (err) {
        console.error('Error updating anomaly table:', err);
    }
}

function addAlertToSidebar(data) {
    const container = document.getElementById('alerts-container');
    if (!container) return;

    const alertHtml = `
        <div class="alert-card danger">
            <span class="alert-icon">⚠️</span>
            <div class="alert-info">
                <p class="alert-type">${data.type}</p>
                <p class="alert-id">Score: ${(data.score * 100).toFixed(1)}%</p>
            </div>
        </div>
    `;

    // Remove waiting message if present
    const waiting = container.querySelector('.waiting');
    if (waiting) waiting.remove();

    container.insertAdjacentHTML('afterbegin', alertHtml);

    // Keep only last 5 alerts
    const alerts = container.querySelectorAll('.alert-card');
    if (alerts.length > 5) {
        alerts[alerts.length - 1].remove();
    }
}

// ============= SIMULATION COMPLETE =============
function showSimulationComplete(data) {
    const section = document.getElementById('simulation-complete-section');
    if (!section) return;

    section.style.display = 'block';

    document.getElementById('winner-name').textContent = (data.winner || '-').toUpperCase();
    document.getElementById('winner-votes').textContent = data.winner_votes || 0;
    document.getElementById('final-samples').textContent = data.total_samples || 0;
    document.getElementById('final-anomalies').textContent = data.total_anomalies || 0;

    // Results table
    const tbody = document.getElementById('results-table-body');
    if (tbody && data.final_votes) {
        tbody.innerHTML = Object.entries(data.final_votes)
            .sort((a, b) => b[1] - a[1])
            .map(([cand, votes]) => `
                <tr>
                    <td>${cand.toUpperCase()}</td>
                    <td>${votes}</td>
                </tr>
            `).join('');
    }

    // Scroll into view
    section.scrollIntoView({ behavior: 'smooth' });
}

function hideSimulationComplete() {
    const section = document.getElementById('simulation-complete-section');
    if (section) section.style.display = 'none';
}

// ============= EVENT LISTENERS =============
function initEventListeners() {
    // Start button
    document.getElementById('start-btn')?.addEventListener('click', async () => {
        hideSimulationComplete();
        state.anomalyCount = 0;
        updateNotificationBadge();

        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start' })
        });

        showNotification('Simulation Started', 'Monitoring for anomalies...', 'success');
    });

    // Stop button
    document.getElementById('stop-btn')?.addEventListener('click', async () => {
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'stop' })
        });

        showNotification('Simulation Paused', 'Data collection stopped', 'info');
    });

    // Reset button
    document.getElementById('reset-btn')?.addEventListener('click', () => {
        hideSimulationComplete();
        location.reload();
    });

    // Threshold slider
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdValue = document.getElementById('threshold-value');
    thresholdSlider?.addEventListener('input', async (e) => {
        const value = parseFloat(e.target.value);
        if (thresholdValue) thresholdValue.textContent = value.toFixed(2);

        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'threshold', threshold: value })
        });
    });

    // Export buttons
    document.getElementById('export-pdf')?.addEventListener('click', async () => {
        try {
            const response = await fetch('/api/export/pdf');
            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `VoteGuard_Report_${new Date().toISOString().slice(0, 10)}.pdf`;
                a.click();
                showNotification('PDF Generated', 'Report downloaded successfully', 'success');
            }
        } catch (err) {
            showNotification('Export Failed', err.message, 'error');
        }
    });

    document.getElementById('export-charts')?.addEventListener('click', () => {
        let exported = 0;
        Object.entries(state.charts).forEach(([name, chart]) => {
            if (chart && chart.canvas) {
                const link = document.createElement('a');
                link.download = `voteguard_${name}.png`;
                link.href = chart.canvas.toDataURL('image/png');
                link.click();
                exported++;
            }
        });
        showNotification('Charts Exported', `${exported} chart(s) downloaded`, 'success');
    });

    // Fullscreen chart
    document.querySelectorAll('.expand-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const card = btn.closest('.chart-card');
            const chartName = card?.dataset.chart;
            if (chartName && state.charts[chartName]) {
                openFullscreenChart(state.charts[chartName]);
            }
        });
    });

    document.getElementById('close-fullscreen')?.addEventListener('click', closeFullscreenChart);
    document.getElementById('fullscreen-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'fullscreen-modal') closeFullscreenChart();
    });

    // Auto-rotate toggle for 3D
    document.getElementById('rotate-toggle')?.addEventListener('click', () => {
        state.autoRotate = !state.autoRotate;
        showNotification('3D View', state.autoRotate ? 'Auto-rotate enabled' : 'Auto-rotate disabled', 'info');
    });

    // Menu toggle for mobile
    document.getElementById('menu-toggle')?.addEventListener('click', () => {
        document.getElementById('sidebar')?.classList.toggle('open');
    });

    // Fullscreen mode
    document.getElementById('fullscreen-btn')?.addEventListener('click', () => {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    });

    // Config modal
    document.querySelectorAll('[data-section="config"]').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            document.getElementById('config-modal')?.classList.add('active');
        });
    });

    // Email modal
    document.getElementById('email-subscribe-btn')?.addEventListener('click', () => {
        document.getElementById('email-modal')?.classList.add('active');
    });

    // History modal
    document.querySelectorAll('[data-section="history"]').forEach(el => {
        el.addEventListener('click', async (e) => {
            e.preventDefault();
            await loadHistory();
            document.getElementById('history-modal')?.classList.add('active');
        });
    });

    // Close modals
    document.querySelectorAll('.close-modal').forEach(btn => {
        btn.addEventListener('click', () => {
            btn.closest('.modal')?.classList.remove('active');
        });
    });

    // Click outside modal to close
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.remove('active');
        });
    });

    // Config tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${btn.dataset.tab}-tab`)?.classList.add('active');
        });
    });

    // Save config
    document.getElementById('save-config')?.addEventListener('click', async () => {
        const config = {
            simulation: {
                num_candidates: parseInt(document.getElementById('config-candidates')?.value || 4),
                duration: parseInt(document.getElementById('config-duration')?.value || 60),
                overt_anomaly_prob: parseFloat(document.getElementById('config-overt')?.value || 0.1),
                stealth_anomaly_prob: parseFloat(document.getElementById('config-stealth')?.value || 0.05)
            },
            ml: {
                iso_contamination: parseFloat(document.getElementById('config-iso-contamination')?.value || 0.1),
                iso_n_estimators: parseInt(document.getElementById('config-iso-estimators')?.value || 100),
                lof_n_neighbors: parseInt(document.getElementById('config-lof-neighbors')?.value || 10),
                ocsvm_kernel: document.getElementById('config-ocsvm-kernel')?.value || 'rbf'
            }
        };

        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        document.getElementById('config-modal')?.classList.remove('active');
        showNotification('Configuration Saved', 'Settings will apply on next simulation', 'success');
    });

    // Subscribe email
    document.getElementById('subscribe-email')?.addEventListener('click', async () => {
        const email = document.getElementById('email-address')?.value;
        const name = document.getElementById('email-name')?.value;

        if (!email) {
            showNotification('Error', 'Please enter an email address', 'error');
            return;
        }

        await fetch('/api/email/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email,
                name,
                on_anomaly: document.getElementById('email-on-anomaly')?.checked,
                on_completion: document.getElementById('email-on-complete')?.checked
            })
        });

        document.getElementById('email-modal')?.classList.remove('active');
        showNotification('Subscribed!', 'You will receive email alerts', 'success');
    });

    // Range input value displays
    document.querySelectorAll('input[type="range"]').forEach(input => {
        const valueSpan = input.parentElement?.querySelector('.range-value') || input.nextElementSibling;
        if (valueSpan && valueSpan.classList.contains('range-value')) {
            input.addEventListener('input', () => {
                valueSpan.textContent = input.id.includes('prob') || input.id.includes('epsilon')
                    ? `${(parseFloat(input.value) * 100).toFixed(0)}%`
                    : parseFloat(input.value).toFixed(2);
            });
        }
    });

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        document.getElementById('notification-btn')?.addEventListener('click', () => {
            Notification.requestPermission();
        });
    }
}

// ============= FULLSCREEN CHART =============
function openFullscreenChart(chart) {
    const modal = document.getElementById('fullscreen-modal');
    const canvas = document.getElementById('fullscreen-chart');
    if (!modal || !canvas) return;

    modal.classList.add('active');

    // Clone chart to fullscreen canvas
    state.fullscreenChart = new Chart(canvas, {
        type: chart.config.type,
        data: JSON.parse(JSON.stringify(chart.data)),
        options: {
            ...chart.options,
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function closeFullscreenChart() {
    const modal = document.getElementById('fullscreen-modal');
    modal?.classList.remove('active');

    if (state.fullscreenChart) {
        state.fullscreenChart.destroy();
        state.fullscreenChart = null;
    }
}

// ============= HISTORY =============
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();

        const container = document.getElementById('history-list');
        if (!container) return;

        if (!data.sessions || data.sessions.length === 0) {
            container.innerHTML = '<p class="empty-state">No previous sessions found</p>';
            return;
        }

        container.innerHTML = data.sessions.map(session => `
            <div class="history-item" data-id="${session.id}">
                <div class="history-info">
                    <h4>${session.name}</h4>
                    <p>${new Date(session.created_at).toLocaleString()}</p>
                </div>
                <div class="history-stats">
                    <span>${session.total_samples} samples</span>
                    <span>${session.total_anomalies} anomalies</span>
                    <span>Winner: ${session.winner?.toUpperCase() || 'N/A'}</span>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('Error loading history:', err);
    }
}

// ============= NOTIFICATIONS =============
function showNotification(title, message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const icons = { success: '✓', error: '✗', warning: '⚠', info: 'ℹ' };

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span class="notification-icon">${icons[type] || 'ℹ'}</span>
        <div class="notification-content">
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        </div>
        <button class="notification-close">×</button>
    `;

    container.appendChild(notification);

    // Close button
    notification.querySelector('.notification-close')?.addEventListener('click', () => {
        notification.remove();
    });

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease forwards';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function updateNotificationBadge() {
    const badge = document.getElementById('notification-badge');
    if (badge) {
        badge.textContent = state.anomalyCount;
        badge.style.display = state.anomalyCount > 0 ? 'block' : 'none';
    }
}

// ============= POLLING =============
function startDataPolling() {
    // Main dashboard update
    setInterval(updateDashboard, 1000);

    // Less frequent updates
    setInterval(() => {
        updateCandidateCharts();
        updateConfidenceGauges();
        updatePrediction();
    }, 2000);

    // Visualizations
    setInterval(() => {
        // update3DTrajectories(); // Removed - 3D section disabled
        updateHeatmap();
        updateDecisionTree();
    }, 3000);
}

// ============= ADD CSS FOR ANIMATIONS =============
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .status-badge.anomaly {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px;
        background: var(--bg-primary);
        border-radius: 8px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .history-item:hover {
        background: var(--bg-card-hover);
    }
    
    .history-info h4 {
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .history-info p {
        font-size: 12px;
        color: var(--text-secondary);
    }
    
    .history-stats {
        display: flex;
        gap: 16px;
        font-size: 12px;
        color: var(--text-secondary);
    }
`;
document.head.appendChild(style);

// ============= CLEANUP =============
window.addEventListener('beforeunload', () => {
    if (state.socket) {
        state.socket.disconnect();
    }
});
