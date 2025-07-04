/**
 * Charts and data visualization utilities for BCI application
 */

// Chart.js default configuration
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.color = '#495057';
Chart.defaults.borderColor = '#dee2e6';

// Color palette for different classes
const BCI_COLORS = {
    'RIGHT_HAND': '#007bff',
    'LEFT_HAND': '#28a745', 
    'FEET': '#ffc107',
    'REST': '#6c757d',
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'secondary': '#6c757d'
};

/**
 * Create a real-time confidence chart
 */
function createConfidenceChart(canvasId, classLabels) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const datasets = classLabels.map((label, index) => ({
        label: label,
        data: [],
        borderColor: BCI_COLORS[label] || `hsl(${index * 360 / classLabels.length}, 70%, 50%)`,
        backgroundColor: (BCI_COLORS[label] || `hsl(${index * 360 / classLabels.length}, 70%, 50%)`) + '20',
        fill: false,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 4
    }));
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Confidence (%)'
                    },
                    grid: {
                        color: '#f1f3f4'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    cornerRadius: 6,
                    displayColors: true
                }
            },
            animation: {
                duration: 300
            }
        }
    });
}

/**
 * Create a training progress chart
 */
function createTrainingChart(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: BCI_COLORS.danger,
                backgroundColor: BCI_COLORS.danger + '20',
                yAxisID: 'y'
            }, {
                label: 'Validation Accuracy',
                data: [],
                borderColor: BCI_COLORS.success,
                backgroundColor: BCI_COLORS.success + '20',
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Training Progress'
                }
            }
        }
    });
}

/**
 * Create a class distribution pie chart
 */
function createClassDistributionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    const colors = labels.map(label => BCI_COLORS[label] || BCI_COLORS.primary);
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(color => color + 'cc'),
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + context.parsed + ' (' + percentage + '%)';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create a session statistics chart
 */
function createSessionStatsChart(canvasId, sessionsData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const approaches = Object.keys(sessionsData);
    const counts = Object.values(sessionsData);
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: approaches.map(a => a.replace('_', ' ').toUpperCase()),
            datasets: [{
                label: 'Sessions',
                data: counts,
                backgroundColor: [
                    BCI_COLORS.primary + '80',
                    BCI_COLORS.info + '80'
                ],
                borderColor: [
                    BCI_COLORS.primary,
                    BCI_COLORS.info
                ],
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Sessions'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Approach'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Sessions by Approach'
                }
            }
        }
    });
}

/**
 * Update chart with new data point
 */
function addDataToChart(chart, label, dataPoint) {
    chart.data.labels.push(label);
    
    if (Array.isArray(dataPoint)) {
        // Multiple datasets
        dataPoint.forEach((value, index) => {
            if (chart.data.datasets[index]) {
                chart.data.datasets[index].data.push(value);
            }
        });
    } else {
        // Single dataset
        chart.data.datasets[0].data.push(dataPoint);
    }
    
    // Keep only last N points
    const maxPoints = 50;
    if (chart.data.labels.length > maxPoints) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }
    
    chart.update('none');
}

/**
 * Create animated progress bar
 */
function createProgressBar(containerId, value, max = 100, label = '') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const percentage = (value / max) * 100;
    
    container.innerHTML = `
        <div class="d-flex justify-content-between mb-1">
            <span class="fw-medium">${label}</span>
            <span class="text-muted">${value}/${max}</span>
        </div>
        <div class="progress" style="height: 8px;">
            <div class="progress-bar progress-bar-animated" 
                 role="progressbar" 
                 style="width: ${percentage}%" 
                 aria-valuenow="${value}" 
                 aria-valuemin="0" 
                 aria-valuemax="${max}">
            </div>
        </div>
    `;
}

/**
 * Create real-time prediction display
 */
function createPredictionDisplay(containerId, classLabels) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const barsHtml = classLabels.map((label, index) => `
        <div class="mb-2">
            <div class="d-flex justify-content-between">
                <span class="fw-medium">${label}</span>
                <span id="prob-${index}-text" class="text-muted">0%</span>
            </div>
            <div class="progress" style="height: 12px;">
                <div class="progress-bar" 
                     id="prob-${index}" 
                     role="progressbar" 
                     style="width: 0%; background-color: ${BCI_COLORS[label] || BCI_COLORS.primary};">
                </div>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="text-center">
                    <h2 id="current-prediction" class="mb-2 text-primary">Waiting...</h2>
                    <p id="current-confidence" class="text-muted">Confidence: ---%</p>
                    <small id="prediction-time" class="text-muted">Processing time: ---ms</small>
                </div>
            </div>
            <div class="col-md-6">
                <div id="probability-bars">
                    ${barsHtml}
                </div>
            </div>
        </div>
    `;
}

/**
 * Update prediction display
 */
function updatePredictionDisplay(prediction) {
    // Update main prediction
    const predictionEl = document.getElementById('current-prediction');
    const confidenceEl = document.getElementById('current-confidence');
    const timeEl = document.getElementById('prediction-time');
    
    if (predictionEl) predictionEl.textContent = prediction.predicted_label;
    if (confidenceEl) confidenceEl.textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
    if (timeEl) timeEl.textContent = `Processing time: ${prediction.prediction_time_ms.toFixed(1)}ms`;
    
    // Update probability bars
    prediction.probabilities.forEach((prob, index) => {
        const percentage = (prob * 100).toFixed(1);
        const barEl = document.getElementById(`prob-${index}`);
        const textEl = document.getElementById(`prob-${index}-text`);
        
        if (barEl) {
            barEl.style.width = `${percentage}%`;
            barEl.style.transition = 'width 0.3s ease';
        }
        if (textEl) textEl.textContent = `${percentage}%`;
    });
}

/**
 * Create EEG signal visualization
 */
function createEEGChart(canvasId, channelNames, sampleRate = 128) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const datasets = channelNames.map((channel, index) => ({
        label: channel,
        data: [],
        borderColor: `hsl(${index * 360 / channelNames.length}, 70%, 50%)`,
        backgroundColor: 'transparent',
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        borderWidth: 1
    }));
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Amplitude (μV)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time (s)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        padding: 10
                    }
                },
                title: {
                    display: true,
                    text: 'Real-time EEG Signals'
                }
            },
            animation: false,
            interaction: {
                intersect: false
            }
        }
    });
}

/**
 * Utility functions
 */
const ChartUtils = {
    /**
     * Get color for class label
     */
    getClassColor: function(classLabel) {
        return BCI_COLORS[classLabel] || BCI_COLORS.primary;
    },
    
    /**
     * Format time for chart labels
     */
    formatTime: function(timestamp) {
        return new Date(timestamp).toLocaleTimeString();
    },
    
    /**
     * Calculate moving average
     */
    movingAverage: function(data, windowSize = 5) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - windowSize + 1);
            const subset = data.slice(start, i + 1);
            const average = subset.reduce((a, b) => a + b, 0) / subset.length;
            result.push(average);
        }
        return result;
    },
    
    /**
     * Smooth data using exponential moving average
     */
    exponentialMovingAverage: function(data, alpha = 0.3) {
        const result = [data[0]];
        for (let i = 1; i < data.length; i++) {
            result.push(alpha * data[i] + (1 - alpha) * result[i - 1]);
        }
        return result;
    },
    
    /**
     * Destroy chart safely
     */
    destroyChart: function(chart) {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    }
};

// Export for global use
window.BCICharts = {
    createConfidenceChart,
    createTrainingChart,
    createClassDistributionChart,
    createSessionStatsChart,
    createPredictionDisplay,
    createEEGChart,
    addDataToChart,
    updatePredictionDisplay,
    createProgressBar,
    ChartUtils,
    BCI_COLORS
};