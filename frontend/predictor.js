/**
 * Sleep Stage Predictor - JavaScript
 * Generates physiological data (HRV, HR, PPG, Acceleration) based on actual model features
 * Features extracted from wearable sensors (heart rate, HRV, acceleration)
 */

// Sleep stage definitions
const SLEEP_STAGES = {
    0: { name: 'Awake', icon: '👁️', description: 'Eyes open, alert', color: '#3b82f6' },
    1: { name: 'Light Sleep', icon: '🌙', description: 'Transitional sleep phase', color: '#8b5cf6' },
    2: { name: 'Deep Sleep', icon: '💤', description: 'Restorative sleep phase', color: '#ec4899' }
};

// 12 Features from actual model training
const FEATURE_NAMES = [
    'mean_hr',          // Average heart rate (bpm)
    'std_hr',           // Heart rate variability (std)
    'min_hr',           // Minimum HR in epoch (bpm)
    'hr_range',         // Max - Min HR (bpm)
    'rmssd',            // Root mean square of successive differences (ms)
    'pnn50',            // % successive intervals >50ms
    'ppg_snr',          // Signal-to-noise ratio (dB)
    'accel_mean',       // Mean acceleration magnitude (g)
    'spectral_power',   // Power in 0.1-1.0 Hz band
    'accel_var',        // Variance of acceleration
    'zcr',              // Zero crossing rate
    'spectral_entropy'  // Entropy of frequency distribution
];

// DOM Elements
const generateBtn = document.getElementById('generateBtn');
const dataDisplay = document.getElementById('dataDisplay');
const emptyState = document.getElementById('emptyState');
const predictionDisplay = document.getElementById('predictionDisplay');
const predictionEmpty = document.getElementById('predictionEmpty');

// Event listeners
generateBtn.addEventListener('click', handleGenerateData);

/**
 * Generate realistic physiological data from wearable sensors
 * Mimics actual patterns: HR, HRV, PPG, Acceleration
 */
function generatePhysiologicalData() {
    // Simulate realistic sensor readings with stage-dependent patterns
    // During deep sleep: lower HR, higher HRV, minimal movement
    // During light sleep: moderate HR, moderate HRV, occasional movement
    // During wake: higher HR, lower HRV, significant movement
    
    const stageType = Math.random();
    
    let data;
    if (stageType < 0.33) {
        // Deep Sleep pattern
        data = {
            mean_hr: 55 + Math.random() * 15,        // 55-70 bpm (low)
            std_hr: 8 + Math.random() * 5,           // 8-13 std
            min_hr: 48 + Math.random() * 8,          // 48-56 bpm minimum
            hr_range: 18 + Math.random() * 12,       // 18-30 bpm range (narrow)
            rmssd: 65 + Math.random() * 50,          // 65-115 ms (high HRV)
            pnn50: 25 + Math.random() * 40,          // 25-65% (high)
            ppg_snr: 18 + Math.random() * 12,        // 18-30 dB (good)
            accel_mean: 0.02 + Math.random() * 0.05, // 0.02-0.07 g (minimal)
            spectral_power: 0.3 + Math.random() * 0.3, // 0.3-0.6
            accel_var: 0.001 + Math.random() * 0.004,  // Very low variance
            zcr: 0.05 + Math.random() * 0.1,        // 0.05-0.15 (low)
            spectral_entropy: 0.4 + Math.random() * 0.2 // Low entropy (regular patterns)
        };
    } else if (stageType < 0.66) {
        // Light Sleep pattern
        data = {
            mean_hr: 65 + Math.random() * 20,        // 65-85 bpm (moderate)
            std_hr: 10 + Math.random() * 8,          // 10-18 std
            min_hr: 55 + Math.random() * 10,         // 55-65 bpm minimum
            hr_range: 25 + Math.random() * 15,       // 25-40 bpm range
            rmssd: 35 + Math.random() * 40,          // 35-75 ms (moderate)
            pnn50: 15 + Math.random() * 30,          // 15-45% (moderate)
            ppg_snr: 16 + Math.random() * 10,        // 16-26 dB
            accel_mean: 0.08 + Math.random() * 0.1,  // 0.08-0.18 g (some movement)
            spectral_power: 0.5 + Math.random() * 0.4, // 0.5-0.9
            accel_var: 0.005 + Math.random() * 0.01,   // Moderate variance
            zcr: 0.15 + Math.random() * 0.2,        // 0.15-0.35 (moderate)
            spectral_entropy: 0.6 + Math.random() * 0.25 // Moderate entropy
        };
    } else {
        // Awake pattern
        data = {
            mean_hr: 75 + Math.random() * 30,        // 75-105 bpm (high)
            std_hr: 12 + Math.random() * 10,         // 12-22 std
            min_hr: 65 + Math.random() * 15,         // 65-80 bpm minimum
            hr_range: 35 + Math.random() * 25,       // 35-60 bpm range (wide)
            rmssd: 15 + Math.random() * 30,          // 15-45 ms (low HRV)
            pnn50: 5 + Math.random() * 15,           // 5-20% (low)
            ppg_snr: 14 + Math.random() * 8,         // 14-22 dB (moderate)
            accel_mean: 0.2 + Math.random() * 0.3,   // 0.2-0.5 g (high movement)
            spectral_power: 0.7 + Math.random() * 0.5, // 0.7-1.2
            accel_var: 0.01 + Math.random() * 0.02,    // High variance
            zcr: 0.35 + Math.random() * 0.3,        // 0.35-0.65 (high)
            spectral_entropy: 0.75 + Math.random() * 0.25 // High entropy (random patterns)
        };
    }
    
    return data;
}

/**
 * Predict sleep stage from physiological features
 * Uses Random Forest-like heuristics from model training
 */
function predictSleepStage(features) {
    let deepScore = 0;
    let lightScore = 0;
    let wakeScore = 0;
    
    // Deep Sleep indicators:
    // - Low heart rate (mean_hr < 65)
    // - High HRV (rmssd > 60, pnn50 > 20)
    // - Minimal movement (accel_mean < 0.1, accel_var < 0.01)
    // - Regular patterns (low spectral_entropy)
    deepScore += Math.max(0, (65 - features.mean_hr) / 65) * 25;  // Low HR favors deep
    deepScore += Math.min(features.rmssd / 120, 1) * 20;          // High RMSSD
    deepScore += Math.min(features.pnn50 / 50, 1) * 20;           // High PNN50
    deepScore += Math.max(0, (0.15 - features.accel_mean) / 0.15) * 20;  // Low movement
    deepScore += Math.max(0, (0.01 - features.accel_var) / 0.01) * 15;   // Low variance
    
    // Light Sleep indicators:
    // - Moderate heart rate (65-85)
    // - Moderate HRV (30 < rmssd < 80)
    // - Some movement (0.05 < accel_mean < 0.2)
    lightScore += Math.max(0, (1 - Math.abs(features.mean_hr - 75) / 75)) * 25;  // Moderate HR
    lightScore += (features.rmssd > 30 && features.rmssd < 80 ? 20 : 5);
    lightScore += (features.pnn50 > 10 && features.pnn50 < 40 ? 15 : 5);
    lightScore += Math.min(Math.abs(0.13 - features.accel_mean) / 0.13, 1) * 20;  // Moderate movement
    lightScore += (features.spectral_entropy > 0.5 && features.spectral_entropy < 0.7 ? 15 : 5);
    
    // Awake indicators:
    // - High heart rate (mean_hr > 85)
    // - Low HRV (rmssd < 40, pnn50 < 20)
    // - Significant movement (accel_mean > 0.15)
    // - High variability (spectral_entropy > 0.75)
    wakeScore += Math.min((features.mean_hr - 65) / 65, 1) * 25;   // High HR
    wakeScore += Math.max(0, (50 - features.rmssd) / 50) * 20;     // Low RMSSD
    wakeScore += Math.max(0, (30 - features.pnn50) / 30) * 20;     // Low PNN50
    wakeScore += Math.min(features.accel_mean / 0.5, 1) * 20;      // High movement
    wakeScore += Math.min(features.spectral_entropy / 1.0, 1) * 15; // High entropy
    
    // Normalize scores to probabilities
    const total = deepScore + lightScore + wakeScore;
    const probabilities = {
        0: (wakeScore / total) * 100,
        1: (lightScore / total) * 100,
        2: (deepScore / total) * 100
    };
    
    // Determine predicted stage from highest probability
    let predictedStage = 0;
    let confidence = probabilities[0];
    
    if (probabilities[1] > confidence) {
        predictedStage = 1;
        confidence = probabilities[1];
    }
    if (probabilities[2] > confidence) {
        predictedStage = 2;
        confidence = probabilities[2];
    }
    
    return {
        stage: predictedStage,
        confidence: Math.round(confidence),
        probabilities: {
            0: Math.round(probabilities[0]),
            1: Math.round(probabilities[1]),
            2: Math.round(probabilities[2])
        }
    };
}

/**
 * Handle Generate Data button click
 */
function handleGenerateData() {
    // Add loading animation
    generateBtn.classList.add('loading');
    generateBtn.disabled = true;
    
    // Simulate processing time
    setTimeout(() => {
        // Generate physiological data
        const physiologicalData = generatePhysiologicalData();
        
        // Make prediction
        const prediction = predictSleepStage(physiologicalData);
        
        // Display results
        displayPhysiologicalData(physiologicalData);
        displayPrediction(prediction);
        
        // Remove loading animation
        generateBtn.classList.remove('loading');
        generateBtn.disabled = false;
    }, 800);
}

/**
 * Display generated physiological data
 * Shows all 12 features: HR, HRV, Movement, PPG Signal Quality, and Variability
 */
function displayPhysiologicalData(data) {
    // Display all 12 features (matching HTML IDs exactly)
    updateSignalBar('delta', data.mean_hr, `Mean HR: ${Math.round(data.mean_hr)} bpm`, 120);
    updateSignalBar('feature5', data.std_hr, `HR StdDev: ${Math.round(data.std_hr)}`, 20);
    updateSignalBar('feature6', data.min_hr, `Min HR: ${Math.round(data.min_hr)} bpm`, 100);
    updateSignalBar('feature7', data.hr_range, `HR Range: ${Math.round(data.hr_range)} bpm`, 60);
    updateSignalBar('feature8', data.rmssd, `RMSSD: ${Math.round(data.rmssd)} ms`, 150);
    updateSignalBar('feature9', data.pnn50, `PNN50: ${Math.round(data.pnn50)}%`, 100);
    updateSignalBar('gamma', data.ppg_snr, `PPG SNR: ${Math.round(data.ppg_snr)} dB`, 35);
    updateSignalBar('feature11', data.accel_mean * 50, `Accel: ${(data.accel_mean).toFixed(2)} g`, 50);
    updateSignalBar('feature12', data.spectral_power * 100, `Spectral: ${Math.round(data.spectral_power * 100)}`, 120);
    updateSignalBar('feature13', data.accel_var * 1000, `Accel Var: ${(data.accel_var).toFixed(4)}`, 20);
    updateSignalBar('feature14', data.zcr * 50, `ZCR: ${(data.zcr).toFixed(2)}`, 50);
    updateSignalBar('feature15', data.spectral_entropy * 100, `Entropy: ${(data.spectral_entropy).toFixed(2)}`, 100);
    
    // Update stats display
    document.getElementById('quality').textContent = Math.round(data.ppg_snr) + ' dB';
    document.getElementById('noise').textContent = Math.round(data.spectral_entropy * 100) / 100;
    
    // Show data display, hide empty state
    dataDisplay.style.display = 'flex';
    emptyState.style.display = 'none';
}

/**
 * Animate speedometer needle to confidence level
 * Rotates from 180° (0%) to 0° (100%)
 */
function animateSpeedometerNeedle(confidence) {
    const needle = document.getElementById('confidenceNeedle');
    const percentValue = document.getElementById('confidencePercent');
    
    // Convert confidence (0-100) to rotation angle (180 to 0 degrees)
    // 0% = 180°, 100% = 0°
    const targetAngle = 180 - (confidence * 1.8);
    
    // Animate the needle
    let currentAngle = 180;
    const startTime = Date.now();
    const duration = 1200; // 1.2 seconds for smooth animation
    
    function animateFrame() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth acceleration/deceleration
        const easeProgress = progress < 0.5 
            ? 2 * progress * progress 
            : -1 + (4 - 2 * progress) * progress;
        
        currentAngle = 180 + (targetAngle - 180) * easeProgress;
        needle.setAttribute('transform', `translate(150, 150) rotate(${currentAngle})`);
        
        // Update percentage text
        const displayPercent = Math.round(confidence * easeProgress);
        percentValue.textContent = displayPercent + '%';
        
        if (progress < 1) {
            requestAnimationFrame(animateFrame);
        } else {
            percentValue.textContent = confidence + '%';
        }
    }
    
    animateFrame();
}

/**
 * Update individual signal bar
 * @param {string} type - signal type (delta, theta, alpha, beta, gamma)
 * @param {number} value - raw value
 * @param {string} label - display label
 * @param {number} maxValue - max scale value for percentage calculation
 */
function updateSignalBar(type, value, label, maxValue) {
    const percentage = Math.min((value / maxValue) * 100, 100);
    
    document.getElementById(`${type}Value`).textContent = label;
    document.getElementById(`${type}Bar`).style.width = percentage + '%';
}

/**
 * Display prediction results
 */
function displayPrediction(prediction) {
    const stage = SLEEP_STAGES[prediction.stage];
    
    // Update main prediction
    document.getElementById('stageIcon').textContent = stage.icon;
    document.getElementById('stageName').textContent = stage.name;
    document.getElementById('stageDesc').textContent = stage.description;
    
    // Animate confidence speedometer needle
    animateSpeedometerNeedle(prediction.confidence);
    
    // Update probabilities
    updateProbabilityBar('Deep', prediction.probabilities[2], 'deep');
    updateProbabilityBar('Light', prediction.probabilities[1], 'light');
    updateProbabilityBar('Wake', prediction.probabilities[0], 'wake');
    
    // Update timestamp
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    document.getElementById('predictTime').textContent = timeStr;
    
    // Show prediction display, hide empty state
    predictionDisplay.style.display = 'flex';
    predictionEmpty.style.display = 'none';
}

/**
 * Update probability bar
 */
function updateProbabilityBar(stageName, percentage, type) {
    document.getElementById(`prob${stageName}`).textContent = percentage + '%';
    document.getElementById(`prob${stageName}Bar`).style.width = percentage + '%';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Sleep Stage Predictor loaded');
    console.log('Click "Generate Data" to start predicting sleep stages');
});
