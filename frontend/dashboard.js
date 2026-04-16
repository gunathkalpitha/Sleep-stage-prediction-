/**
 * Sleep Stage Dashboard - JavaScript
 * Loads mock data and populates the interactive dashboard
 */

let mockData = null;

// Load mock data on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('mock_data.json');
        mockData = await response.json();
        
        // Populate dashboard
        populateDashboard();
        
        // Start animations
        animateStageTransitions();
        startTimeUpdates();
    } catch (error) {
        console.error('Error loading mock data:', error);
        loadSampleData();
    }
});

/**
 * Load sample data if mock_data.json not available
 */
function loadSampleData() {
    mockData = {
        'sleep_stages': generateSleepStages(),
        'current_metrics': {
            'current_time': new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
            'sleep_started': '23:30',
            'duration_so_far': '7h 45m',
            'quality_score': 82,
            'deep_sleep_percent': 22,
            'light_sleep_percent': 45,
            'wake_percent': 8,
            'cycles_completed': 4,
            'avg_heart_rate': 58,
            'avg_heart_rate_change': -2
        },
        'nightly_stats': {
            'total_sleep': '7h 45m',
            'wake': '37m (8%)',
            'light': '3h 29m (45%)',
            'deep': '1h 42m (22%)',
            'rem': '1h 57m (25%)',
            'efficiency': '98%',
            'quality_score': 82,
            'quality_label': 'Excellent'
        }
    };
    populateDashboard();
    animateStageTransitions();
    startTimeUpdates();
}

/**
 * Generate sample sleep stages
 */
function generateSleepStages() {
    const stages = [];
    const stagePattern = ['Wake', 'Light', 'Deep', 'Light', 'Wake', 'Light', 'Deep', 'Light', 'Deep', 'Light', 'Light'];
    const durations = [5, 10, 25, 15, 2, 12, 20, 18, 15, 25, 30];
    
    let minute = 0;
    stagePattern.forEach((stage, idx) => {
        for (let i = 0; i < durations[idx]; i++) {
            stages.push({
                'minute': minute,
                'stage': stage,
                'time': `${Math.floor(minute / 60):02d}:${minute % 60:02d}`,
                'quality_score': Math.random() * 30 + 60
            });
            minute++;
        }
    });
    
    return stages;
}

/**
 * Populate dashboard with mock data
 */
function populateDashboard() {
    if (!mockData) return;

    // Current metrics
    document.getElementById('timeDisplay').textContent = mockData.current_metrics.current_time;
    document.getElementById('currentStage').textContent = getLatestStage();
    document.getElementById('timeInStage').textContent = mockData.current_metrics.duration_so_far;
    document.getElementById('totalDuration').textContent = mockData.nightly_stats.total_sleep;
    document.getElementById('cyclesCount').textContent = mockData.current_metrics.cycles_completed;
    document.getElementById('qualityScoreMini').textContent = mockData.current_metrics.quality_score + '%';
    
    // Gauge score
    document.getElementById('gaugeScore').textContent = mockData.current_metrics.quality_score;
    
    // Sleep stats
    document.getElementById('totalSleep').textContent = mockData.nightly_stats.total_sleep;
    document.getElementById('efficiency').textContent = mockData.nightly_stats.efficiency;
    document.getElementById('cycles').textContent = mockData.current_metrics.cycles_completed;
    document.getElementById('heartRate').textContent = mockData.current_metrics.avg_heart_rate + ' bpm';
    
    // Calculate deep and light amounts
    const deepMatch = mockData.nightly_stats.deep.match(/(\d+h\s+\d+m)/);
    const lightMatch = mockData.nightly_stats.light.match(/(\d+h\s+\d+m)/);
    document.getElementById('deepAmount').textContent = deepMatch ? deepMatch[0] : '1h 42m';
    document.getElementById('lightAmount').textContent = lightMatch ? lightMatch[0] : '3h 29m';
    
    // Percentages
    document.getElementById('wakePercent').textContent = mockData.current_metrics.wake_percent + '%';
    document.getElementById('lightPercent').textContent = mockData.current_metrics.light_sleep_percent + '%';
    document.getElementById('deepPercent').textContent = mockData.current_metrics.deep_sleep_percent + '%';
    document.getElementById('remPercent').textContent = '25%';
    
    // Timeline
    generateTimeline();
    
    // Update status icon based on current stage
    updateStatusIcon();
}

/**
 * Get the latest sleep stage
 */
function getLatestStage() {
    if (!mockData || !mockData.sleep_stages || mockData.sleep_stages.length === 0) {
        return 'Deep Sleep';
    }
    
    const lastStage = mockData.sleep_stages[mockData.sleep_stages.length - 1];
    return lastStage.stage === 'Wake' ? 'Light Sleep' : lastStage.stage + ' Sleep';
}

/**
 * Update status icon based on stage
 */
function updateStatusIcon() {
    const stage = getLatestStage().split(' ')[0];
    const iconEl = document.getElementById('statusIcon');
    
    const icons = {
        'Wake': '👁️',
        'Light': '🌙',
        'Deep': '💤'
    };
    
    iconEl.textContent = icons[stage] || '💤';
}

/**
 * Generate timeline bars
 */
function generateTimeline() {
    if (!mockData || !mockData.sleep_stages) return;
    
    const timeline = document.getElementById('timelineCanvas');
    timeline.innerHTML = '';
    
    const stages = mockData.sleep_stages;
    const totalMinutes = stages.length;
    
    // Group consecutive stages
    let groups = [];
    let currentGroup = { stage: stages[0].stage, count: 1 };
    
    for (let i = 1; i < stages.length; i++) {
        if (stages[i].stage === currentGroup.stage) {
            currentGroup.count++;
        } else {
            groups.push(currentGroup);
            currentGroup = { stage: stages[i].stage, count: 1 };
        }
    }
    groups.push(currentGroup);
    
    // Create timeline bars
    groups.forEach((group, idx) => {
        const bar = document.createElement('div');
        bar.className = `timeline-bar ${group.stage.toLowerCase()}`;
        bar.style.flex = group.count;
        
        // Tooltip on hover
        bar.title = `${group.stage} - ${group.count}m`;
        bar.addEventListener('click', () => {
            console.log(`${group.stage}: ${group.count} minutes`);
        });
        
        timeline.appendChild(bar);
    });
}

/**
 * Animate stage transitions
 */
function animateStageTransitions() {
    if (!mockData || !mockData.sleep_stages) return;
    
    let currentIndex = 0;
    const stages = mockData.sleep_stages;
    
    setInterval(() => {
        if (currentIndex < stages.length) {
            const stage = stages[currentIndex];
            
            // Update display
            const stageNames = { 'Wake': '👁️', 'Light': '🌙', 'Deep': '💤' };
            document.getElementById('statusIcon').textContent = stageNames[stage.stage] || '💤';
            document.getElementById('currentStage').textContent = stage.stage + ' Sleep';
            
            currentIndex++;
        } else {
            currentIndex = Math.max(0, currentIndex - 1);
        }
    }, 3000); // Update every 3 seconds
}

/**
 * Update time display
 */
function startTimeUpdates() {
    function updateTime() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false
        });
        document.getElementById('timeDisplay').textContent = timeStr;
    }
    
    updateTime();
    setInterval(updateTime, 1000);
}

/**
 * Add interactivity to stat cards
 */
document.addEventListener('DOMContentLoaded', () => {
    const statCards = document.querySelectorAll('.stat-card');
    
    statCards.forEach(card => {
        card.addEventListener('click', function() {
            // Add click animation
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
            }, 100);
        });
    });
});

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        populateDashboard,
        generateTimeline,
        getLatestStage,
        updateStatusIcon
    };
}
