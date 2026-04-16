"""
Mock Sleep Data Generator
Generates realistic sleep stage data for visualization
"""

import json
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sleep_session():
    """Generate a realistic 8-hour sleep session with sleep stage transitions"""
    
    # Sleep stage progression (realistic pattern)
    sleep_pattern = [
        {'stage': 'Wake', 'duration': 5},      # 5 min falling asleep
        {'stage': 'Light', 'duration': 10},    # 10 min light sleep
        {'stage': 'Deep', 'duration': 25},     # 25 min deep sleep (first cycle)
        {'stage': 'Light', 'duration': 15},    # back to light
        {'stage': 'Wake', 'duration': 2},      # brief awakening
        {'stage': 'Light', 'duration': 12},    # 12 min light
        {'stage': 'Deep', 'duration': 20},     # 20 min deep (second cycle)
        {'stage': 'Light', 'duration': 18},    # 18 min light
        {'stage': 'Deep', 'duration': 15},     # 15 min deep (third cycle)
        {'stage': 'Light', 'duration': 25},    # 25 min light (longer in late cycles)
        {'stage': 'Deep', 'duration': 10},     # 10 min deep (shorter in late cycles)
        {'stage': 'Light', 'duration': 30},    # 30 min light (mostly light in last hour)
        {'stage': 'Wake', 'duration': 10},     # 10 min waking up
    ]
    
    stages_data = []
    current_minute = 0
    
    for cycle in sleep_pattern:
        for minute in range(cycle['duration']):
            stages_data.append({
                'minute': current_minute,
                'stage': cycle['stage'],
                'time': f"{current_minute // 60:02d}:{current_minute % 60:02d}",
                'quality_score': np.random.randint(70, 95) if cycle['stage'] == 'Deep' else np.random.randint(60, 85)
            })
            current_minute += 1
    
    return stages_data

def generate_current_metrics():
    """Generate current sleep metrics"""
    return {
        'current_time': datetime.now().strftime("%H:%M"),
        'sleep_started': '23:30',
        'duration_so_far': '7h 45m',
        'quality_score': 82,
        'deep_sleep_percent': 22,
        'light_sleep_percent': 45,
        'wake_percent': 8,
        'cycles_completed': 4,
        'avg_heart_rate': 58,
        'avg_heart_rate_change': -2  # compared to awake
    }

def generate_nightly_stats():
    """Generate statistics for the night"""
    total_minutes = 465  # 7h 45m
    
    wake_minutes = int(total_minutes * 0.08)
    light_minutes = int(total_minutes * 0.45)
    deep_minutes = int(total_minutes * 0.22)
    rem_minutes = total_minutes - wake_minutes - light_minutes - deep_minutes
    
    return {
        'total_sleep': f"{total_minutes // 60}h {total_minutes % 60}m",
        'wake': f"{wake_minutes}m ({wake_minutes/total_minutes*100:.0f}%)",
        'light': f"{light_minutes}m ({light_minutes/total_minutes*100:.0f}%)",
        'deep': f"{deep_minutes}m ({deep_minutes/total_minutes*100:.0f}%)",
        'rem': f"{rem_minutes}m ({rem_minutes/total_minutes*100:.0f}%)",
        'efficiency': f"{(total_minutes / (total_minutes + 15) * 100):.0f}%",
        'quality_score': 82,
        'quality_label': 'Excellent'
    }

def save_mock_data():
    """Save mock data to JSON file"""
    data = {
        'sleep_stages': generate_sleep_session(),
        'current_metrics': generate_current_metrics(),
        'nightly_stats': generate_nightly_stats(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to frontend folder
    output_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'mock_data.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Mock data saved to: {output_path}")
    return data

if __name__ == '__main__':
    data = save_mock_data()
    print(f"\n📊 Generated Data Summary:")
    print(f"   Total timestamps: {len(data['sleep_stages'])}")
    print(f"   Sleep started: {data['current_metrics']['sleep_started']}")
    print(f"   Current quality score: {data['current_metrics']['quality_score']}/100")
    print(f"   Cycles completed: {data['current_metrics']['cycles_completed']}")
