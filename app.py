import json
from datetime import datetime
import os

CONFIG_HISTORY_FILE = 'config_history.json'

def load_config_history():
    if os.path.exists(CONFIG_HISTORY_FILE):
        with open(CONFIG_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_config_history(config, timestamp, description):
    history = load_config_history()
    history.append({
        'config': config,
        'timestamp': timestamp,
        'description': description
    })
    # Keep only the last 50 versions
    history = history[-50:]
    with open(CONFIG_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

@app.route('/api/config/history')
def get_config_history():
    try:
        history = load_config_history()
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/config/update', methods=['POST'])
def update_config():
    try:
        data = request.json
        config = data['config']
        timestamp = data['timestamp']
        description = data['description']
        
        # Save the new configuration
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save to history
        save_config_history(config, timestamp, description)
        
        return jsonify({
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }) 