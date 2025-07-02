import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Helper to load a pickle file or return None
def load_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

# Load K-means and RF results
kmeans_results = load_pickle('kmeans_results.pkl')
rf_results = load_pickle('rf_results.pkl')

@app.route('/')
def index():
    kmeans_error = None
    rf_error = None
    if not kmeans_results:
        kmeans_error = 'K-means results not available. Please run k-means.py first.'
    if not rf_results:
        rf_error = 'Random Forest results not available. Please run random-forest.py first.'
    return render_template(
        'index.html',
        best_k=kmeans_results['best_k'] if kmeans_results else None,
        silhouette_score=kmeans_results['silhouette_score'] if kmeans_results else None,
        interpretation=kmeans_results['interpretation'] if kmeans_results else None,
        model_metrics=kmeans_results['model_metrics'] if kmeans_results else None,
        cluster_stats=kmeans_results['cluster_stats'] if kmeans_results else None,
        rf_results=rf_results,
        kmeans_error=kmeans_error,
        rf_error=rf_error
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    import pandas as pd
    import numpy as np
    # Load the trained model and scaler
    try:
        with open('rf_model.pkl', 'rb') as f:
            best_rf = pickle.load(f)
        with open('rf_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('rf_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    except Exception as e:
        return jsonify({'error': f'Could not load model: {str(e)}'})
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    try:
        df_upload = pd.read_csv(file)
        features = [
            'Time_Spent_on_Videos',
            'Quiz_Attempts',
            'Quiz_Scores',
            'Forum_Participation',
            'Assignment_Completion_Rate',
            'Final_Exam_Score',
            'Feedback_Score'
        ]
        missing_cols = [col for col in features if col not in df_upload.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'})
        X_upload = df_upload[features]
        X_upload_scaled = scaler.transform(X_upload)
        predictions = best_rf.predict(X_upload_scaled)
        probabilities = best_rf.predict_proba(X_upload_scaled)
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'student_id': f'Upload_{i+1}',
                'predicted_dropout': le.inverse_transform([pred])[0],
                'dropout_probability': f'{prob[1]*100:.2f}%',
                'stay_probability': f'{prob[0]*100:.2f}%'
            })
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

if __name__ == '__main__':
    import webbrowser
    import socket
    import threading
    import time
    # Find available port
    ports_to_try = [5000, 5001, 5002, 5003, 8000, 8080]
    port = None
    for p in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', p))
                if result != 0:
                    port = p
                    break
        except:
            continue
    if port is None:
        print('No available ports found.')
        exit(1)
    url = f'http://127.0.0.1:{port}'
    def open_browser():
        time.sleep(2)
        webbrowser.open_new(url)
    threading.Thread(target=open_browser, daemon=True).start()
    print(f'Unified dashboard running at {url}')
    app.run(debug=False, port=port, host='127.0.0.1', use_reloader=False) 