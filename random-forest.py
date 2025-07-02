import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from flask import Flask, render_template, request, jsonify
import os
import socket
import webbrowser
import time
import sys
import pickle
import joblib

def run_random_forest_analysis():
    """Run Random Forest classification analysis"""
    print("=== RANDOM FOREST CLASSIFICATION ANALYSIS ===")
    
    # Check if saved model exists
    model_files = ['rf_model.pkl', 'rf_scaler.pkl', 'rf_encoder.pkl', 'rf_results.pkl']
    if all(os.path.exists(f) for f in model_files):
        print("📁 Loading saved Random Forest model...")
        try:
            # Load saved model and results
            with open('rf_model.pkl', 'rb') as f:
                best_rf = pickle.load(f)
            with open('rf_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('rf_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            with open('rf_results.pkl', 'rb') as f:
                saved_results = pickle.load(f)
            
            print("✅ Loaded saved Random Forest model successfully!")
            print(f"   Accuracy: {saved_results['accuracy']:.4f} ({saved_results['accuracy']*100:.2f}%)")
            print(f"   F1-Score: {saved_results['f1_score']:.4f} ({saved_results['f1_score']*100:.2f}%)")
            
            features = [
                'Time_Spent_on_Videos',
                'Quiz_Attempts', 
                'Quiz_Scores',
                'Forum_Participation',
                'Assignment_Completion_Rate',
                'Final_Exam_Score',
                'Feedback_Score'
            ]
            
            return saved_results, scaler, le, features
        except Exception as e:
            print(f"⚠️  Error loading saved model: {e}")
            print("🔄 Will retrain the model...")
    
    print("🔄 Training new Random Forest model...")
    
    # Load training and testing datasets
    print("Loading datasets...")
    train_df = pd.read_csv('csv_80_percent.csv')
    test_df = pd.read_csv('csv_20_percent.csv')

    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")

    # Features and target
    features = [
        'Time_Spent_on_Videos',
        'Quiz_Attempts', 
        'Quiz_Scores',
        'Forum_Participation',
        'Assignment_Completion_Rate',
        'Final_Exam_Score',
        'Feedback_Score'
    ]
    target = 'Dropout_Likelihood'

    # Encode target
    le = LabelEncoder()
    train_df['Dropout_Binary'] = le.fit_transform(train_df[target])
    test_df['Dropout_Binary'] = le.transform(test_df[target])

    # Prepare data
    X_train = train_df[features]
    y_train = train_df['Dropout_Binary']
    X_test = test_df[features]
    y_test = test_df['Dropout_Binary']

    print(f"Class distribution in training: {np.bincount(y_train)}")
    print(f"Class distribution in testing: {np.bincount(y_test)}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Address class imbalance using SMOTE
    print("Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

    # Optimized hyperparameter grid for faster execution
    param_grid = {
        'n_estimators': [200, 300],  # Reduced from [300, 400, 500]
        'max_depth': [20, 25],       # Reduced from [25, 30, 35]
        'min_samples_split': [2],    # Reduced from [2, 3]
        'min_samples_leaf': [1],     # Reduced from [1, 2]
        'max_features': ['sqrt'],    # Reduced from ['sqrt', 'log2']
        'bootstrap': [True]          # Reduced from [True, False]
    }

    print("Starting hyperparameter optimization...")
    print(f"Total combinations to test: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['bootstrap'])}")
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)  # Reduced cv from 5 to 3, verbose=0
    gs.fit(X_train_resampled, y_train_resampled)

    best_rf = gs.best_estimator_
    y_pred = best_rf.predict(X_test_scaled)
    y_pred_proba = best_rf.predict_proba(X_test_scaled)

    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    pos_label_index = list(le.classes_).index('Yes')
    precision = precision_score(y_test, y_pred, pos_label=pos_label_index)
    recall = recall_score(y_test, y_pred, pos_label=pos_label_index)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label_index)

    print(f'Random Forest Results:')
    print(f'Best Parameters: {gs.best_params_}')
    print(f'Random Forest Accuracy: {acc:.4f} ({acc*100:.2f}%)')
    print(f"Precision (for 'Yes'): {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (for 'Yes'): {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score (for 'Yes'): {f1:.4f} ({f1*100:.2f}%)")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f'Feature Importance:')
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/confusion_matrix_plot.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    colors = ['#2E8B57' if x > 0.1 else '#FFD700' if x > 0.05 else '#FF8C00' 
              for x in feature_importance['importance']]
    bars = plt.barh(feature_importance['feature'], feature_importance['importance'], 
                    color=colors, alpha=0.8, edgecolor='navy')
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()

    # Add value labels on bars
    for bar, importance in zip(bars, feature_importance['importance']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{importance:.3f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('static/feature_importance_plot.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Save detailed predictions
    test_results = test_df.copy()
    test_results['Predicted_Dropout'] = le.inverse_transform(y_pred)
    test_results['Prediction_Probability'] = y_pred_proba[:, 1]  # Probability of 'Yes'
    test_results['Correct_Prediction'] = (test_results['Dropout_Likelihood'] == test_results['Predicted_Dropout'])

    # Calculate prediction statistics
    total_predictions = len(test_results)
    correct_predictions = test_results['Correct_Prediction'].sum()
    incorrect_predictions = total_predictions - correct_predictions

    # Dropout prediction breakdown
    actual_dropouts = test_results[test_results['Dropout_Likelihood'] == 'Yes']
    predicted_dropouts = test_results[test_results['Predicted_Dropout'] == 'Yes']
    true_positives = test_results[(test_results['Dropout_Likelihood'] == 'Yes') & 
                                 (test_results['Predicted_Dropout'] == 'Yes')]
    false_positives = test_results[(test_results['Dropout_Likelihood'] == 'No') & 
                                  (test_results['Predicted_Dropout'] == 'Yes')]
    false_negatives = test_results[(test_results['Dropout_Likelihood'] == 'Yes') & 
                                  (test_results['Predicted_Dropout'] == 'No')]

    print(f'Prediction Breakdown:')
    print(f'Total test samples: {total_predictions}')
    print(f'Correct predictions: {correct_predictions} ({correct_predictions/total_predictions*100:.2f}%)')
    print(f'Incorrect predictions: {incorrect_predictions} ({incorrect_predictions/total_predictions*100:.2f}%)')
    print(f'Actual dropouts: {len(actual_dropouts)}')
    print(f'Predicted dropouts: {len(predicted_dropouts)}')
    print(f'True Positives: {len(true_positives)}')
    print(f'False Positives: {len(false_positives)}')
    print(f'False Negatives: {len(false_negatives)}')

    # Save results
    test_results.to_csv('random_forest_test_predictions.csv', index=False)
    print('Detailed predictions saved to random_forest_test_predictions.csv')

    # Prepare data for web interface
    rf_results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_params': gs.best_params_,
        'feature_importance': feature_importance.to_dict('records'),
        'prediction_stats': {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions,
            'actual_dropouts': len(actual_dropouts),
            'predicted_dropouts': len(predicted_dropouts),
            'true_positives': len(true_positives),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives)
        },
        'test_results': test_results.head(20).to_dict('records')  # First 20 results for display
    }

    # Save model and results
    print("💾 Saving Random Forest model and results...")
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)
    with open('rf_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('rf_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('rf_results.pkl', 'wb') as f:
        pickle.dump(rf_results, f)
    print("✅ Random Forest model saved successfully!")

    return rf_results, scaler, le, features

# Flask app setup
app = Flask(__name__)

# Global variables for the trained model
rf_results = None
scaler = None
le = None
features = None
best_rf = None

@app.route('/')
def index():
    global rf_results
    if rf_results is None:
        rf_results, _, _, _ = run_random_forest_analysis()
    
    return render_template('index.html', rf_results=rf_results)

@app.route('/upload', methods=['POST'])
def upload_file():
    global scaler, le, features, best_rf
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read uploaded file
        df_upload = pd.read_csv(file)
        
        # Check if required columns exist
        missing_cols = [col for col in features if col not in df_upload.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'})
        
        # Prepare data
        X_upload = df_upload[features]
        X_upload_scaled = scaler.transform(X_upload)
        
        # Make predictions
        predictions = best_rf.predict(X_upload_scaled)
        probabilities = best_rf.predict_proba(X_upload_scaled)
        
        # Prepare results
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

@app.route('/manual-predict', methods=['POST'])
def manual_predict():
    global scaler, le, features, best_rf
    
    try:
        # Get form data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'time_spent_videos', 'quiz_attempts', 'quiz_scores', 
            'forum_participation', 'assignment_completion', 
            'final_exam_score', 'feedback_score'
        ]
        
        missing_fields = [field for field in required_fields if field not in data or data[field] == '']
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'})
        
        # Create input array
        input_data = np.array([[
            float(data['time_spent_videos']),
            float(data['quiz_attempts']),
            float(data['quiz_scores']),
            float(data['forum_participation']),
            float(data['assignment_completion']),
            float(data['final_exam_score']),
            float(data['feedback_score'])
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = best_rf.predict(input_scaled)[0]
        probabilities = best_rf.predict_proba(input_scaled)[0]
        
        # Prepare result
        result = {
            'predicted_dropout': le.inverse_transform([prediction])[0],
            'dropout_probability': f'{probabilities[1]*100:.2f}%',
            'stay_probability': f'{probabilities[0]*100:.2f}%',
            'confidence': f'{max(probabilities)*100:.2f}%'
        }
        
        return jsonify({'success': True, 'result': result})
    
    except ValueError as e:
        return jsonify({'error': 'Invalid numeric values. Please check your input.'})
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'})

@app.route('/api/rf-results')
def get_rf_results():
    global rf_results
    if rf_results is None:
        rf_results, _, _, _ = run_random_forest_analysis()
    return jsonify(rf_results)

if __name__ == '__main__':
    # Run Random Forest analysis first
    rf_results, scaler, le, features = run_random_forest_analysis()
    
    # Load the best model for predictions
    with open('rf_model.pkl', 'rb') as f:
        best_rf = pickle.load(f)
    
    # Then start Flask app with better port management
    import webbrowser
    import time
    import sys
    
    # Try multiple ports
    ports_to_try = [5000, 5001, 5002, 5003, 5004, 5005, 8000, 8001, 8002, 8080, 8081, 8082]
    port = None
    
    for p in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', p))
                if result != 0:  # Port is available
                    port = p
                    break
        except:
            continue
    
    if port is None:
        print("❌ No available ports found. Please close other applications and try again.")
        sys.exit(1)
    
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Starting Flask app on port {port}")
    print(f"📊 Access the dashboard at: {url}")
    
    # Try to open browser after a short delay
    def open_browser():
        time.sleep(2)  # Wait for server to start
        try:
            webbrowser.open_new(url)
            print("✅ Browser opened automatically")
        except:
            print("⚠️  Could not open browser automatically.")
            print(f"   Please open this URL manually: {url}")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        app.run(debug=False, port=port, host='127.0.0.1', use_reloader=False)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        print("💡 Try running: python random-forest.py")
        sys.exit(1)