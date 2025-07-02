import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
import socket
import sys
import time
import pickle
import joblib

def run_kmeans_analysis():
    """Run K-means clustering analysis"""
    print("=== K-MEANS CLUSTERING ANALYSIS ===")
    
    # Check if saved model exists
    model_files = ['kmeans_model.pkl', 'kmeans_scaler.pkl', 'kmeans_results.pkl']
    if all(os.path.exists(f) for f in model_files):
        print("📁 Loading saved K-means model...")
        try:
            # Load saved model and results
            with open('kmeans_model.pkl', 'rb') as f:
                kmeans = pickle.load(f)
            with open('kmeans_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('kmeans_results.pkl', 'rb') as f:
                saved_results = pickle.load(f)
            
            print("✅ Loaded saved K-means model successfully!")
            print(f"   Best k: {saved_results['best_k']}")
            print(f"   Silhouette Score: {saved_results['silhouette_score']:.3f}")
            print(f"   Interpretation: {saved_results['interpretation']}")
            
            return saved_results
        except Exception as e:
            print(f"⚠️  Error loading saved model: {e}")
            print("🔄 Will retrain the model...")
    
    print("🔄 Training new K-means model...")
    
    df = pd.read_csv('personalized_learning_dataset.csv')

    df['attendance'] = df['Assignment_Completion_Rate']
    df['avg_score'] = df[['Quiz_Scores', 'Final_Exam_Score']].mean(axis=1)
    features = ['attendance', 'avg_score']

    for col in features:
        df[col] = df[col].fillna(df[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    k_range = range(2, 9)
    silhouette_scores = []

    print("Performing silhouette analysis...")
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"k={k}: Silhouette Score = {score:.3f}")

    best_k = k_range[np.argmax(silhouette_scores)]
    best_silhouette_score = max(silhouette_scores)

    print(f"Best k-value: {best_k} clusters")
    print(f"Best Silhouette Score: {best_silhouette_score:.3f}")

    # Interpret silhouette score
    if best_silhouette_score >= 0.71:
        interpretation = "Excellent"
    elif best_silhouette_score >= 0.51:
        interpretation = "Good"
    elif best_silhouette_score >= 0.26:
        interpretation = "Fair"
    else:
        interpretation = "Poor"

    print(f"Interpretation: {interpretation} clustering quality")

    plt.figure(figsize=(10, 6))
    colors = ['limegreen' if k == best_k else 'skyblue' for k in k_range]
    bars = plt.bar(k_range, silhouette_scores, color=colors, alpha=0.7, edgecolor='navy', width=0.6)

    for bar, score in zip(bars, silhouette_scores):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.01, f'{score:.3f}', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('Number of Clusters (k)', fontsize=11)
    plt.ylabel('Silhouette Score', fontsize=11)
    plt.title('Silhouette Score Analysis for Optimal k Selection', fontsize=14, fontweight='bold')
    plt.xticks(list(k_range))
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)

    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/silhouette_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    # KMeans clustering with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Calculate cluster characteristics and risk scores
    cluster_means = df.groupby('cluster')[features].mean()
    cluster_means['risk_score'] = (100 - cluster_means['attendance']) + (100 - cluster_means['avg_score'])

    # Assign meaningful labels based on risk score
    cluster_ranking = cluster_means['risk_score'].sort_values(ascending=True)
    label_mapping = {cluster_ranking.index[i]: i for i in range(len(cluster_ranking))}
    df['risk_label'] = df['cluster'].map(label_mapping)
    df['risk_score'] = (100 - df['attendance']) + (100 - df['avg_score'])

    # Color and label mappings
    color_mapping = {0: '#2E8B57', 1: '#FFD700', 2: '#FF8C00'}
    label_names = {0: 'Will Not Drop Out', 1: 'Likely to Drop Out', 2: 'Will Drop Out'}

    # Print cluster characteristics
    print(f"Cluster Characteristics:")
    for cluster_id in cluster_means.index:
        attendance = cluster_means.loc[cluster_id, 'attendance']
        avg_score = cluster_means.loc[cluster_id, 'avg_score']
        risk_score = cluster_means.loc[cluster_id, 'risk_score']
        count = len(df[df['cluster'] == cluster_id])
        percentage = count / len(df) * 100
        label = label_mapping[cluster_id]
        label_name = label_names[label]
        print(f"Cluster {cluster_id} ({label_name}): Students={count} ({percentage:.1f}%), "
              f"Attendance={attendance:.1f}%, Avg Score={avg_score:.1f}, Risk Score={risk_score:.1f}")

    print(f"Total students analyzed: {len(df)}")

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # PCA table for dashboard
    explained_var = pca.explained_variance_ratio_
    cum_var_2 = explained_var[:2].sum()
    pca_table = [
        ['PCA1', f'{explained_var[0]:.4f}', f'{explained_var[0]*100:.2f}%'],
        ['PCA2', f'{explained_var[1]:.4f}', f'{explained_var[1]*100:.2f}%'],
        ['Cumulative (PCA1+PCA2)', f'{cum_var_2:.4f}', f'{cum_var_2*100:.2f}%']
    ]

    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. PCA Cluster Plot
    colors = [color_mapping[label] for label in df['risk_label']]
    ax1.scatter(df['PCA1'], df['PCA2'], c=colors, s=60, alpha=0.7, edgecolor='k', linewidth=0.5)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[label], 
                                  markersize=10, label=label_names[label]) 
                       for label in sorted(label_mapping.values())]
    ax1.set_title(f'Student Dropout Prediction Clusters (k={best_k})', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PCA Component 1 ({explained_var[0]:.1%} variance)', fontsize=11)
    ax1.set_ylabel(f'PCA Component 2 ({explained_var[1]:.1%} variance)', fontsize=11)
    ax1.legend(handles=legend_elements, title='Dropout Prediction', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Attendance vs Score Scatter
    for label in sorted(label_mapping.values()):
        mask = df['risk_label'] == label
        ax2.scatter(df[mask]['attendance'], df[mask]['avg_score'], 
                    c=color_mapping[label], s=50, alpha=0.6, 
                    label=label_names[label], edgecolor='k', linewidth=0.3)
    ax2.set_xlabel('Attendance (%)', fontsize=11)
    ax2.set_ylabel('Average Score', fontsize=11)
    ax2.set_title('Student Performance by Dropout Prediction', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Cluster Size Distribution
    cluster_counts = [len(df[df['risk_label'] == label]) for label in sorted(label_mapping.values())]
    cluster_names = [label_names[label] for label in sorted(label_mapping.values())]
    cluster_colors = [color_mapping[label] for label in sorted(label_mapping.values())]
    bars = ax3.bar(cluster_names, cluster_counts, color=cluster_colors, alpha=0.8, edgecolor='k')
    ax3.set_ylabel('Number of Students', fontsize=11)
    ax3.set_title('Student Distribution by Dropout Prediction', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    # 4. Performance Metrics Comparison
    cluster_summary = df.groupby('risk_label')[features].mean()
    attendance_values = [cluster_summary.loc[label, 'attendance'] for label in sorted(label_mapping.values())]
    score_values = [cluster_summary.loc[label, 'avg_score'] for label in sorted(label_mapping.values())]
    x = np.arange(len(cluster_names))
    width = 0.35

    ax4.bar(x - width/2, attendance_values, width, label='Attendance (%)', 
            color='skyblue', alpha=0.8, edgecolor='k')
    ax4.bar(x + width/2, score_values, width, label='Average Score', 
            color='lightcoral', alpha=0.8, edgecolor='k')

    ax4.set_xlabel('Dropout Prediction', fontsize=11)
    ax4.set_ylabel('Percentage/Score', fontsize=11)
    ax4.set_title('Performance Metrics by Cluster', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cluster_names, rotation=45)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('static/enhanced_cluster_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Prepare cluster statistics for Flask
    cluster_stats_data = []
    for label in sorted(label_mapping.values()):
        mask = df['risk_label'] == label
        cluster_data = df[mask]
        stats = {
            'label': label,
            'name': label_names[label],
            'color': color_mapping[label],
            'count': len(cluster_data),
            'attendance': cluster_data['attendance'].mean(),
            'avg_score': cluster_data['avg_score'].mean(),
            'percentage': len(cluster_data) / len(df) * 100,
            'risk_score': cluster_data['risk_score'].mean(),
            'cluster_id': label
        }
        cluster_stats_data.append(stats)

    # Save model and results
    results = {
        'best_k': best_k,
        'silhouette_score': best_silhouette_score,
        'interpretation': interpretation,
        'pca_table': pca_table,
        'explained_var': explained_var,
        'cum_var_2': cum_var_2,
        'cluster_stats': cluster_stats_data
    }
    
    print("💾 Saving K-means model and results...")
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    with open('kmeans_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('kmeans_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✅ K-means model saved successfully!")
    
    return results

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    # Run K-means analysis
    kmeans_results = run_kmeans_analysis()
    
    return render_template('index.html',
        silhouette_img='static/silhouette_analysis.png',
        cluster_img='static/enhanced_cluster_analysis.png',
        cluster_stats=kmeans_results['cluster_stats'],
        best_k=kmeans_results['best_k'],
        silhouette_score=kmeans_results['silhouette_score'],
        interpretation=kmeans_results['interpretation'],
        pca_table=kmeans_results['pca_table'],
        explained_var=kmeans_results['explained_var'],
        cum_var_2=kmeans_results['cum_var_2']
    )

if __name__ == '__main__':
    # Run K-means analysis first
    run_kmeans_analysis()
    
    # Then start Flask app with better port management
    import webbrowser
    import time
    
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
        print("💡 Try running: python k-means.py")
        sys.exit(1)