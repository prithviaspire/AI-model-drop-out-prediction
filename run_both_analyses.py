#!/usr/bin/env python3
"""
Unified Student Learning Analytics Dashboard
Runs both K-means clustering and Random Forest classification automatically
"""

import subprocess
import sys
import os
import time
import importlib.util

def run_analysis():
    print("=" * 60)
    print("STUDENT LEARNING ANALYTICS DASHBOARD")
    print("=" * 60)
    print("Starting comprehensive analysis...")
    
    # Step 1: Run K-means analysis (without Flask)
    print("\n1. Running K-means clustering analysis...")
    try:
        # Import and run K-means analysis directly
        spec = importlib.util.spec_from_file_location("k_means", "k-means.py")
        k_means_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(k_means_module)
        kmeans_results = k_means_module.run_kmeans_analysis()
        print("✅ K-means analysis completed!")
        print(f"   Best k: {kmeans_results['best_k']}")
        print(f"   Silhouette Score: {kmeans_results['silhouette_score']:.3f}")
        print(f"   Interpretation: {kmeans_results['interpretation']}")
    except Exception as e:
        print(f"❌ K-means analysis failed: {e}")
        return False
    
    # Step 2: Run Random Forest analysis (without Flask)
    print("\n2. Running Random Forest classification analysis...")
    try:
        # Import and run Random Forest analysis directly
        spec = importlib.util.spec_from_file_location("random_forest", "random-forest.py")
        random_forest_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(random_forest_module)
        rf_results, _, _, _ = random_forest_module.run_random_forest_analysis()
        print("✅ Random Forest analysis completed!")
        print(f"   Accuracy: {rf_results['accuracy']:.4f} ({rf_results['accuracy']*100:.2f}%)")
        print(f"   F1-Score: {rf_results['f1_score']:.4f} ({rf_results['f1_score']*100:.2f}%)")
    except Exception as e:
        print(f"❌ Random Forest analysis failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Both K-means and Random Forest analyses have been executed.")
    print("\nGenerated files:")
    print("- static/silhouette_analysis.png")
    print("- static/enhanced_cluster_analysis.png")
    print("- static/confusion_matrix_plot.png")
    print("- static/feature_importance_plot.png")
    print("- random_forest_test_predictions.csv")
    print("\nSaved models (for future runs):")
    print("- kmeans_model.pkl, kmeans_scaler.pkl, kmeans_results.pkl")
    print("- rf_model.pkl, rf_scaler.pkl, rf_encoder.pkl, rf_results.pkl")
    print("\nTo view the web dashboard, run:")
    print("  python k-means.py     (for K-means dashboard)")
    print("  python random-forest.py (for Random Forest dashboard)")
    print("\n💡 Next time you run this, it will load saved models instantly!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    run_analysis() 