# Student Learning Analytics - Clustering and Dropout Prediction

A comprehensive machine learning pipeline for analyzing student learning patterns, performing unsupervised clustering to identify learning behaviors, and predicting student dropout likelihood using supervised learning algorithms.

## 🎯 Overview

This project combines **K-Means clustering** and **Random Forest classification** to provide insights into student learning analytics. It features an interactive web dashboard for real-time analysis, batch predictions, and manual parameter entry for dropout prediction.

## 🚀 Key Features

- **Unsupervised Learning**: K-Means clustering with automatic optimal cluster selection
- **Supervised Learning**: Random Forest classifier for dropout prediction
- **Interactive Dashboard**: Flask-based web interface with visualizations
- **Batch Processing**: CSV upload for both clustering and classification
- **Manual Prediction**: Interactive form for individual student predictions
- **Model Persistence**: Automatic saving/loading of trained models
- **Dynamic Port Management**: Automatic port selection to avoid conflicts

## 📊 AI Models & Algorithms

### K-Means Clustering
- **Purpose**: Group students by learning behavior patterns
- **Features Used**: Attendance rates, quiz scores, assignment completion
- **Optimization**: Silhouette score analysis for optimal k selection (k=2 to 8)
- **Visualization**: PCA-reduced 2D scatter plots with cluster assignments
- **Risk Assessment**: Automatic labeling based on dropout risk (Will Not Drop Out, Likely to Drop Out, Will Drop Out)

### Random Forest Classifier
- **Purpose**: Predict student dropout likelihood (Yes/No)
- **Features**: Time spent on videos, quiz attempts, scores, forum participation, assignment completion, final exam scores, feedback scores
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Reduces high-dimensional features to 2D for visualization
- **Variance Explanation**: Tracks cumulative variance captured by principal components

## 🛠️ Technology Stack

- **Language**: Python 3.x
- **Web Framework**: Flask
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML5, CSS3, Jinja2 templating
- **Model Persistence**: pickle, joblib

## 📁 Project Structure

```
k-means/
├── k-means.py                    # Main clustering + web app
├── random-forest.py              # Random Forest classification + web app
├── run_both_analyses.py          # Unified dashboard for both models
├── unified_dashboard.py          # Enhanced dashboard
├── requirements.txt              # Python dependencies
├── personalized_learning_dataset.csv  # Main dataset (10,000 students)
├── csv_80_percent.csv            # Training set (8,000 students)
├── csv_20_percent.csv            # Test set (2,000 students)
├── static/                       # Generated visualizations and CSS
│   ├── style.css
│   ├── silhouette_analysis.png
│   ├── enhanced_cluster_analysis.png
│   └── confusion_matrix_plot.png
├── templates/                    # HTML templates
│   └── index.html
├── *.pkl                         # Serialized models and results
└── README.md                     # This file
```

## 🚦 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd k-means
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Applications

#### Option 1: K-Means Clustering Dashboard
```bash
python k-means.py
```
- Launches K-Means clustering analysis
- Opens web dashboard at `http://127.0.0.1:<available_port>`
- Features: Cluster visualization, silhouette analysis, batch clustering

#### Option 2: Random Forest Classification Dashboard
```bash
python random-forest.py
```
- Launches Random Forest classification
- Opens web dashboard with classification features
- Features: Batch classification, manual prediction, performance metrics

#### Option 3: Unified Dashboard (Both Models)
```bash
python run_both_analyses.py
```
- Comprehensive dashboard with both K-Means and Random Forest
- Switch between clustering and classification modes
- Complete analytics suite

## 📋 Features in Detail

### K-Means Clustering Features
- **Automatic k Selection**: Silhouette score analysis to find optimal cluster count
- **Risk Labeling**: Students categorized as "Will Not Drop Out", "Likely to Drop Out", "Will Drop Out"
- **Visual Analytics**:
  - Silhouette score plot for cluster quality assessment
  - PCA 2D cluster visualization with color-coded risk levels
  - Student performance scatter plots
  - Cluster size distribution charts
  - Performance metrics comparison

### Random Forest Classification Features
- **Comprehensive Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Classification report with per-class metrics
  - Confusion matrix visualization
- **Prediction Methods**:
  - **Batch Upload**: CSV file processing for multiple students
  - **Manual Entry**: Interactive form for individual predictions
- **Input Validation**: Automatic checking for missing features and invalid data

### Web Dashboard Features
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Immediate results for predictions
- **File Upload Support**: Drag-and-drop CSV upload
- **Dynamic Port Selection**: Automatically finds available ports (5000-8082)
- **Error Handling**: Comprehensive error messages and validation

## 📊 Data Schema

### Input Features
- `Time_Spent_on_Videos`: Minutes spent watching video content
- `Quiz_Attempts`: Number of quiz attempts
- `Quiz_Scores`: Average quiz scores (0-100)
- `Forum_Participation`: Forum interaction count
- `Assignment_Completion_Rate`: Percentage of assignments completed
- `Final_Exam_Score`: Final exam score (0-100)
- `Feedback_Score`: Student feedback rating

### Output Targets
- **Clustering**: Risk level assignment (0: Low Risk, 1: Medium Risk, 2: High Risk)
- **Classification**: Dropout likelihood (Yes/No)

## 🎯 Model Performance

### K-Means Clustering
- **Quality Metric**: Silhouette Score (typically 0.3-0.7)
- **Interpretation**:
  - ≥0.71: Excellent clustering
  - ≥0.51: Good clustering
  - ≥0.26: Fair clustering
  - <0.26: Poor clustering

### Random Forest Classification
- **Typical Performance**:
  - Accuracy: 85-95%
  - Precision: 80-90%
  - Recall: 75-85%
  - F1-Score: 77-88%
- **Class Balancing**: SMOTE improves minority class prediction

## 🔧 Advanced Features

### Model Persistence
- Trained models automatically saved as `.pkl` files
- Faster startup times by loading pre-trained models
- Automatic retraining if model files are corrupted

### Hyperparameter Optimization
- **K-Means**: Automatic k selection via silhouette analysis
- **Random Forest**: GridSearchCV for optimal parameters:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [10, 20, None]
  - `min_samples_split`: [2, 5, 10]

### Data Preprocessing
- **Missing Values**: Median imputation for numerical features
- **Feature Scaling**: StandardScaler for normalization
- **Dimensionality Reduction**: PCA for visualization
- **Class Imbalance**: SMOTE for minority class oversampling

## 📈 Usage Examples

### Batch Clustering
1. Navigate to K-Means dashboard
2. Upload CSV with required features
3. View cluster assignments and risk levels
4. Download results with cluster labels

### Individual Dropout Prediction
1. Navigate to Random Forest dashboard
2. Fill in student data manually
3. Get instant dropout prediction
4. View confidence scores and risk factors

### Performance Analysis
1. View confusion matrix for classification accuracy
2. Analyze silhouette scores for cluster quality
3. Compare metrics across different model configurations

## 🚨 Important Notes

- **Port Conflicts**: The app automatically finds available ports if default ports are busy
- **Data Requirements**: Ensure CSV files contain all required features
- **Model Retraining**: Models are automatically retrained if saved files are missing or corrupted
- **Browser Compatibility**: Works best with modern browsers (Chrome, Firefox, Safari)

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new features. Please ensure:
- Code follows Python PEP 8 guidelines
- New features include appropriate tests
- Documentation is updated for new functionality

## 📞 Support

For questions, issues, or improvements:
- **Phone**: +977 9843735448
- **Email**: [Contact information]
- **Issues**: Report via repository issue tracker

## 📄 License

This project is open-source and available under the MIT License.

---

**Last Updated**: April 2026  
**Version**: 1.0  
**Maintainer**: Project Team
