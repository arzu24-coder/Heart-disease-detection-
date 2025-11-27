# Heart Disease Prediction

A machine learning project to predict heart disease using various classification algorithms. This project analyzes heart disease data and builds predictive models to help identify patients at risk of heart disease.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a comprehensive heart disease prediction system using machine learning techniques. The analysis includes data exploration, visualization, preprocessing, and model building to predict whether a patient has heart disease based on various medical attributes.

## 📊 Dataset

The project uses the `heart.csv` dataset containing medical records with the following key attributes:

- **Age**: Age of the patient
- **Sex**: Gender (0 = female, 1 = male)
- **CP**: Chest pain type (0-3)
- **Trestbps**: Resting blood pressure
- **Chol**: Serum cholesterol level
- **FBS**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **Restecg**: Resting electrocardiographic results (0-2)
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise induced angina (1 = yes, 0 = no)
- **Oldpeak**: Depression induced by exercise relative to rest
- **Slope**: Slope of the peak exercise ST segment
- **CA**: Number of major vessels colored by fluoroscopy
- **Thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- **Target**: Heart disease diagnosis (1 = disease, 0 = no disease)

## ✨ Features

### Data Analysis
- Comprehensive dataset exploration
- Statistical analysis and data profiling
- Missing value analysis
- Data type verification

### Data Visualization
- Correlation heatmap analysis
- Feature relationship visualization
- Distribution plots
- Model performance comparison charts

### Data Preprocessing
- Categorical variable encoding using dummy variables
- Feature scaling with StandardScaler
- Feature selection based on correlation analysis

### Machine Learning Models
- K-Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Heart-Disease-Detection.git
cd Heart-Disease-Detection
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Ensure you have the `heart.csv` dataset in the project directory.

## 💻 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "Heart Disease Prediction.ipynb"
```

2. Run all cells sequentially to:
   - Load and explore the dataset
   - Visualize data patterns
   - Preprocess the data
   - Train multiple ML models
   - Compare model performance

## 🤖 Machine Learning Models

### 1. K-Neighbors Classifier
- **Configuration**: k=12 neighbors
- **Validation**: 10-fold cross-validation
- **Purpose**: Classification based on similarity to neighboring data points

### 2. Decision Tree Classifier
- **Configuration**: max_depth=3 (optimized through testing depths 1-10)
- **Validation**: 10-fold cross-validation
- **Purpose**: Rule-based classification with interpretable decision paths

### 3. Random Forest Classifier
- **Configuration**: n_estimators=90 (optimized through testing 10-100 estimators)
- **Validation**: 5-fold cross-validation
- **Purpose**: Ensemble method combining multiple decision trees

## 📈 Results

The project evaluates model performance using cross-validation accuracy scores:

- **KNeighbors Classifier (k=12)**: Cross-validated accuracy reported
- **Decision Tree Classifier (max_depth=3)**: Cross-validated accuracy reported
- **Random Forest Classifier (n_estimators=90)**: Cross-validated accuracy reported

*Note: Run the notebook to see specific accuracy percentages*

## 📊 Visualizations

The project includes several visualization components:

1. **Correlation Heatmap**: Shows relationships between all features
2. **Decision Tree Performance**: Accuracy vs. tree depth analysis
3. **Random Forest Performance**: Accuracy vs. number of estimators analysis

## 🔧 Key Technical Details

### Data Preprocessing Steps:
1. **Categorical Encoding**: Convert binary and categorical features (sex, cp, fbs, restecg, exang, slope, ca, thal) to dummy variables
2. **Feature Scaling**: Standardize numerical features (age, trestbps, chol, thalach, oldpeak)
3. **Feature Selection**: Use correlation analysis to identify important features

### Model Optimization:
- **Decision Tree**: Grid search over max_depth values (1-10)
- **Random Forest**: Grid search over n_estimators values (10-100, step=10)
- **Cross-Validation**: Multiple validation strategies to ensure robust performance estimates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This project is for educational and research purposes. Always consult with medical professionals for actual health-related decisions.
