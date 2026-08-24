# Medical Insurance Cost Prediction

A machine learning project to predict medical insurance costs using patient demographics and health metrics.

## 📋 Project Overview

This project builds a regression model to predict medical insurance charges based on patient information including age, sex, BMI, number of children, smoking status, and region. The goal is to identify key factors driving healthcare costs and build an accurate predictive model.

##  Objectives

- Explore and analyze the medical insurance dataset
- Engineer meaningful features to improve model performance
- Build and optimize regression models to predict insurance charges
- Identify the most important factors affecting medical costs

## 📊 Dataset

**Source:** Medical Cost Personal Datasets  
**Size:** 2,772 records  
**Features:**
- `age`: Age of the patient
- `sex`: Gender (female/male)
- `bmi`: Body Mass Index
- `children`: Number of children covered by insurance
- `smoker`: Smoking status (yes/no)
- `region`: Geographic region (northeast, northwest, southeast, southwest)
- `charges`: Medical insurance costs (target variable)

## 🔧 Methodology

### Data Preprocessing
1. **Categorical Encoding:**
   - Mapped `smoker`: yes→1, no→0
   - Mapped `sex`: female→1, male→0
   - OneHotEncoder applied to `region` (drop first to avoid multicollinearity)

2. **Feature Engineering:**
   - `high_bmi`: Binary indicator for BMI ≥ 30 (obese)
   - `young`: Binary indicator for age ≤ 30
   - `healthy_bmi`: Binary indicator for BMI < 25 (normal weight)
   - `bmi*age`: Interaction term between BMI and age
   - Removed `senior` feature (zero importance in initial model)

### Model Development

**Algorithm:** Random Forest Regressor

**Hyperparameter Tuning:**
- Used `RandomizedSearchCV` with 5-fold cross-validation
- Optimized parameters:
  - `n_estimators`: [300, 500, 700]
  - `max_depth`: [10, 20, 30, 40, 50, None]
  - `min_samples_split`: [2, 5, 10, 15]
  - `min_samples_leaf`: [1, 2, 4, 6, 10]
  - `max_features`: ['sqrt', 'log2', 0.5, 0.8, None]

**Best Parameters:**
```python
{
    'n_estimators': 500,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'random_state': 42
}
```

## 📈 Results

### Model Performance

| Model Version | MAE (Absolute) | MAE (Relative) | Improvement |
|--------------|----------------|----------------|-------------|
| Baseline (Default RF) | ~$1,380 | 14.84% | - |
| Tuned RF (v1) | ~$1,327 | 14.26% | +3.9% |
| **Final Model** | **$1,026** | **10.95%** | **+26.2%** |

### Key Findings

**Top 5 Most Important Features:**
1. **Smoker** (58.77%) - Smoking status is by far the strongest predictor
2. **BMI × Age Interaction** (12.37%) - Combined effect of age and BMI
3. **BMI** (11.56%) - Body mass index
4. **Age** (8.47%) - Patient age
5. **Children** (2.53%) - Number of dependents

**Insights:**
- Smoking status dominates medical cost predictions, accounting for nearly 60% of feature importance
- The interaction between BMI and age reveals that older individuals with higher BMI incur significantly higher costs
- Geographic region has minimal impact on costs (<1% each)
- Gender shows very low predictive power in this dataset

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization
- **Scikit-learn** - Machine learning
  - RandomForestRegressor
  - RandomizedSearchCV
  - ColumnTransformer
  - OneHotEncoder
  - train_test_split

## 📁 Project Structure

```
medical_insurance/
├── data/
│   └── medical_insurance.csv
├── medical_insurance.ipynb
└── README.md
```

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical_insurance.git
cd medical_insurance
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Run the notebook:
```bash
jupyter notebook medical_insurance.ipynb
```

## 🔮 Future Improvements

- **Try Gradient Boosting:** Test XGBoost or LightGBM for potentially better performance
- **Advanced Feature Engineering:** 
  - Create age groups/categories
  - BMI categories (underweight, normal, overweight, obese)
  - Interaction: smoker × bmi
- **Handle Outliers:** Investigate and handle extreme charge values
- **Cross-Validation:** Implement more robust validation strategies
- **Model Interpretability:** Add SHAP values for deeper insights

## 📝 Key Learnings

1. **Feature Importance Analysis:** Identifying and removing irrelevant features (like `senior`) improves model efficiency
2. **Interaction Terms:** Creating interaction features (bmi×age) can capture complex relationships
3. **Hyperparameter Tuning:** Systematic tuning improved MAE by nearly 4%
4. **Domain Knowledge:** Medical cost prediction is heavily influenced by lifestyle factors (smoking) and health metrics (BMI)

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created by Lahsaini Yazid

---

*Note: This project is for educational purposes. The model predictions should not be used for actual insurance pricing without proper validation and regulatory compliance.*
