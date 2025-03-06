# Gender Recognition using Logistic Regression

## Project Overview
This project aims to classify voice recordings based on gender using machine learning techniques. The dataset contains various acoustic properties of voice samples and their corresponding labels (male or female). The classification model is built using Logistic Regression and is optimized using feature selection and hyperparameter tuning.

## Dataset
- **Source:** The dataset is loaded from `voice.csv`.
- **Features:** 20 numerical acoustic properties such as mean frequency, median, standard deviation, skewness, entropy, mode, and spectral centroid.
- **Target Variable:** `label` (Male or Female), which is encoded for binary classification.
- **Data Preprocessing:**
  - Missing values imputed using mean and constant strategies.
  - Standardization applied to numerical features using `StandardScaler`.
  - Label encoding applied to categorical labels using `LabelEncoder`.
  - Feature selection using `SelectKBest` with ANOVA F-statistics (`f_classif`).

## Implementation Steps
1. **Data Loading and Exploration:**
   - Read dataset using `pandas`.
   - Explore statistical summaries and correlation matrices.
   - Visualize data distributions using `matplotlib` and `seaborn`.
2. **Preprocessing Pipeline:**
   - Construct a `Pipeline` for numeric and categorical feature transformations.
   - Apply feature scaling and imputation.
   - Select the top 4 most significant features using `SelectKBest`.
3. **Model Training and Evaluation:**
   - Split dataset into training and test sets (70-30 split).
   - Train Logistic Regression using `GridSearchCV` for hyperparameter tuning.
   - Evaluate using classification metrics (accuracy, precision, recall, F1-score).
   - Generate a confusion matrix for performance analysis.
4. **Cross-Validation:**
   - Perform k-fold cross-validation (k=5) to validate model robustness.
   - Compute mean accuracy and standard deviation.
5. **Visualization and Interpretation:**
   - Confusion matrix visualization.
   - Regression plot between `meanfreq` and `label`.
   - Calculation of classification accuracy, error rate, true positive rate, false positive rate, and specificity.

## Performance Metrics
- **Confusion Matrix Analysis:**
  - TP (True Positives)
  - TN (True Negatives)
  - FP (False Positives)
  - FN (False Negatives)
- **Key Metrics Computed:**
  - Classification Accuracy
  - Classification Error Rate
  - Sensitivity (Recall / True Positive Rate)
  - Specificity (True Negative Rate)
  - False Positive Rate (FPR)

## Results
- Achieved an accuracy score of 97.3%.
- Cross-validation accuracy mean and standard deviation are computed.
- Performance insights derived from confusion matrix analysis.

## Future Improvements
- Experiment with other classifiers such as SVM, Random Forest, or Deep Learning models.
- Perform hyperparameter tuning using Bayesian Optimization.
- Implement PCA or LDA for dimensionality reduction.
- Incorporate additional acoustic features for improved classification.

## Dependencies
Ensure you have the following Python libraries installed:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
Install them using:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/Severus-193/Gender_Recognition.git
   cd Gender_Recognition
   ```
2. Execute the script:
   ```sh
   python gender.py
   ```
