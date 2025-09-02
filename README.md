💳 Credit Card Fraud Detection System

This project implements a fraud detection system for credit card transactions using machine learning. Multiple models were explored including Logistic Regression, Random Forest, XGBoost, and LightGBM, and XGBoost was selected as the best-performing model.

Key Steps for XGBoost

1) Hyperparameter Tuning:

--> Tuned n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, and scale_pos_weight using RandomizedSearchCV.

--> Optimized for performance while reducing bias and overfitting.

2) Threshold Optimization:
--> Determined the optimal probability threshold using precision-recall curve to balance fraud detection performance.

3) Data Preprocessing:

--> Scaled Amount and Time features.

--> Handled class imbalance using scale_pos_weight.

4) Model Evaluation:

--> Evaluated using precision, recall, F1-score, ROC-AUC, and confusion matrix.

5) Deployment:

--> Built a Streamlit web app allowing users to input transactions manually or via CSV upload. Displays predictions, fraud probabilities, probability distribution plots, and evaluation metrics. Includes a download option for predictions.