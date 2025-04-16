EARLY FLOOD PREDICTION IN UNCERTAIN WEATHER FORECASTS

🌀 PROJECT OVERVIEW  
This project focuses on building an early flood prediction system using machine learning algorithms including K-Nearest Neighbors (KNN), Decision Tree Classifier (DTC), Logistic Regression (LR), Gradient Boosting Model (GBM), and XGBoost. The primary goal is to predict flood occurrence based on monthly weather data and evaluate model performance through statistical and graphical analyses.

🌀 FEATURES  
- Comparison of KNN with DTC, GBM, LR, and XGBoost  
- KNN achieved the highest accuracy of 98 percent  
- SPSS statistical analysis used to validate model performance  
- Visualization using confusion matrix and ROC curve  
 
🌀 TECH STACK  
Python  
scikit-learn  
XGBoost  
NumPy and Pandas  
Matplotlib and Seaborn  
SPSS (for external statistical evaluation)

🌀 DATASET  
Monthly weather data from January to December was used with a target variable named flood (0 for no flood, 1 for flood).  
The dataset was cleaned and normalized using StandardScaler before feeding it to the models.

🌀 SETUP AND INSTALLATION  

1. Clone the repository  
git clone your_repository_link_here  
cd your_project_directory

2. Install required Python libraries  
pip install pandas numpy scikit-learn matplotlib seaborn xgboost

3. Run any of the scripts depending on the comparison  
python knn_vs_dtc.py  
python knn_vs_gbm.py  
python knn_vs_lr.py  
python knn_vs_xgboost.py  
Make sure the dataset file named dataset.csv is in the same directory as the scripts.

🌀 MODEL INSIGHTS  
All models follow these common steps:  
- Data normalization using StandardScaler  
- Training and testing split (80:20)  
- Model evaluation using accuracy score, classification report, confusion matrix, and ROC curve  

🌀 RESULTS  
KNN demonstrated a superior performance with an accuracy of 98 percent.  
Across all comparisons, KNN outperformed other models in both classification metrics and ROC-AUC score.  
The SPSS comparison graphs are used to visually verify KNN’s accuracy advantage.

🌀 SPSS STATISTICAL ANALYSIS  
Graphs created using SPSS Analytics 26 showed the mean accuracy difference with error bars between KNN and other models. These visual representations further supported the finding that KNN performed best among all models considered.

🌀 FUTURE ENHANCEMENTS  
- Include more weather-related features like rainfall and temperature trends  
- Integrate the prediction system with real-time flood alerts  
- Explore deep learning approaches like LSTM for improved performance

🌀 CONTRIBUTING  
Contributions are welcome. Fork the repository, make your changes, and create a pull request.

🌀 LICENSE  
This project is under the MIT License. You are free to use, share, and improve the code.

