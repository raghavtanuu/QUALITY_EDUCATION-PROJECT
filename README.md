# Quality Education Project – India SDG Index

## 📌 Overview
This project analyzes education-related Sustainable Development Goal (SDG) indicators across Indian states. It provides:

1. **Unsupervised Clustering** – Groups states based on education indicators using **KMeans**.
2. **Supervised Classification** – Predicts cluster membership using **RandomForest, SVC, and Logistic Regression**.
3. **Interactive Dashboard** – Visualizes data and insights via **Streamlit**.

---

## 🛠️ Features
[India_SDG_Index_Indicator_List_2021.csv](https://github.com/user-attachments/files/23682227/India_SDG_Index_Indicator_List_2021.csv)

- Upload your dataset ('India_SDG_Index_Indicator_List_2021(Modified).csv') in the app.
- Explore clusters:
  - Cluster assignments for all states
  - Count plots and strip plots of clusters
  - Heatmap of average feature values per cluster
  - Optional PCA 2D visualization
- Supervised model predictions:
  - Accuracy and classification reports for RandomForest, SVC, and Logistic Regression
- Interactive and visually appealing dashboard using Streamlit and Lottie animations.



## 🚀 Live Demo

View the interactive Streamlit app here:  
[Live Dashboard](https://share.streamlit.io/raghavtanuu/QUALITY_EDUCATION-PROJECT/main/deployed_quality_education_project.py)




## 📂 Project Structure

Quality_Education_Deploy/
├── deployed_quality_education_project.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── India_SDG_Index_Indicator_List_2021.csv # Sample dataset
Run the Streamlit app locally:

streamlit run deployed_quality_education_project.py


Open your browser at http://localhost:8501 to interact with the dashboard.
