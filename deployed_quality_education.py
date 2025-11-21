import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ----------------------------------------------------------
# Load Lottie Animations (Safe loader)
# ----------------------------------------------------------
def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Verified working animations
lottie_cluster = load_lottie("https://assets4.lottiefiles.com/packages/lf20_touohxv0.json")
lottie_upload = load_lottie("https://assets9.lottiefiles.com/packages/lf20_j1adxtyb.json")

# ----------------------------------------------------------
# Streamlit Page Settings
# ----------------------------------------------------------
st.set_page_config(
    page_title="India SDG Clustering",
    layout="wide",
    page_icon="📊"
)

# ----------------------------------------------------------
# Header Section
# ----------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🇮🇳 India SDG Index – Interactive Clustering Dashboard")
    st.markdown("""
    ### 🔍 Explore SDG Education Indicators Across Indian States  
    Upload the dataset, choose number of clusters, and visualize insights interactively.
    """)
with col2:
    if lottie_cluster:
        st_lottie(lottie_cluster, height=140)
    else:
        st.write("🎨 Animation unavailable")

# ----------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

theme = st.sidebar.selectbox("Theme Mode", ["Light", "Dark"])
k_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
pca_toggle = st.sidebar.checkbox("Show PCA Visualization", True)

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ using Streamlit")

# Apply theme mode
if theme == "Dark":
    sns.set_theme(style="darkgrid")
else:
    sns.set_theme(style="whitegrid")

# ----------------------------------------------------------
# File Upload Section
# ----------------------------------------------------------
st.subheader("📁 Upload Your Dataset")

if lottie_upload:
    st_lottie(lottie_upload, height=120)
else:
    st.info("Upload your CSV file below:")

uploaded_file = st.file_uploader("Upload India_SDG_Index_Indicator_List_2021 CSV", type=["csv"])

# ----------------------------------------------------------
# Main App Logic (Runs After CSV Upload)
# ----------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")
    st.write("### 🔎 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ----------------------------------------------------------
    # Column Cleanup + Renaming
    # ----------------------------------------------------------
    rename_cols = {
        'Gross Enrollment Ratio in Higher Education (18-23 years)': 'GER_HigherEd',
        'Literacy Rate of Youth (15-24 years)': 'Youth_Literacy',
        'Adjusted Net Enrolment Rate (ANER) in elementary education (class 1-8) (%)': 'ANER',
        'Average annual dropout rate at secondary level (class 9-10)': 'Avg_dropout_rate',
        'Gross Enrolment Ratio (GER) in higher secondary (class 11-12) (%)': 'GER_11_12',
        'Percentage of students in grade VIII achieving at least a minimum proficiency level in terms of nationally defined learning outcomes to be attained by the pupils at the end of the grade': 'End_of_grade',
        'Gross Enrolment Ratio (GER) in higher education (18-23 years)': 'GER_18_23',
        'Percentage of persons with disability (15 years and above) who have completed at least secondary education': 'Disability_SecondaryEd',
        'Gender Parity Index (GPI) for higher education (18-23 years)': 'GPI_18_23',
        'Percentage of persons 15 years and above who are literate': 'Literacy_15plus',
        'Percentage of schools with access to basic infrastructure (electricity and drinking water-both)': 'Basic_Infrastructure',
        'Percentage of schools with computers': 'Schools_with_Computers',
        'Percentage of trained teachers at secondary level (class 9-10)': 'Trained_Teachers_9_10',
        'Pupil Teacher Ratio (PTR) at secondary level (class 9-10)': 'PTR_9_10'
    }
    df.rename(columns=rename_cols, inplace=True)

    # Fill missing data
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    areas = df["Area"]
    X = df.drop(columns=["Area"])

    # ----------------------------------------------------------
    # Scaling
    # ----------------------------------------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------------------------------------
    # KMeans Clustering
    # ----------------------------------------------------------
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # ----------------------------------------------------------
    # Clustered Data Table
    # ----------------------------------------------------------
    st.subheader("📊 Cluster Assignment")
    st.dataframe(df[["Area", "Cluster"]].sort_values("Cluster"), use_container_width=True)

    # ----------------------------------------------------------
    # Count Plot
    # ----------------------------------------------------------
    st.subheader("📈 States per Cluster")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(x=df["Cluster"], palette="viridis", ax=ax1)
    st.pyplot(fig1)

    # ----------------------------------------------------------
    # Strip Plot
    # ----------------------------------------------------------
    st.subheader("📍 State Distribution by Cluster")
    fig2, ax2 = plt.subplots(figsize=(7, 14))
    sns.stripplot(x=df["Cluster"], y=df["Area"], palette="viridis", ax=ax2)
    st.pyplot(fig2)

    # ----------------------------------------------------------
    # Heatmap
    # ----------------------------------------------------------
    st.subheader("🔥 Cluster Feature Heatmap")
    cluster_summary = df.groupby("Cluster").mean(numeric_only=True)

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    sns.heatmap(cluster_summary.round(2), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # ----------------------------------------------------------
    # PCA Visualization (Optional)
    # ----------------------------------------------------------
    if pca_toggle:
        st.subheader("🧭 PCA Visualization (2D Projection)")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        scatter = ax4.scatter(
            pca_data[:, 0], pca_data[:, 1],
            c=df["Cluster"], cmap="viridis",
            s=90, edgecolors="black"
        )
        ax4.set_xlabel("PC 1")
        ax4.set_ylabel("PC 2")
        plt.legend(*scatter.legend_elements(), title="Cluster")
        st.pyplot(fig4)

    # ----------------------------------------------------------
    # States Per Cluster
    # ----------------------------------------------------------
    st.subheader("📝 States in Each Cluster")
    for cluster in sorted(df["Cluster"].unique()):
        states = df[df["Cluster"] == cluster]["Area"].tolist()
        st.write(f"### Cluster {cluster}")
        st.success(", ".join(states))

else:
    st.info("☝️ Upload the dataset to begin.")
