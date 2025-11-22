import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: rgba(0,0,0,0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stSidebar"] > div {
        background: rgba(0,0,0,0.1) !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #FCFCFC !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
/* Main Background */
.stApp {
    background-image: url('https://img.freepik.com/free-vector/gradient-style-abstract-wireframe-background_23-2148993321.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main .block-container {
    background: rgba(0,0,0,0.3);
    backdrop-filter: blur(5px);
    border-radius: 20px;
    padding: 1.5rem;
    margin-top: 0rem;
}
/* Main Dashboard Styling */
.main-header {
    background:transparent;
    font-size: 3rem;
    color: #ecf0f1 !important;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 800;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    border-radius: 15px;
    padding: 0 5rem 0rem 6rem;
}

/* Enhanced Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    background-image: radial-gradient( circle farthest-corner at -5.6% -6.8%,  rgba(103,49,145,1) 37.3%, rgba(50,0,129,1) 73.5% );
    background:transparent;
    padding: 1.5rem;
    border-radius: 15px;
    color: #ffffff;
    text-align: center;
    box-shadow: 0 8px 32px rgba(44, 62, 80, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.metric-card:hover::before {
    left: 100%;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.metric-label {
    font-size: 0.9rem;
    color: #ecf0f1;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Dashboard Container */
.dashboard-container {
    background: rgba(255,255,255,0.05);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.1);
    padding: 0.5rem;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
    display: flex;
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.2);
    color: #FCFCFC;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    background-image: radial-gradient( circle farthest-corner at -5.6% -6.8%,  rgba(103,49,145,1) 37.3%, rgba(50,0,129,1) 73.5% );
    color: #FCFCFC;
    box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
    border: 1px solid #2c3e50;
}

/* Chart Containers */
.chart-container {
    background: rgba(0,0,0,0.1);
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
}

.chart-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #FCFCFC;
    margin-bottom: 1rem;
    text-align: center;
}

/* Analysis Cards */
.analysis-card {
    background: rgba(0,0,0,0.1);
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    border-left: 4px solid #3498db;
    margin: 1rem 0;
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
}

.analysis-card:hover {
    transform: translateX(5px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.12);
}

/* Prediction Section */
.prediction-container {
   // background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    //background-image: radial-gradient( circle farthest-corner at -5.6% -6.8%,  rgba(103,49,145,1) 37.3%, rgba(50,0,129,1) 73.5% );
    padding: 2rem;
    border-radius: 20px;
    color: #ffffff;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(44, 62, 80, 0.3);
}

.prediction-form {
    background: rgba(255,255,255,0.95);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.3);
    color: #2c3e50;
}

.prediction-result {
    background: rgba(76, 175, 80, 0.2);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #4caf50;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

/* Data Table Styling */
.dataframe {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

/* Button Enhancements */
.stButton > button {
    background: linear-gradient(45deg, #2c3e50, #34495e);
    color: #ffffff;
    border: none;
    border-radius: 25px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(44, 62, 80, 0.4);
    background: linear-gradient(45deg, #34495e, #2c3e50);
}

/* Success/Info Messages */
.stSuccess {
    background: linear-gradient(135deg, #4caf50, #45a049);
    border-radius: 10px;
    border: none;
    color: white;
}

.stInfo {
    background: linear-gradient(135deg, #2196f3, #1976d2);
    border-radius: 10px;
    border: none;
    color: white;
}

/* Loading Animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
        text-align: center !important;
    }
    
    .metric-card {
        padding: 1rem;
    }
    
    .dashboard-container {
        padding: 1rem;
    }
}

/* Sidebar Styling */
.stSidebar > div:first-child {
    background: rgba(44, 62, 80, 0.2) !important;
    backdrop-filter: blur(15px);
}

.css-1d391kg {
    background: rgba(44, 62, 80, 0.2) !important;
    backdrop-filter: blur(15px);
}

.css-1cypcdb {
    background: rgba(44, 62, 80, 0.2) !important;
    backdrop-filter: blur(15px);
}

.css-17eq0hr {
    background: rgba(44, 62, 80, 0.2) !important;
    backdrop-filter: blur(15px);
}

.sidebar .sidebar-content {
    background: rgba(44, 62, 80, 0.2) !important;
    color: #ffffff;
}

/* Hide Streamlit UI */
.css-18e3th9 {
    padding-top: 0 !important;
}

.css-1d391kg .css-1v0mbdj {
    display: none;
}

header[data-testid="stHeader"] {
    background: rgba(0,0,0,0) !important;
    height: 0 !important;
}

.css-z5fcl4 {
    padding-top: 1rem !important;
}

.css-1y4p8pa {
    padding: 0 !important;
}

/* Sidebar Header */
.sidebar-header {
    background: rgba(255,255,255,0.05);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    text-align: center;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
}

.sidebar-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.sidebar-subtitle {
    font-size: 0.9rem;
    color: #bdc3c7;
    margin: 0.5rem 0 0 0;
    font-weight: 400;
}

/* Section Dividers */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #ffffff40, transparent);
    margin: 1.5rem 0;
    border-radius: 1px;
}

/* Control Groups */
.control-group {
    background: rgba(255,255,255,0.08);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border-left: 3px solid #4fc3f7;
}

.control-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #ecf0f1;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Quick Stats */
.quick-stat {
    background: rgba(255,255,255,0.1);
    padding: 0.8rem;
    border-radius: 6px;
    margin: 0.5rem 0;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}

.stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.stat-label {
    font-size: 0.75rem;
    color: #bdc3c7;
    margin-top: 0.2rem;
    font-weight: 400;
}

/* Action Buttons */
.action-btn {
    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
    color: white;
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 20px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin: 0.3rem 0;
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(238, 90, 36, 0.4);
}

/* Info Cards */
.info-card {
    background: rgba(76, 175, 80, 0.1);
    border-left: 4px solid #4caf50;
    padding: 0.8rem;
    border-radius: 4px;
    margin: 0.5rem 0;
}

.info-text {
    font-size: 0.8rem;
    color: #a5d6a7;
    margin: 0;
    font-weight: 400;
}

/* Slider Customization */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #4fc3f7, #29b6f6);
}

/* Selectbox Styling */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px;
}

/* Multiselect Styling */
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# Enhanced Title with Animation
st.markdown("""
<div style="text-align: center; padding: 2rem 0; width: 100%; display: flex; flex-direction: column; align-items: center;">
    <h1 class="main-header" style="text-align: center; width: 100%;">🎯 Customer Segmentation Dashboard</h1>
    <p style="font-size: 1.2rem; color: #bdc3c7; margin-top: -1rem; text-align: center;">Advanced Analytics & Machine Learning Insights</p>
    <div style="width: 100px; height: 3px; background: linear-gradient(90deg, #1e3c72, #2a5298); margin: 1rem auto; border-radius: 2px;"></div>
</div>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('customer_segmentation_data.csv')
    return df

@st.cache_data
def prepare_data(df):
    # Clean data
    df_clean = df.drop_duplicates()
    
    # Remove outliers
    numerical_cols = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    df_clean['gender_encoded'] = le_gender.fit_transform(df_clean['gender'])
    df_clean['category_encoded'] = le_category.fit_transform(df_clean['preferred_category'])
    
    return df_clean, le_gender, le_category

@st.cache_data
def perform_clustering(df_clean, n_clusters=4):
    features = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 
               'last_purchase_amount', 'gender_encoded', 'category_encoded']
    
    X = df_clean[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return clusters, X_pca, kmeans, scaler, features

# Load data
df = load_data()
df_clean, le_gender, le_category = prepare_data(df)

# Enhanced Sidebar
with st.sidebar:
    # Header Section
    st.markdown("""
    <div class="sidebar-header">
        <h2 class="sidebar-title">🎛️ Control Panel</h2>
        <p class="sidebar-subtitle">Configure your analysis parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering Configuration
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown('<p class="control-label">🎯 Clustering Settings</p>', unsafe_allow_html=True)
    n_clusters = st.slider(
        "Number of Clusters", 
        min_value=2, max_value=8, value=4,
        help="Adjust the number of customer segments"
    )
    
    algorithm = st.selectbox(
        "Algorithm",
        ["K-Means", "Hierarchical", "DBSCAN"],
        help="Choose clustering algorithm"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Feature Selection
    st.markdown('<div class="control-group">', unsafe_allow_html=True)
    st.markdown('<p class="control-label">📊 Feature Selection</p>', unsafe_allow_html=True)
    
    all_features = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']
    selected_features = st.multiselect(
        "Analysis Features",
        all_features,
        default=['age', 'income', 'spending_score'],
        help="Select features for detailed analysis"
    )
    
    # Quick feature toggles
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Financial", help="Select income & spending"):
            selected_features = ['income', 'spending_score', 'last_purchase_amount']
            st.experimental_rerun()
    with col2:
        if st.button("👤 Behavioral", help="Select behavior features"):
            selected_features = ['purchase_frequency', 'membership_years', 'spending_score']
            st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown('<p class="control-label">📋 Dataset Overview</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="quick-stat">
            <div class="stat-value">1000</div>
            <div class="stat-label">Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="quick-stat">
            <div class="stat-value">9</div>
            <div class="stat-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Info Cards
    st.markdown("""
    <div class="info-card">
        <p class="info-text">💡 <strong>Tip:</strong> Higher cluster numbers provide more granular segments but may reduce interpretability.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown('<p class="control-label">⚡ Quick Actions</p>', unsafe_allow_html=True)
    
    if st.button("🔄 Reset to Default", help="Reset all parameters"):
        st.experimental_rerun()
    
    if st.button("📊 Auto-Optimize", help="Find optimal parameters"):
        st.info("🔍 Finding optimal clusters...")
    
    # Export Options
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="control-label">💾 Export Options</p>', unsafe_allow_html=True)
    
    export_format = st.radio(
        "Format",
        ["CSV", "Excel", "JSON"],
        horizontal=True
    )

# Perform clustering
clusters, X_pca, kmeans, scaler, features = perform_clustering(df_clean, n_clusters)
df_clean['cluster'] = clusters

# Enhanced KPI Dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👥</div>
        <div class="metric-value">{:,}</div>
        <div class="metric-label">Total Customers</div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">Active Database</div>
    </div>
    """.format(len(df_clean)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div class="metric-value">{}</div>
        <div class="metric-label">Segments</div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">ML Clusters</div>
    </div>
    """.format(n_clusters), unsafe_allow_html=True)

with col3:
    avg_income = df_clean['income'].mean()
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💰</div>
        <div class="metric-value">${:,.0f}</div>
        <div class="metric-label">Avg Income</div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">Annual Revenue</div>
    </div>
    """.format(avg_income), unsafe_allow_html=True)

with col4:
    avg_spending = df_clean['spending_score'].mean()
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
        <div class="metric-value">{:.1f}</div>
        <div class="metric-label">Spending Score</div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">Engagement Level</div>
    </div>
    """.format(avg_spending), unsafe_allow_html=True)

# Enhanced Tabs with Icons and Descriptions
st.markdown('<br>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visualization", 
    "📈 Analytics", 
    "🎯 Prediction", 
    "📋 Dataset"
])

with tab1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 0;">
        <h2 style="color: #ecf0f1; margin-bottom: 0.5rem;">🎨 Interactive Cluster Visualization</h2>
        <p style="color: #bdc3c7; margin-bottom: 0;">Explore customer segments through advanced visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="chart-title">🔍 PCA Cluster Map</h3>', unsafe_allow_html=True)
        
        # Enhanced PCA visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                           cmap='viridis', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('First Principal Component', fontsize=12, fontweight='bold', color='#FCFCFC')
        ax.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold', color='#FCFCFC')
        ax.tick_params(colors='#FCFCFC')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor((0, 0, 0, 0.05))
        
        # Enhanced colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12, fontweight='bold', color='#FCFCFC')
        cbar.ax.tick_params(colors='#FCFCFC')
        
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="chart-title">📈 Feature Relationship</h3>', unsafe_allow_html=True)
        
        if len(selected_features) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('white')
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            
            for i, cluster in enumerate(range(n_clusters)):
                cluster_data = df_clean[df_clean['cluster'] == cluster]
                ax.scatter(cluster_data[selected_features[0]], cluster_data[selected_features[1]], 
                          label=f'Segment {cluster}', alpha=0.7, s=60, 
                          color=colors[i % len(colors)], edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel(selected_features[0].replace('_', ' ').title(), fontsize=12, fontweight='bold', color='#FCFCFC')
            ax.set_ylabel(selected_features[1].replace('_', ' ').title(), fontsize=12, fontweight='bold', color='#FCFCFC')
            ax.tick_params(colors='#FCFCFC')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor((0, 0, 0, 0.05))
            ax.legend(frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
        else:
            st.info("🔧 Select at least 2 features from the sidebar to view relationships")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional visualization row
    st.markdown('<br>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    
    with col3:
        #st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="chart-title">📊 Cluster Distribution</h3>', unsafe_allow_html=True)
        
        cluster_sizes = df_clean['cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(range(len(cluster_sizes)), cluster_sizes.values, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(cluster_sizes)],
                     alpha=0.8, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Customer Segment', fontweight='bold', color='#FCFCFC')
        ax.set_ylabel('Number of Customers', fontweight='bold', color='#FCFCFC')
        ax.tick_params(colors='#FCFCFC')
        ax.set_xticks(range(len(cluster_sizes)))
        ax.set_xticklabels([f'Segment {i}' for i in range(len(cluster_sizes))])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold', color='#FCFCFC')
        
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
       # st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #ecf0f1; margin-bottom: 1rem;">🎯 Insights Summary</h3>', unsafe_allow_html=True)
        
        # Dynamic insights based on clusters
        total_customers = len(df_clean)
        largest_cluster = cluster_sizes.idxmax()
        largest_cluster_pct = (cluster_sizes.max() / total_customers) * 100
        
        st.markdown(f"""
        <div style="background:rgba(0, 0, 0, 0.05); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong>🏆 Dominant Segment:</strong> Segment {largest_cluster}<br>
            <span style="color: #27ae60;">Contains {largest_cluster_pct:.1f}% of customers</span>
        </div>
        
        <div style="background:rgba(0, 0, 0, 0.05); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong>📈 Segmentation Quality:</strong><br>
            <span style="color: #1976d2;">Well-distributed clusters detected</span>
        </div>
        
        <div style="background:rgba(0, 0, 0, 0.05); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <strong>💡 Recommendation:</strong><br>
            <span style="color: #f57c00;">Focus marketing on largest segments</span>
        </div>
        """,unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("<h2 style='text-align: center;'>Cluster Analysis</h2>", unsafe_allow_html=True)
    
    # Cluster statistics
    cluster_stats = df_clean.groupby('cluster').agg({
        'age': 'mean',
        'income': 'mean',
        'spending_score': 'mean',
        'membership_years': 'mean',
        'purchase_frequency': 'mean',
        'last_purchase_amount': 'mean'
    }).round(2)
    
    st.subheader("Cluster Statistics")
    st.dataframe(cluster_stats)
    
    # Cluster sizes
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_sizes = df_clean['cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        cluster_sizes.plot(kind='bar', ax=ax)
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Customers')
        st.pyplot(fig)
    
    with col2:
        # Gender distribution
        gender_dist = pd.crosstab(df_clean['cluster'], df_clean['gender'])
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_dist.plot(kind='bar', ax=ax)
        ax.set_title('Gender Distribution by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.legend(title='Gender')
        st.pyplot(fig)

with tab3:
    st.markdown("""
    <div class="prediction-container">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: white; margin-bottom: 0.5rem;">AI-Powered Customer Prediction</h2>
            <p style="color: rgba(255,255,255,0.9);">Enter customer details to predict their segment using machine learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
   # st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
    
    # Enhanced input form with better organization
    st.markdown('<h3 style="color: #2c3e50; text-align: center; margin-bottom: 1.5rem;">📝 Customer Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 style="color: #34495e;">👤 Demographics</h4>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 70, 35, help="Customer's age in years")
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], help="Customer's gender")
        
        st.markdown('<h4 style="color: #34495e; margin-top: 1.5rem;">💰 Financial Profile</h4>', unsafe_allow_html=True)
        income = st.slider("Annual Income ($)", 30000, 150000, 75000, step=5000, help="Customer's annual income")
        spending_score = st.slider("Spending Score", 1, 100, 50, help="Customer's spending behavior score")
    
    with col2:
        st.markdown('<h4 style="color: #34495e;">🛍️ Shopping Behavior</h4>', unsafe_allow_html=True)
        purchase_frequency = st.slider("Monthly Purchases", 1, 50, 25, help="Number of purchases per month")
        last_purchase = st.slider("Last Purchase ($)", 10, 1000, 500, step=10, help="Amount of last purchase")
        
        st.markdown('<h4 style="color: #34495e; margin-top: 1.5rem;">🏪 Preferences</h4>', unsafe_allow_html=True)
        membership_years = st.slider("Membership Years", 1, 10, 5, help="Years as a customer")
        category = st.selectbox("Preferred Category", 
                               ['Groceries', 'Sports', 'Clothing', 'Electronics', 'Home & Garden'],
                               help="Most frequently purchased category")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced prediction button
    st.markdown('<div style="display: flex; justify-content: center; margin: 1rem 0;">', unsafe_allow_html=True)
    if st.button("🚀 Predict Customer Segment", help="Click to analyze customer profile"):
            with st.spinner('🔮 Analyzing customer profile...'):
                # Prepare input data
                gender_encoded = le_gender.transform([gender])[0]
                category_encoded = le_category.transform([category])[0]
                
                input_data = np.array([[age, income, spending_score, membership_years, 
                                       purchase_frequency, last_purchase, gender_encoded, category_encoded]])
                
                # Scale input
                input_scaled = scaler.transform(input_data)
                
                # Predict cluster
                predicted_cluster = kmeans.predict(input_scaled)[0]
                
                # Enhanced results display
                st.markdown("""
                <div class="prediction-result">
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h2 style="color: #27ae60; margin: 0;">🎯 Prediction Result</h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show prediction with enhanced styling
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #27ae60, #2ecc71);
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 8px 25px rgba(39, 174, 96, 0.3);
                        margin: 1rem 0;
                    ">
                        <h1 style="margin: 0; font-size: 3rem;">#{predicted_cluster}</h1>
                        <h3 style="margin: 0.5rem 0;">Customer Segment</h3>
                        <p style="margin: 0; opacity: 0.9;">AI Confidence: High</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show detailed cluster characteristics
                cluster_data = df_clean[df_clean['cluster'] == predicted_cluster]
                
                st.markdown('<h3 style="color: #ecf0f1; text-align: center; margin: 2rem 0 1rem 0;">📊 Segment Characteristics</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
               
                
                # Add space between rows
                st.markdown('<br>', unsafe_allow_html=True)
                
                # Second row - Score and Category
                col3, col4 = st.columns(2)
                
                with col1:
                    avg_age = cluster_data['age'].mean()
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.3); padding: 2rem 2rem; border-radius: 10px; text-align: center;">
                        <h4 style="color: #27ae60; margin: 0;">👤 Age</h4>
                        <h2 style="color: #2c3e50; margin: 0.5rem 0;">{avg_age:.1f}</h2>
                        <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">Average Years</p>
                    </div>
                    """,unsafe_allow_html=True)
                
                with col2:
                    avg_income = cluster_data['income'].mean()
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.3); padding: 2rem 2rem; border-radius: 10px; text-align: center;">
                        <h4 style="color: #1976d2; margin: 0;">💰 Income</h4>
                        <h2 style="color: #2c3e50; margin: 0.5rem 0;">${avg_income:,.0f}</h2>
                        <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">Annual Average</p>
                    </div>
                    """,unsafe_allow_html=True)
                
                with col3:
                    avg_spending = cluster_data['spending_score'].mean()
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.3); padding: 2rem 2rem; border-radius: 10px; text-align: center;">
                        <h4 style="color: #f57c00; margin: 0;">📊 Score</h4>
                        <h2 style="color: #2c3e50; margin: 0.5rem 0;">{avg_spending:.1f}</h2>
                        <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">Spending Level</p>
                    </div>
                    """,unsafe_allow_html=True  )
                
                with col4:
                    common_category = cluster_data['preferred_category'].mode()[0]
                    st.markdown(f"""
                    <div style="background: rgba(0,0,0,0.3); padding: 2rem 2rem; border-radius: 10px; text-align: center;">
                        <h4 style="color: #c2185b; margin: 0;">🛍️ Category</h4>
                        <h2 style="color: #2c3e50; margin: 0.5rem 0; font-size:1.9rem">{common_category}</h2>
                        <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">Most Popular</p>
                    </div>
                    """,unsafe_allow_html=True)

with tab4:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df_clean.shape}")
        st.write(f"**Features:** {len(features)}")
        st.write("**Data Types:**")
        st.write(df_clean.dtypes)
    
    with col2:
        st.subheader("Summary Statistics")
        st.write(df_clean.describe())
    
    st.subheader("Sample Data")
    st.dataframe(df_clean.head(10))
    
    # Download results
    if st.button("Download Cluster Results"):
        csv = df_clean.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )

# Enhanced Footer
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown("""
<div style="
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-top: 3rem;
">
    <p style="margin: 0; opacity: 0.9;">Powered by Advanced Machine Learning & Streamlit</p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 1rem; opacity: 0.8;">📊 Analytics</span>
        <span style="margin: 0 1rem; opacity: 0.8;">🤖 AI/ML</span>
        <span style="margin: 0 1rem; opacity: 0.8;">📈 Insights</span>
    </div>
</div>
""", unsafe_allow_html=True)