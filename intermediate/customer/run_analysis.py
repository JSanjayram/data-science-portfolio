#!/usr/bin/env python3
"""
Customer Segmentation Analysis Runner
Executes the complete customer segmentation pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the customer data"""
    print("📊 Loading and cleaning data...")
    
    # Load data
    df = pd.read_csv('customer_segmentation_data.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Remove duplicates
    df_clean = df.drop_duplicates()
    print(f"After removing duplicates: {df_clean.shape}")
    
    # Remove outliers using IQR method
    numerical_cols = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']
    
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    print(f"After removing outliers: {df_clean.shape}")
    return df_clean

def prepare_features(df):
    """Prepare features for clustering"""
    print("🔧 Preparing features...")
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    
    df_features = df.copy()
    df_features['gender_encoded'] = le_gender.fit_transform(df_features['gender'])
    df_features['category_encoded'] = le_category.fit_transform(df_features['preferred_category'])
    
    # Select features for clustering
    features = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 
               'last_purchase_amount', 'gender_encoded', 'category_encoded']
    
    X = df_features[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Features prepared: {features}")
    return X_scaled, features, scaler, le_gender, le_category

def find_optimal_clusters(X_scaled):
    """Find optimal number of clusters using elbow method and silhouette score"""
    print("🎯 Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best silhouette score: {best_silhouette:.3f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Method
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette Scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_k, silhouette_scores

def apply_clustering_algorithms(X_scaled, optimal_k):
    """Apply different clustering algorithms"""
    print("🤖 Applying clustering algorithms...")
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Calculate silhouette scores
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
    hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
    
    if len(set(dbscan_labels)) > 1:
        dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    else:
        dbscan_silhouette = -1
    
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")
    print(f"Hierarchical Silhouette Score: {hierarchical_silhouette:.3f}")
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.3f}")
    print(f"DBSCAN found {len(set(dbscan_labels))} clusters (including noise)")
    
    return kmeans, kmeans_labels, hierarchical_labels, dbscan_labels

def visualize_clusters(X_scaled, kmeans_labels, hierarchical_labels, dbscan_labels, optimal_k):
    """Create cluster visualizations"""
    print("📈 Creating visualizations...")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data
    axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    axes[0,0].set_title('Original Data')
    axes[0,0].set_xlabel('PC1')
    axes[0,0].set_ylabel('PC2')
    
    # K-Means
    scatter = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    axes[0,1].set_title(f'K-Means Clustering (k={optimal_k})')
    axes[0,1].set_xlabel('PC1')
    axes[0,1].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[0,1])
    
    # Hierarchical
    scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
    axes[1,0].set_title('Hierarchical Clustering')
    axes[1,0].set_xlabel('PC1')
    axes[1,0].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[1,0])
    
    # DBSCAN
    scatter = axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
    axes[1,1].set_title('DBSCAN Clustering')
    axes[1,1].set_xlabel('PC1')
    axes[1,1].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca

def analyze_clusters(df_clean, kmeans_labels, optimal_k):
    """Analyze cluster characteristics"""
    print("🔍 Analyzing cluster characteristics...")
    
    # Add cluster labels to dataframe
    df_analysis = df_clean.copy()
    df_analysis['cluster'] = kmeans_labels
    
    # Cluster statistics
    cluster_stats = df_analysis.groupby('cluster').agg({
        'age': ['mean', 'std'],
        'income': ['mean', 'std'],
        'spending_score': ['mean', 'std'],
        'membership_years': ['mean', 'std'],
        'purchase_frequency': ['mean', 'std'],
        'last_purchase_amount': ['mean', 'std']
    }).round(2)
    
    print("\\nCluster Statistics:")
    print(cluster_stats)
    
    # Cluster sizes
    cluster_sizes = df_analysis['cluster'].value_counts().sort_index()
    print(f"\\nCluster Sizes:")
    print(cluster_sizes)
    
    # Create cluster profiles
    cluster_profiles = {}
    
    for cluster in range(optimal_k):
        cluster_data = df_analysis[df_analysis['cluster'] == cluster]
        
        profile = {
            'size': len(cluster_data),
            'avg_age': cluster_data['age'].mean(),
            'avg_income': cluster_data['income'].mean(),
            'avg_spending_score': cluster_data['spending_score'].mean(),
            'avg_membership_years': cluster_data['membership_years'].mean(),
            'avg_purchase_frequency': cluster_data['purchase_frequency'].mean(),
            'avg_last_purchase': cluster_data['last_purchase_amount'].mean(),
            'dominant_gender': cluster_data['gender'].mode()[0],
            'dominant_category': cluster_data['preferred_category'].mode()[0]
        }
        
        cluster_profiles[f'Cluster_{cluster}'] = profile
    
    # Save results
    profiles_df = pd.DataFrame(cluster_profiles).T
    profiles_df.to_csv('cluster_profiles.csv')
    df_analysis.to_csv('customer_segments.csv', index=False)
    
    print("\\nCluster Profiles:")
    print(profiles_df.round(2))
    
    return df_analysis, profiles_df

def create_business_insights(profiles_df, optimal_k):
    """Generate business insights and recommendations"""
    print("💡 Generating business insights...")
    
    insights = []
    
    for i in range(optimal_k):
        cluster_name = f'Cluster_{i}'
        profile = profiles_df.loc[cluster_name]
        
        # Classify cluster based on characteristics
        if profile['avg_income'] > 100000 and profile['avg_spending_score'] > 70:
            cluster_type = "High-Value Customers"
            recommendation = "Premium products, exclusive offers, loyalty programs"
        elif profile['avg_income'] < 60000 and profile['avg_spending_score'] < 40:
            cluster_type = "Budget Shoppers"
            recommendation = "Discount campaigns, value products, price promotions"
        elif profile['avg_purchase_frequency'] > 30:
            cluster_type = "Frequent Buyers"
            recommendation = "Bulk discounts, subscription services, convenience features"
        else:
            cluster_type = "Occasional Customers"
            recommendation = "Engagement campaigns, targeted promotions, retention programs"
        
        insight = {
            'cluster': cluster_name,
            'type': cluster_type,
            'size': int(profile['size']),
            'avg_income': f"${profile['avg_income']:,.0f}",
            'avg_spending': f"{profile['avg_spending_score']:.1f}",
            'recommendation': recommendation
        }
        
        insights.append(insight)
    
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv('business_insights.csv', index=False)
    
    print("\\n🎯 Business Insights:")
    for insight in insights:
        print(f"\\n{insight['cluster']} - {insight['type']}:")
        print(f"  Size: {insight['size']} customers")
        print(f"  Avg Income: {insight['avg_income']}")
        print(f"  Avg Spending Score: {insight['avg_spending']}")
        print(f"  Recommendation: {insight['recommendation']}")
    
    return insights_df

def main():
    """Main execution function"""
    print("🚀 Starting Customer Segmentation Analysis...")
    print("=" * 50)
    
    try:
        # Step 1: Load and clean data
        df_clean = load_and_clean_data()
        
        # Step 2: Prepare features
        X_scaled, features, scaler, le_gender, le_category = prepare_features(df_clean)
        
        # Step 3: Find optimal clusters
        optimal_k, silhouette_scores = find_optimal_clusters(X_scaled)
        
        # Step 4: Apply clustering algorithms
        kmeans, kmeans_labels, hierarchical_labels, dbscan_labels = apply_clustering_algorithms(X_scaled, optimal_k)
        
        # Step 5: Visualize clusters
        X_pca = visualize_clusters(X_scaled, kmeans_labels, hierarchical_labels, dbscan_labels, optimal_k)
        
        # Step 6: Analyze clusters
        df_analysis, profiles_df = analyze_clusters(df_clean, kmeans_labels, optimal_k)
        
        # Step 7: Generate business insights
        insights_df = create_business_insights(profiles_df, optimal_k)
        
        print("\\n✅ Analysis completed successfully!")
        print("\\n📁 Files generated:")
        print("- optimal_clusters.png")
        print("- customer_segmentation_analysis.png")
        print("- customer_segments.csv")
        print("- cluster_profiles.csv")
        print("- business_insights.csv")
        
        print("\\n🎯 Next steps:")
        print("1. Review the generated visualizations")
        print("2. Examine cluster profiles and business insights")
        print("3. Run 'streamlit run streamlit_app.py' for interactive dashboard")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your data file and try again.")

if __name__ == "__main__":
    main()