import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# =======================
# Load Dataset
# =======================
print("\n=== LOADING AND PREPROCESSING DATA ===")
df = pd.read_csv("ai_ghibli_trend_dataset_v2.csv")
df.dropna(inplace=True)

# =======================
# Feature Engineering
# =======================
print("\n=== FEATURE ENGINEERING ===")
# Keep original likes for visualization
df['likes_original'] = df['likes']

# Clean and process prompt text
df['prompt_clean'] = df['prompt'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', str(x).lower()))

# TF-IDF Vectorization for prompt with n-grams
# Using n-grams to capture multi-word phrases
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['prompt_clean']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

# Combine TF-IDF with original dataframe
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# =======================
# Dimensionality Reduction (PCA)
# =======================
print("\n=== DIMENSIONALITY REDUCTION ===")
# Define columns to drop
drop_cols_pca = ['prompt', 'prompt_clean', 'likes_original', 'likes']
if 'image_id' in df.columns:
    drop_cols_pca.append('image_id')

# Get numerical features for PCA
X_for_pca = df.drop(columns=drop_cols_pca).select_dtypes(include=[np.number])

# Scale features
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_for_pca)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_pca_scaled)

# Save PCA plot
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['likes'], cmap='coolwarm', s=10)
plt.colorbar(label='Likes')
plt.title("PCA - Feature Reduction Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_visualization.png")
plt.close()

# =======================
# Anomaly Detection on GPU Usage
# =======================
print("\n=== ANOMALY DETECTION ===")
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(df[['gpu_usage']])
total_anomalies = (df['anomaly'] == -1).sum()
print(f"\n[Anomaly Detection] Total anomalies detected in GPU usage: {total_anomalies}")

# =======================
# Clustering (Unsupervised Learning)
# =======================
print("\n=== CLUSTERING ANALYSIS ===")
# Define features for clustering
clustering_features = ['gpu_usage', 'likes', 'style_accuracy_score']
X_cluster = df[clustering_features]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

# Add clusters and PCA components to DataFrame
df['cluster'] = clusters
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# PCA-based Cluster Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='cluster', palette='Set2')
plt.title("K-Means Clusters (PCA Visualization)")
plt.savefig("kmeans_clusters.png")
plt.close()

# Likes vs Style Accuracy (Cluster Colored)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='likes_original', y='style_accuracy_score', hue='cluster', palette='Set2')
plt.title("Clusters by Likes vs Style Accuracy Score")
plt.xlabel("Likes")
plt.ylabel("Style Accuracy Score")
plt.savefig("cluster_likes_vs_accuracy.png")
plt.close()

# Cluster Summary
cluster_summary = df.groupby('cluster')[['gpu_usage', 'likes_original', 'style_accuracy_score']].mean()
print("\n[Cluster Analysis] Averages by Cluster:\n", cluster_summary.round(2))

# =======================
# Recommendation System with Multiple Keywords
# =======================
print("\n=== RECOMMENDATION SYSTEM ===")
print("\n[Recommendation System] Finding best combinations to maximize style accuracy score and likes...\n")

# Prepare data for recommendation
df['gpu_usage_rounded'] = df['gpu_usage'].round()

# Calculate multiple top TF-IDF keywords for each prompt
tfidf_columns = [col for col in df.columns if col.startswith("tfidf_")]
TOP_N_KEYWORDS = 3  # Extract top 3 keywords per prompt

# Get feature names from TF-IDF vectorizer
feature_names = tfidf.get_feature_names_out()
top_prompt_words = {f'tfidf_{i}': word for i, word in enumerate(feature_names)}

# For storing multiple keywords
df['top_keywords'] = ""

# For each row, get top N keywords
for idx, row in df.iterrows():
    # Get TF-IDF values for this row
    tfidf_values = row[tfidf_columns].values
    # Get indices of top N values
    top_indices = np.argsort(-tfidf_values)[:TOP_N_KEYWORDS]
    # Convert to column names
    top_cols = [tfidf_columns[i] for i in top_indices]
    # Map to actual words
    top_words = [top_prompt_words[col] for col in top_cols]
    # Join with comma
    df.at[idx, 'top_keywords'] = ", ".join(top_words)

# Keep single top word for backward compatibility
df['top_prompt_weight'] = df[tfidf_columns].idxmax(axis=1)
df['top_prompt_word'] = df['top_prompt_weight'].map(top_prompt_words)

# Create keyword categories based on combinations
print("\n[Keyword Analysis] Creating keyword categories based on common combinations...\n")
# Get most common keyword patterns
keyword_counts = df['top_keywords'].value_counts().head(20)
print("Most common keyword combinations:")
print(keyword_counts)

# Group data for recommendations - using multiple keywords
grouped_multi = df.groupby(['platform', 'gpu_usage_rounded', 'resolution', 'top_keywords']).agg({
    'style_accuracy_score': 'mean',
    'likes_original': 'mean',
    'platform': 'count'
}).rename(columns={'platform': 'count'}).reset_index()

top_by_accuracy_multi = grouped_multi.sort_values(by='style_accuracy_score', ascending=False).head(10)
top_by_likes_multi = grouped_multi.sort_values(by='likes_original', ascending=False).head(10)

# For comparison and backward compatibility - group by single keyword too
grouped = df.groupby(['platform', 'gpu_usage_rounded', 'resolution', 'top_prompt_word']).agg({
    'style_accuracy_score': 'mean',
    'likes_original': 'mean',
    'platform': 'count'
}).rename(columns={'platform': 'count'}).reset_index()

top_by_accuracy = grouped.sort_values(by='style_accuracy_score', ascending=False).head(10)
top_by_likes = grouped.sort_values(by='likes_original', ascending=False).head(10)

# Print results using multiple keywords
print("\n🎨 Top 10 Recommendations based on Style Accuracy Score (Multiple Keywords):")
print(top_by_accuracy_multi[['platform', 'gpu_usage_rounded', 'resolution', 'top_keywords', 'style_accuracy_score', 'likes_original']].to_string(index=False))

print("\n❤️ Top 10 Recommendations based on Likes (Multiple Keywords):")
print(top_by_likes_multi[['platform', 'gpu_usage_rounded', 'resolution', 'top_keywords', 'likes_original', 'style_accuracy_score']].to_string(index=False))

# Print results using single keyword for comparison
print("\n🎨 Top 10 Recommendations based on Style Accuracy Score (Single Keyword):")
print(top_by_accuracy[['platform', 'gpu_usage_rounded', 'resolution', 'top_prompt_word', 'style_accuracy_score', 'likes_original']].to_string(index=False))

print("\n❤️ Top 10 Recommendations based on Likes (Single Keyword):")
print(top_by_likes[['platform', 'gpu_usage_rounded', 'resolution', 'top_prompt_word', 'likes_original', 'style_accuracy_score']].to_string(index=False))

# Visualize Top Multi-Keyword Recommendations
plt.figure(figsize=(14, 8))
# Plot top 5 for clarity
sns.barplot(data=top_by_accuracy_multi.head(5), x='platform', y='style_accuracy_score', hue='resolution')
plt.title("Top Platforms + Resolutions by Style Accuracy (Multi-Keyword)")
plt.ylabel("Avg Style Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_recommendations_accuracy_multi.png")
plt.close()

plt.figure(figsize=(14, 8))
sns.barplot(data=top_by_likes_multi.head(5), x='platform', y='likes_original', hue='resolution')
plt.title("Top Platforms + Resolutions by Likes (Multi-Keyword)")
plt.ylabel("Avg Likes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_recommendations_likes_multi.png")
plt.close()

# ==============================
# ✅ FINAL SUMMARY BLOCK
# ==============================
print("\n" + "="*60)
print("🔍 FINAL PROJECT SUMMARY: AI Ghibli Art Engagement Analysis")
print("="*60)

# 🚨 Anomaly Detection Summary
print(f"\n🚨 Anomaly Detection:")
print(f"   ➤ Total anomalies in GPU usage: {total_anomalies}")

# 🔗 Clustering Summary
print("\n🔗 Clustering Summary (KMeans):")
print(cluster_summary.round(2).to_string())

# 🎯 Recommendation Highlights
print("\n🎯 Recommendation Highlights (Multiple Keywords):")

print("\n🎨 Top 5 by Style Accuracy:")
print(top_by_accuracy_multi[['platform', 'gpu_usage_rounded', 'resolution', 'top_keywords', 'style_accuracy_score', 'likes_original']].head(5).to_string(index=False))

print("\n❤️ Top 5 by Likes:")
print(top_by_likes_multi[['platform', 'gpu_usage_rounded', 'resolution', 'top_keywords', 'likes_original', 'style_accuracy_score']].head(5).to_string(index=False))

# Data Info
print(f"\nData Loaded from: {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"Clusters identified: {df['cluster'].nunique()} unique clusters.")
print(f"Top Common Keyword Combinations: {df['top_keywords'].value_counts().head(5).index.tolist()}")

print("\n✅ End of Summary")
print("="*60)