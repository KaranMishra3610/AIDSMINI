import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
# Log-transform likes
df['likes_log'] = np.log1p(df['likes'])  # log(1 + likes)
df['likes_original'] = df['likes']       # keep original likes for visualization

# Clean and process prompt text
df['prompt_clean'] = df['prompt'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', str(x).lower()))

# TF-IDF Vectorization for prompt with n-grams
# Using n-grams to capture multi-word phrases
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['prompt_clean']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

# Combine TF-IDF with original dataframe
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# Add interaction term
df['gpu_likes_interaction'] = df['gpu_usage'] * df['likes']

# =======================
# Dimensionality Reduction (PCA)
# =======================
print("\n=== DIMENSIONALITY REDUCTION ===")
# Define columns to drop
drop_cols = ['prompt', 'prompt_clean', 'likes_log', 'likes_original']
if 'image_id' in df.columns:
    drop_cols.append('image_id')

# Get numerical features for PCA
X_for_pca = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

# Scale features
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_for_pca)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_pca_scaled)

# Save PCA plot
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['likes_log'], cmap='coolwarm', s=10)
plt.colorbar(label='Log(Likes)')
plt.title("PCA - Feature Reduction Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_visualization.png")
plt.close()

# =======================
# Likes Prediction Modeling
# =======================
print("\n=== LIKES PREDICTION MODELING ===")
target_likes = 'likes_log'

# Prepare feature set for likes prediction
drop_cols_likes = ['style_accuracy_score', 'prompt', 'prompt_clean', 'likes_log', 'likes_original']
if 'image_id' in df.columns:
    drop_cols_likes.append('image_id')

X_likes = df.drop(columns=drop_cols_likes).select_dtypes(include=[np.number])
y_likes = df[target_likes]

# Scale features
scaler_likes = StandardScaler()
X_likes_scaled = scaler_likes.fit_transform(X_likes)

# Train-Test Split for likes prediction
X_train_likes, X_test_likes, y_train_likes, y_test_likes = train_test_split(
    X_likes_scaled, y_likes, test_size=0.2, random_state=42
)

# =======================
# Evaluation Function
# =======================
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{model_name}] Evaluation Metrics:")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

# =======================
# Likes Prediction Models
# =======================
print("\n=== TRAINING LIKES PREDICTION MODELS ===")
# Linear Regression
lr_likes = LinearRegression()
lr_likes.fit(X_train_likes, y_train_likes)
y_pred_lr_likes = lr_likes.predict(X_test_likes)
lr_likes_metrics = evaluate_model(y_test_likes, y_pred_lr_likes, "Linear Regression (Likes)")

# Grid Search for Random Forest
print("\n=== HYPERPARAMETER TUNING FOR RANDOM FOREST ===")
rf_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf_gs = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=3, scoring='neg_mean_squared_error')
rf_gs.fit(X_train_likes, y_train_likes)
rf_best = rf_gs.best_estimator_
y_pred_rf_likes = rf_best.predict(X_test_likes)
rf_likes_metrics = evaluate_model(y_test_likes, y_pred_rf_likes, "Tuned Random Forest (Likes)")

# Grid Search for Gradient Boosting
print("\n=== HYPERPARAMETER TUNING FOR GRADIENT BOOSTING ===")
gb_grid = {
    'n_estimators': [100],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5]
}
gb_gs = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_grid, cv=3, scoring='neg_mean_squared_error')
gb_gs.fit(X_train_likes, y_train_likes)
gb_best = gb_gs.best_estimator_
y_pred_gb_likes = gb_best.predict(X_test_likes)
gb_likes_metrics = evaluate_model(y_test_likes, y_pred_gb_likes, "Gradient Boosting (Tuned, Likes)")

# Feature Importance Analysis
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
importances = rf_best.feature_importances_
important_indices = np.argsort(importances)[::-1][:20]
important_features = X_likes.columns[important_indices]

# Print top 10 important features
print("\nTop 10 Important Features for Likes Prediction:")
for i, feature in enumerate(important_features[:10]):
    print(f"{i+1}. {feature}: {importances[important_indices[i]]:.4f}")

# RF with Top 20 Features
X_important = X_likes[important_features]
X_imp_scaled = scaler_likes.fit_transform(X_important)
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_imp_scaled, y_likes, test_size=0.2, random_state=42)
rf_imp = RandomForestRegressor(random_state=42)
rf_imp.fit(X_train_imp, y_train_imp)
y_pred_imp = rf_imp.predict(X_test_imp)
rf_imp_metrics = evaluate_model(y_test_imp, y_pred_imp, "RF with Top 20 Important Features (Likes)")

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
clustering_features = ['gpu_usage', 'likes_log', 'style_accuracy_score']
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

# 📊 Likes Prediction Regression Summary
print("\n📊 Likes Prediction Model Results:")
likes_results = {
    "Linear Regression": lr_likes_metrics,
    "Tuned Random Forest": rf_likes_metrics,
    "Gradient Boosting (Tuned)": gb_likes_metrics,
    "RF (Top 20 Features)": rf_imp_metrics
}

for model, metrics in likes_results.items():
    print(f"\n🔹 {model}:")
    for metric, value in metrics.items():
        print(f"   {metric: <6}: {value:.4f}")

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
print(f"Important Features for Likes Prediction: {important_features[:10].to_list()}")
print(f"Clusters identified: {df['cluster'].nunique()} unique clusters.")
print(f"Top Common Keyword Combinations: {df['top_keywords'].value_counts().head(5).index.tolist()}")

print("\n✅ End of Summary")
print("="*60)