# src/models/feature_importance.py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create reports directory if needed
Path('reports').mkdir(exist_ok=True)

# Load model
with open('models/lightgbm_v2_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importances
features = pd.read_csv('data/processed/selected_features.csv')['feature'].tolist()
importances = model.feature_importances_

# Create DataFrame
df_imp = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

# Save CSV ← THIS WAS MISSING!
df_imp.to_csv('reports/feature_importance.csv', index=False)
print(f"✓ Saved feature importance CSV: reports/feature_importance.csv")

# Plot top 20
plt.figure(figsize=(12, 8))
sns.barplot(data=df_imp.head(20), x='importance', y='feature')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=150)
print(f"✓ Saved feature importance plot: reports/feature_importance.png")

print("\nTop 20 features:")
print(df_imp.head(20))
