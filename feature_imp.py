
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path("models/nids_model.joblib")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["feature_columns"]

forest = model.named_steps["classifier"]

importance = pd.Series(
    forest.feature_importances_,
    index=features,
).sort_values(ascending=False)

print("Top 15 most important features:")
print(importance.head(15).to_string())

plt.figure(figsize=(10, 7))
importance.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=150,
)
plt.show()
