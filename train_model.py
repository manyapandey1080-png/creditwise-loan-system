import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
X = np.array([
    [50000, 20000, 100000, 750],
    [20000, 10000, 150000, 600],
    [80000, 30000, 200000, 800],
    [25000, 5000, 120000, 580]
])

y = [1, 0, 1, 0]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")