import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n = 10  # Number of rows

# Generate synthetic data
ages = np.random.randint(18, 70, size=n)
incomes = np.random.randint(15, 150, size=n)  # in k$
scores = np.random.randint(1, 101, size=n)

# Define numeric segment logic similar to clustering labels
def assign_segment(age, income, score):
    if income > 100 and score > 70:
        return 4  # High-Value
    elif income < 50 and score < 40:
        return 0  # Low-Value
    elif age < 30 and score > 75:
        return 1  # Young-Spender
    elif age > 55 and income > 80:
        return 2  # Senior-Affluent
    else:
        return 3  # Mid-Tier

# Apply segment assignment
segments = [assign_segment(age, income, score) for age, income, score in zip(ages, incomes, scores)]

# Create DataFrame
df = pd.DataFrame({
    "age": ages,
    "income": incomes,
    "score": scores,
    "segment": segments
})

# Save to CSV
df.to_csv("customer_segments_500.csv", index=False)
