import pandas as pd

df = pd.read_csv('data/customers.csv')

def assign_segment(row):
    age, income, score = row['age'], row['income'], row['score']

    if income < 40 and score < 40:
        return 0  # Budget-Conscious
    elif age <= 30 and 40 <= score <= 70:
        return 1  # Young Explorers
    elif income >= 90 and score >= 70:
        return 2  # High Income, High Spend
    elif 40 <= income <= 80 and 40 <= score <= 70:
        return 3  # Moderate Spenders
    elif (income < 40 and score >= 70) or (score > 90):
        return 4  # Occasional Shoppers
    else:
        return 3  # Default to moderate

df['new_segment'] = df.apply(assign_segment, axis=1)

# Save the enhanced version
df.to_csv('data/enhanced_customers.csv', index=False)