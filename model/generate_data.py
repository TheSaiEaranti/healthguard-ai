import pandas as pd
import numpy as np

np.random.seed(42)

n = 500  # number of synthetic patients

data = {
    'age': np.random.randint(30, 80, n),
    'blood_pressure': np.random.randint(90, 180, n),
    'cholesterol': np.random.randint(150, 300, n),
    'glucose_level': np.random.randint(70, 200, n),
    'smoking': np.random.randint(0, 2, n),  # 0 = no, 1 = yes
    'bmi': np.round(np.random.normal(27, 5, n), 1),
}

df = pd.DataFrame(data)
df['risk_label'] = (
    (df['age'] > 60).astype(int) +
    (df['blood_pressure'] > 140).astype(int) +
    (df['cholesterol'] > 240).astype(int) +
    (df['glucose_level'] > 140).astype(int) +
    (df['smoking'] == 1).astype(int) +
    (df['bmi'] > 30).astype(int)
) >= 3

df['risk_label'] = df['risk_label'].astype(int)
df.to_csv('data/patients.csv', index=False)
print("Synthetic data saved to data/patients.csv")
