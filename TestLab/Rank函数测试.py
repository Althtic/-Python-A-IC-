import pandas as pd

df = pd.DataFrame({
    'group': ['A', 'A', 'A', 'A'],
    'value': [10, 30, 20, 40]
})

df['rank_pct_default'] = df.groupby('group')['value'].rank(pct=True)
print(df)