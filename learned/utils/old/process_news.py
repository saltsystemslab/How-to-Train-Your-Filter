import pandas as pd
# Combine the Fake and True Datasets into one combined dataset 
true_df = pd.read_csv('True.csv')
true_df['label'] = '1'
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = '-1'
combined = pd.concat([true_df, fake_df], ignore_index=True)
combined.columns = ['key','text','subject','date','label']
combined.to_csv('News_data.csv', index=False)
