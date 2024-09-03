import pandas as pd

# Paths to your CSV files
clean_data_path = ''
dirty_data_path = ''
result_data_path = ''
clean_data = pd.read_csv(clean_data_path, dtype=str, encoding='utf-8', keep_default_na=False, na_values=['', ])
dirty_data = pd.read_csv(dirty_data_path, dtype=str, encoding='utf-8', keep_default_na=False, na_values=['', ])
clean_data.fillna('null', inplace=True)
dirty_data.fillna('null', inplace=True)
indicators = (clean_data != dirty_data).astype(int)
print(indicators.sum().sum())
# 保存检测结果为CSV文件
indicators.to_csv(result_data_path, index=False, encoding='utf-8')
# Read the clean and dirty data
print('Comparison complete.')
