import pandas as pd

# Paths to your CSV files
clean_data_path = f'datasets/tax/tax_clean_1000.csv'
dirty_data_path = f'tax-1k-error-rate/error_rate_0.5.csv'
result_data_path = f'tax-1k-error-rate/error_rate_0.5_error_detection.csv'
clean_data = pd.read_csv(clean_data_path, dtype=str, encoding='utf-8', keep_default_na=False, na_values=['', ])
dirty_data = pd.read_csv(dirty_data_path, dtype=str, encoding='utf-8', keep_default_na=False, na_values=['', ])
clean_data.fillna('null', inplace=True)
dirty_data.fillna('null', inplace=True)
indicators = (clean_data != dirty_data).astype(int)
print(indicators.sum().sum())
# 保存指示符DataFrame为CSV文件
indicators.to_csv(result_data_path, index=False, encoding='utf-8')
# Read the clean and dirty data
print('Comparison complete. Discrepancies marked in ***_error_detection.csv.')
