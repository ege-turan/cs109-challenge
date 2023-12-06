import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load the data from pwdb_haemod_params.csv
data = pd.read_csv(r'BP\pwdb_haemod_params.csv')

# Select relevant columns for analysis
columns_of_interest = [' age [years]', ' HR [bpm]', ' SV [ml]', ' CO [l/min]', ' LVET [ms]', ' dp/dt [mmHg/s]', ' PFT [ms]', ' RFV [ml]', ' SBP_a [mmHg]', ' DBP_a [mmHg]', ' MAP_a [mmHg]', ' PP_a [mmHg]', ' SBP_b [mmHg]', ' DBP_b [mmHg]', ' MBP_b [mmHg]', ' PP_b [mmHg]', ' PP_amp [ratio]', ' AP [mmHg]', ' AIx [%]', ' Tr [ms]', ' PWV_a [m/s]', ' PWV_cf [m/s]', ' PWV_br [m/s]', ' PWV_fa [m/s]', ' dia_asca [mm]', ' dia_dta [mm]', ' dia_abda [mm]', ' dia_car [mm]', ' Len [mm]', ' drop fin [mmHg]', ' drop ankle [mmHg]', ' SVR [10^6 Pa s / m3]']

# Create a correlation matrix
correlation_matrix = data[columns_of_interest].corr()

# Sort parameters based on correlation with blood pressure
sorted_params_sbp = correlation_matrix[' SBP_a [mmHg]'].abs().sort_values(ascending=False)
sorted_params_dbp = correlation_matrix[' DBP_a [mmHg]'].abs().sort_values(ascending=False)
sorted_params_mbp = correlation_matrix[' MAP_a [mmHg]'].abs().sort_values(ascending=False)
sorted_params_pp = correlation_matrix[' PP_a [mmHg]'].abs().sort_values(ascending=False)

# Print or visualize the results
print("Sorted Parameters for SBP:")
print(sorted_params_sbp)

print("\nSorted Parameters for DBP:")
print(sorted_params_dbp)

print("\nSorted Parameters for MBP:")
print(sorted_params_mbp)

print("\nSorted Parameters for PP:")
print(sorted_params_pp)

# Plot correlation matrix heatmap
plt.figure(figsize=(12, 10))
plt.title("Correlation Matrix Heatmap")
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(columns_of_interest)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(columns_of_interest)), correlation_matrix.columns)
plt.show()