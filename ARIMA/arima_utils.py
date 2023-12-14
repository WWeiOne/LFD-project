import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Utils functions for ARIMA

# Print AIC after finishing the training grid search.
# Save it to file for latter plot.
def print_and_save_AIC(results_list):
    print('ARIMA,Seasonal,AIC')
    for (param, param_seasonal, aic) in results_list:
        param_str = ''.join(map(str, param))
        seasonal_str = ''.join(map(str, param_seasonal[:-1]))
        print(f"{param_str},{seasonal_str},{int(aic)}")

    with open('AIC.csv', 'w') as file:
        file.write('ARIMA,Seasonal,AIC\n')
        for (param, param_seasonal, aic) in results_list:
            param_str = ''.join(map(str, param))
            seasonal_str = ''.join(map(str, param_seasonal[:-1]))
            file.write(f"{param_str},{seasonal_str},{int(aic)}\n")

# plot AIC results for grid search
def plot_AIC():
    file_path = 'AIC.csv'  
    df = pd.read_csv(file_path, dtype={'ARIMA': str, 'Seasonal': str})

    pivot_table = df.pivot(index='ARIMA', columns='Seasonal', values='AIC')

    max_aic = pivot_table.max().max()
    min_aic = pivot_table.min().min()
    aic_normalized = (pivot_table - min_aic) / (max_aic - min_aic)

    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(pivot_table, cmap='Blues', vmin=min_aic, vmax=max_aic)

    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            ax.text(j, i, int(pivot_table.iloc[i, j]), va='center', ha='center', color='black')

    ax.set_xlabel('Seasonal Parameters')
    ax.set_ylabel('ARIMA Parameters')
    ax.set_title('AIC Values for Different ARIMA and Seasonal Parameters')

    plt.show()