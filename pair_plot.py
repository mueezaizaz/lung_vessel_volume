import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#def plot_pair_plots

df_read = pd.read_excel(r'G:\My Drive\CV-2021\Applications 2021\BRAINOMIX-Algorithm Researcher\BRAINOMIX challenge/Results.xlsx', sheet_name='Combined Results')
print(df_read)

sns.pairplot(df_read, diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k',})
plt.show()
sns.pairplot(df_read, hue = 'lung_vessel_ratio (%)', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
plt.show()
