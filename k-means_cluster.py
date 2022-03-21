import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

X = pd.read_excel("G:\My Drive\CV-2021\Applications 2021\BRAINOMIX-Algorithm Researcher\BRAINOMIX "
                  "challenge\Results.xlsx", sheet_name="Combined Results", usecols=[1, 2, 3], names=['lung volume ml', 'vessel_vol ml', 'lung_vessel_ratio (%)'])
df=X.copy()
X = X.iloc[:, 0:-1]
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled, columns=['lung volume ml','vessel_vol ml'])

#normalized_df = (X - X.mean()) / X.std()

#X = normalized_df
distorsions = []
for k in range(1, 9):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(1, 9), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()

kmeans_fin = KMeans(n_clusters=4)
clusters_final = kmeans_fin.fit_predict(X)
X['Clusters'] = clusters_final
X['vessel_ratio'] = df['lung_vessel_ratio (%)']
print(clusters_final)

'fig,ax = plt.subplots()'
sns.scatterplot(data=X, hue='Clusters', x='lung volume ml', y='vessel_vol ml',  palette="deep")
for i in range(X.shape[0]): plt.text(X.iloc[i, 0], X.iloc[i, 1], str(i+1))
plt.show()

sns.pairplot(X,hue='Clusters',  diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
plt.show()


sns.scatterplot(data=X, x='lung volume ml', y='vessel_ratio',  palette="deep")
#for i in range(X.shape[0]): plt.text(X.iloc[i, 0], X.iloc[i, 1], str(i+1))
plt.show()

sns.scatterplot(data=X, x='vessel_vol ml', y='vessel_ratio',  palette="deep")
#for i in range(X.shape[0]): plt.text(X.iloc[i, 0], X.iloc[i, 1], str(i+1))
plt.show()

fig, ax = plt.subplots()
sns.scatterplot(data=X, x='vessel_vol ml', y='vessel_ratio',  palette="deep")
ax.set_xlim(0, 0.2)
plt.show()