import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from perceptron_class import Perceptron

# Define local cache file
cache_file = 'iris_data.csv'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Check if cached file exists
if os.path.exists(cache_file):
    print('Loading from cache:', cache_file)
    df = pd.read_csv(cache_file, header=None, encoding='utf-8')
else:
    print('Downloading from URL:', url)
    df = pd.read_csv(url, header=None, encoding='utf-8')
    # Save to cache
    df.to_csv(cache_file, index=False)
    print('Saved to cache:', cache_file)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
print(y)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1],
              color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')

plt.savefig('output.png', dpi=300, bbox_inches='tight')
print('Plot saved to output.png')

# iniciar a classe Perceptron
ppn = Perceptron(eta=0.0001, n_iter=50)

# treinar o modelo
ppn.fit(X, y)

fig_2 = plt.figure()
# plotar as interações que geraram erros
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='^')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.savefig('errors_plot.png', dpi=300, bbox_inches='tight')

