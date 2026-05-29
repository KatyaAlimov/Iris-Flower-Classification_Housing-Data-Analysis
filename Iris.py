import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris          # get the function that loads the data set

iris_Data = load_iris()            # load data into object
X, y = iris_Data['data'], iris_Data['target']
         
 # pca 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#standarize features
X_norm = StandardScaler().fit_transform(X)
#fit pca
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_norm)

names = ["Setosa", "Versicolour", "Virginica"]
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("Iris PCA (2D)")
plt.xlabel("PC 1", color = 'red')
plt.ylabel("PC 2", color = 'red')

#create the side bar with classes 
plt.legend(handles=scatter.legend_elements()[0], labels=names, title="Classes")
plt.show()

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_norm)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c= y)
ax.set_title("Iris PCA (3D)")
ax.set_xlabel("PC one", color = "red")
ax.set_ylabel("PC two", color = "red")
ax.set_zlabel("PC three", color = "red")
ax.legend(handles=scatter.legend_elements()[0], labels=names, title="Classes")
plt.show()


import pandas as pd
# task 2: 
#load the dataset
housing_data = pd.read_csv('housing_training.csv', header=None, sep=',')
# denote the column names for the housing dataset
print("Top Rows of Hosuing Data: ")
print(housing_data.head()) # for debugging print first 5 row of dataset
#changing columns names
print("Changed Col Names: ")
housing_data.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
print(housing_data.head())
#print info to make sure data is clean 
housing_data.info()

plt.figure(figsize=(10, 6))
# the colums I want to plot with 
housing_data.boxplot(column=["k", "m", "n"])

plt.xlabel("Colums")
plt.ylabel("Values")
plt.title("Boxplot for the housing dataset")
plt.xticks([1, 2, 3], ["k", "m", "n"])
plt.show()

# task 3 
plt.figure(figsize=(10,5))
plt.hist(housing_data['a'], bins=20, color='pink', edgecolor='blue')
plt.xlabel ('Crime rate')
plt.ylabel ('Number of towns')
plt.title('Distribution of crime per Capita')
plt.show()


