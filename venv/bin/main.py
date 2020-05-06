# This will explore the K-Nearest Neighbours Algo.

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv"

# Step 1 (Always) Load the data

df = pd.read_csv(path)
print(df.head())

# Step 2: Visualize our data
# Assume out target is custcat (i.e. Customer Category)
print(df['custcat'].value_counts())

# Lets view as a histogram
df.hist(column="income", bins=50)

# Huh, this isn't very helpful due to outliers.

plt.show()

# Step 3: We need a numpy array of our feature sets for the classifier
print(df.columns)

# Lets take everything but our target
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
print(X[0:5])

# Now, lets get our target. Notice how this is a 1D array.
y = df["custcat"].values
print(y[0:5])

# Looks good!

# Step 4: Normalize the data
# Since KNN uses Euclidean distance, we have to normalize so large values
# like income don't overrule smaller values like age

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

# Nice, they're all z-scores.

# Step 5: we now need to do a train test split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Step 6: We have our data. It isn't perfectly wrangled, but that isn't
# The purpose of this exercise.
# Now, lets try to use k=4 and get yhat

k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

yhat = neigh.predict(X_test)
print(yhat[0:5])

# Step 7: Now that we have yhat, lets check the accuracy.
# We will use Jaccard Index
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# That's not great, especially out of test.
# Maybe we try it with k=6
# Not much better

k = 6
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

yhat = neigh.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Okay, so now we should iteratively determine the best k value
# Maybe we can graph it and see

maxK = 100
mean_acc = np.zeros((maxK - 1)) # Creates a 1D array with 9 elements
std_acc =  np.zeros((maxK - 1))
ConfusionMatrix = []

for n in range(1,maxK):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat) # only looking at out-of-sample
    # This is the Root Mean Square Error. Shows the range of error in the graph
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Plotting
plt.plot(range(1,maxK),mean_acc,'g')
plt.fill_between(range(1,maxK),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3x std'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# Looks like k=38 would be optimal. Still bad
# To improve the accuracy here, fields should be removed from the predictor set
# For example, it's possible that address doesn't really affect
# Or marital status. These are guesses, and bad ones at that.
# More models should be evaluated to determine the best fit.
# However, it's worth noting that there are 4 categories, and this is 41%
# Accurate. This is better than chance.


