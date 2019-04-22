# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#print('hello world')

from sklearn import datasets 
import numpy as np

iris  = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target #the data we're trying to predict 

print('Class labels: ', np.unique(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y: ', np.bincount(y) )
print('Labels counts in y_train: ', np.bincount(y_train) )
print('Labels counts in y_test: ', np.bincount(y_test) )


'''
The standard score of a sample x is calculated as:

z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False, and
 s is the standard deviation of the training samples or one if with_std=False.
'''

from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()

'''
* Took from StackExchange: 
  https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models?newreg=870fdba765bc441ebef071ed70bd7b43  

To center the data (make it have zero mean and unit standard error), you
subtract the mean and then divide the result by the standard deviation.

x′=x−μσ

You do that on the training set of data. But then you have to apply the same
transformation to your testing set (e.g. in cross-validation), or to newly
obtained examples before forecast. But you have to use the same two parameters
 μ and σ (values) that you used for centering the training set.

Hence, every sklearn's transform's fit() just calculates the parameters (e.g. μ
and σ in case of StandardScaler) and saves them as an internal objects state.
Afterwards, you can call its transform() method to apply the transformation to
a particular set of examples.

fit_transform() joins these two steps and is used for the initial fitting of 
parameters on the training set x, but it also returns a transformed x′. 
Internally, it just calls first fit() and then transform() on the same data.
'''


sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

"""

fix the line below: 

#print( 'Misclassified samples: %d' % y.test!=  y_pred.sum() ) #what does this line do?????

"""

from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('\n\n')


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

"""
Q on Stack Exchange 

How are the two declarations different from one another?
""" 

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02): 
    """
    setup marker generator and color map
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    '''
    plot the decision surface
    '''
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

'''
Fnction above is copied from book
'''



# Training a perceptron model using the standardized training data:



X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()


'''
FINDING: Classes are not linearly seperable

Things to look at: meshgrid, decision_region, np.hstack, 
'''

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100, random_state=1)

'''
Parameters passsed. 
Link: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

C :     float, default: 1.0
Inverse of regularization strength; must be a positive float. 
Like in support vector machines, smaller values specify stronger regularization.


random_state :      int, RandomState instance or None, optional, default: None
The seed of the pseudo random number generator to use when shuffling the data.
 If int, random_state is the seed used by the random number generator; If
 RandomState instance, random_state is the random number generator;
 If None, the random number generator is the RandomState instance used 
 by np.random. Used when solver == ‘sag’ or ‘liblinear’.
'''

lr.fit(X_train_std, y_train)

plot_decision_regions (X_combined_std, 
                       y_combined, 
                       classifier = lr, 
                       test_idx = range(105, 150)
        )

plt.xlabel('petal length standardized')
plt.ylabel('petal width standardized')
plt.legend(loc = 'upper left')
plt.show()

weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C= 10.**c, random_state=1)
    lr.fit(X_train, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
    
weights  = np.array(weights)
plt.plot(params, weights[:0], 
         label='petal length')

plt.plot(params, weights[:1], linestyle= '--', label='petal width')
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.legend(loc='uper left')
plt.xscale('log')
plt.show() # not showing