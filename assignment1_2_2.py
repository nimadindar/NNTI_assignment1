import numpy as np
import matplotlib.pyplot as plt

def _create_meshgrid(x, grid_step):

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    return xx, yy

def plot_decision_boundry(x,y, grid_step, classifier, title):

    xx, yy = _create_meshgrid(x, grid_step)

    clf = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    clf = clf.reshape(xx.shape)

    plt.contourf(xx, yy, clf, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.coolwarm)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

def plot_double_decision_boundry(x,y,grid_step, classifier1, classifier2, title):

    xx, yy = _create_meshgrid(x,grid_step)

    clf1 = classifier1.predict(np.c_[xx.ravel(), yy.ravel()])
    clf1 = clf1.reshape(xx.shape)

    clf2 = classifier2.predict(np.c_[xx.ravel(), yy.ravel()])
    clf2 = clf2.reshape(xx.shape)

    plt.contourf(xx, yy, clf1, alpha=0.3, cmap=plt.cm.coolwarm,linestyles='-')

    plt.contour(xx, yy, clf2, alpha=0.3, cmap=plt.cm.PuOr, linestyles='--')

    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, edgecolor='k', cmap=plt.cm.coolwarm)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()