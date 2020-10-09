import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from ipywidgets import interact, ToggleButtons

'''Modified from luwiji package'''
def demo():
    def knn(N=1, x=-5, y=0, show_decision=False):
        X_data, y_data = make_blobs(n_samples=20, centers=2, cluster_std=3, random_state=1)
        X_test = np.array([[x, y]])
        distance = np.linalg.norm(X_data - X_test, axis=1)
        closest = distance.argsort()
        clf = KNeighborsClassifier(n_neighbors=N).fit(X_data, y_data)
        y_pred = clf.predict(X_test)
        y_pred = 'r' if y_pred[0] else 'b'
        plt.figure(figsize=(7, 7))
        plt.xlim(-13, 4)
        plt.ylim(-8, 9)
        for idx in closest[:N]:
            fmt = 'b-' if y_data[idx] == 0 else 'r-'
            w = 2
            plt.plot((X_test[0, 0], X_data[idx, 0]), (X_test[0, 1], X_data[idx, 1]), fmt, zorder=-1, linewidth=w)
            plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, s=100, cmap='bwr', edgecolors='c')
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=100, cmap='bwr', edgecolors='k', marker='D');
            if show_decision:
                xx = np.linspace(-13, 4, 200)
                yy = np.linspace(-8, 9, 200)
                X1, X2 = np.meshgrid(xx, yy)
                X_grid = np.c_[X1.ravel(), X2.ravel()]
                decision = clf.predict_proba(X_grid)[:, 1]
                plt.contourf(X1, X2, decision.reshape(X1.shape), levels=[0, 0.5, 1], alpha=0.3, cmap='bwr', zorder=-2)
    return interact(knn, N=(1, 11, 2), x=(-10, 0, 0.5), y=(-5, 6, 0.5))
 
'''Modified from Brendon Hall tutorial'''
 
def plot_log(df, well):
    df = df[df['Well Name'] == well]
    fig = plt.figure(figsize=(8, 12))
    # create all axes we need
    
    facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72',
                     '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

    # logs = logs.sort_values('Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')

    cluster=np.repeat(np.expand_dims(df.Facies,1), 100, 1)
    
    ztop=df.index.min(); zbot=df.index.max()
    
    #Plot Gamma Ray    
    ax0 = plt.subplot(161)
    ax0.plot(df.GR, df.index, ls='-', color='green')
    ax0.set_ylim(ztop,zbot)
    ax0.invert_yaxis()
    ax0.grid()
    ax0.locator_params(axis='x', nbins=3)
    ax0.set_xlabel("Gamma Ray")

    #Plot Medium Resistivity
    ax1 = plt.subplot(162)
    ax1.plot(df.ILD_log10, df.index, ls='-', color='blue')
    ax1.set_ylim(ztop,zbot)
    ax1.invert_yaxis()
    ax1.grid()
    ax1.locator_params(axis='x', nbins=3)
    ax1.set_xlabel("Resistivity")
    ax1.set_yticklabels([])
    
    #Plot Deep Resistivity
    ax2 = plt.subplot(163)
    ax2.plot(df.DeltaPHI, df.index, ls='-', color='grey')
    ax2.set_ylim(ztop,zbot)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.locator_params(axis='x', nbins=3)
    ax2.set_xlabel("DeltaPHI")
    ax2.set_yticklabels([])
    
    #Plot Neutron Porosity Difference
    ax3 = plt.subplot(164)
    ax3.plot(df.PHIND, df.index, ls='-', color='red')
    ax3.set_ylim(ztop,zbot)
    ax3.invert_yaxis()
    ax3.grid()
    ax3.locator_params(axis='x', nbins=3)
    ax3.set_xlabel("PHIND")
    ax3.set_yticklabels([])
    
    #Plot PE
    ax4 = plt.subplot(165)
    ax4.plot(df.PE, df.index, ls='-', color='black')
    ax4.set_ylim(ztop,zbot)
    ax4.invert_yaxis()
    ax4.grid()
    ax4.locator_params(axis='x', nbins=3)
    ax4.set_xlabel("PE")
    ax4.set_yticklabels([])
    
    ax5 = plt.subplot(166)
    im=ax5.imshow(cluster, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    ax5.set_xlabel("Facies")
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    
    fig.suptitle('Well: %s'%df['Well Name'].iloc[0], fontsize=14,y=0.94)