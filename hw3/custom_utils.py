# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Simple plotter for 3d function - from surface and contour perspective
    '''             

    # my cost history plotter
    def plot_cost(self,g,w_history):
        # make a figure
        fig,ax= plt.subplots(1,1,figsize = (8,3))

        # compute cost vales
        cost_vals = [g(w) for w in w_history]

        # plot the cost values
        ax.plot(cost_vals)

        # cleanup graph and label axes
        ax.set_xlabel('num of (outer loop) iterations')
        ax.set_ylabel('cost function value')

    def transform_least_squares_to_quadratic(self,x,y):
        '''
        This function takes in a regression dataset and outputs the constants of the 
        quadratic form of its associated Least Squares function: a,b, and C
        '''
        # make a local copy of the input data
        x_local = copy.deepcopy(x)

        # containers for our quadratic formula
        a = 0
        b = 0
        C = 0
        for p in range(len(y)):
            # get p^th point
            x_p = np.asarray(x_local[:,p])
            y_p = y[p]

            # form C
            x_p.shape = (np.size(x_p),1)
            C+= np.dot(x_p,x_p.T)

            # form b
            b -= 2*x_p*y_p

            # form a
            a += y_p**2

        return a,b,C

    def predict_and_plot(self,x,y,splits,levels,w):
        '''
        Make prediction given input stumps and plot result
        '''
        
        # create figure
        fig, ax = plt.subplots(1, 1, figsize=(4,3))
        
        # scatter data
        ax.scatter(x,y,color = 'k',edgecolor = 'w')

        # make input for prediction
        xmin = min(x)
        xmax = max(x)
        s = np.linspace(xmin,xmax,200)
        
        # loop over plotting input and evalute each point using stumps, take weighted combinatino
        t = []   # output container
        for pt in s:
            # initialize return value and add bias weight
            val = 0
            val += w[0]

            # loop over the stump collectionand calculate weighted contribution
            for u in range(len(splits)):
                # get current stump f_u
                split = splits[u]
                level = levels[u]

                ### our stump function f_u(x)
                if pt <= split:  # lies to the left - so evaluate at left level
                    val += w[u+1]*level[0]
                else:
                    val += w[u+1]*level[1]
             
            # store evaluation
            t.append(val)
       
        # plot prediction
        ax.plot(s,t,c = 'lime',linewidth = 2.5,zorder = 3)
        plt.show()
