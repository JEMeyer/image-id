import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_points(path):
    '''
    Input: 
      path: String. Path to csv file
    Returns:
      data: np.ndarray. Matrix of data
    '''
    data = np.loadtxt(path)
    return data

def plot_points(data):
    '''
    Input:
      data: np.ndarray. Matrix of data where each col is a dimension (x or y)
    Returns:
      Nothing
    '''
    numCols = data.shape[1]
    x, y1, y2 = np.split(data, numCols, axis=1)

    plt.plot(x,y1)
    plt.plot(x,y2)

    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('% Accuracy', fontsize=14)
    plt.title('Cool Title', fontsize=18)

    plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    data = load_points(path)
    plot_points(data)
