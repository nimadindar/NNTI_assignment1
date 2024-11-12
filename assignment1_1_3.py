import matplotlib.pyplot as plt

def plot(array1, array2):
    x1 = array1[:, 0]  
    y1 = array1[:, 1]
    
    x2 = array2[:, 0] 
    y2 = array2[:, 1] 

    plt.scatter(x1, y1, color='blue', label='Cluster 1')
    plt.scatter(x2, y2, color='red', label='Cluster 2')

    plt.title("Scatter plot of two sets of data points")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.legend() 

    plt.show()