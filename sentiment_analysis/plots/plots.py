import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies(accuracies, algorithms):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    
    y_pos = np.arange(len(algorithms))
    performance = accuracies
    
    
    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algorithms)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Accuracy')
    ax.set_title('Classifier Comparison')
    
    plt.show()