import numpy as np

def plot_feature_importances(estimator, cols):
    import matplotlib.pyplot as plt
    y = estimator.feature_importances_
    #plot
    fig, ax = plt.subplots() 
    width = 0.4 # the width of the bars 
    ind = np.arange(len(y)) # the x locations for the groups
    ax.barh(ind, y, width, color="green")
    ax.set_yticks(ind+width/10)
    ax.set_yticklabels(cols, minor=False)
    plt.title("Feature importance in RandomForest Classifier")
    plt.xlabel("Relative importance")
    plt.ylabel("feature") 
    plt.figure(figsize=(5,5))
    fig.set_size_inches(6.5, 4.5, forward=True)

    plt.show()


def plot_trees(estimator, features):
    from sklearn.tree import export_graphviz
    export_graphviz(
        estimator, 
        out_file="tree.dot", # Need to make into jpeg...
        feature_names = features,
        class_names = ["Long", "Short"],
        rounded = True, 
        proportion = False, 
        precision = 2, 
        filled = True
    )
