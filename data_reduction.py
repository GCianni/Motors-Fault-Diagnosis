from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def pca_components(data, component_percent, graph: bool):
    pca = PCA(n_components=component_percent)
    pca.fit(data)
    if graph:
        show_graph(data, component_percent, False)
    return pca.transform(data)


def show_graph(data, component_percent: float, save: bool):
    pca = PCA().fit(data)
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    xi = np.arange(1, len(data.columns) + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.ylim(0.0, 1.1)

    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(1, len(data.columns)+1, step=1))

    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=component_percent, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)
    ax.grid(axis='x')

    plt.show()
    if save:
        fig.savefig(r'C:\\Git\\Motor Fault Detection\\Teste_Data\\saved_jpg\\img.png')
    plt.close()
    return
