import matplotlib.pyplot as plt
import numpy as np

def plot_pca_clusters(X_pca, clusters, title='PCA Clusters', jitter_strength=0.02):
    """Visualize PCA clusters in 2D and 3D."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    X_pca_jittered = X_pca + np.random.normal(scale=jitter_strength * (X_pca.max(axis=0) - X_pca.min(axis=0)),
                                              size=X_pca.shape)

    axes[0].set_title('PCA Component 1 vs 2')
    axes[1].set_title('PCA Component 1 vs 3')
    axes[2].set_title('PCA Component 2 vs 3')

    for label in np.unique(clusters):
        axes[0].scatter(X_pca_jittered[clusters == label, 0], X_pca_jittered[clusters == label, 1],
                        label=f'Cluster {label}', s=100, alpha=0.6)
        axes[1].scatter(X_pca_jittered[clusters == label, 0], X_pca_jittered[clusters == label, 2],
                        label=f'Cluster {label}', s=100, alpha=0.6)
        axes[2].scatter(X_pca_jittered[clusters == label, 1], X_pca_jittered[clusters == label, 2],
                        label=f'Cluster {label}', s=100, alpha=0.6)

    for ax in axes:
        ax.set_aspect('auto')
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('PCA Component')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(conf_matrix):
    """Visualize confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(conf_matrix, cmap='Blues')
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(conf_matrix):
        color = 'white' if val > conf_matrix.max() / 2 else 'black'
        ax.text(j, i, f'{val}', ha='center', va='center', color=color)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()
