import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embeddings(embeddings):
    # Perform dimensionality reduction using t-SNE
    embeddings = torch.tensor(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot the embeddings
    x = embeddings_tsne[:, 0]
    y = embeddings_tsne[:, 1]
    plt.scatter(x, y)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Embeddings Visualization')
    plt.show()
