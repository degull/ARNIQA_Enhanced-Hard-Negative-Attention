from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_embeddings(model, dataloader, device):
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["img_anchor"].to(device)
            proj_anchor, _, _ = model(inputs, inputs, inputs)  # 임베딩만 사용
            embeddings.append(proj_anchor.cpu().numpy())
            labels.extend(batch["label"].numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    # t-SNE 적용
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 시각화
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE Visualization of Embeddings")
    plt.show()
