from sklearn.manifold import TSNE
import plotly.graph_objects as go
import torch
import torch.nn as nn
import numpy as np
from numba.core.errors import NumbaDeprecationWarning
import warnings
from umap import UMAP

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)


def visualize_comparison_tsne_umap(models: dict, dataloader: torch.utils.data.DataLoader, tsne_args: dict, umap_args: dict,
                                   device: torch.device) -> dict[str, go.Figure]:
    """
    Visualize the features extracted by two models (e.g., ARNIQA model with and without attention) using t-SNE and UMAP
    with the corresponding distortion levels and MOS scores. Supports only the KADID10K dataset.

    Args:
        models (dict): Dictionary with keys as model names and values as models (e.g., {"ARNIQA": model1, "ARNIQA+Attention": model2}).
        dataloader (torch.utils.data.DataLoader): the data loader to use
        tsne_args (dict): the arguments for t-SNE
        umap_args (dict): the arguments for UMAP
        device (torch.device): the device to use for training

    Returns:
        dict: Figures with keys indicating model name, visualization method, and type (e.g., "ARNIQA_T-SNE", "ARNIQA+Attention_UMAP").
    """
    print("Generating visualizations for comparison...")

    methods = ["T-SNE", "UMAP"]
    dist_color_maps = {
        "blur": "0, 80, 239",
        "color_distortion": "227, 200, 0",
        "jpeg": "216, 9, 168",
        "noise": "245, 0, 56",
        "brightness_change": "0, 204, 204",
        "spatial_distortion": "160, 82, 45",
        "sharpness_contrast": "96, 169, 23"
    }
    dist_shades = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
    mos_color_map = "138, 10, 10"
    mos_shades = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

    figures = {}
    for model_name, model in models.items():
        features = torch.zeros((0, model.encoder.feat_dim))
        dist_colors = []
        mos_colors = []

        for i, batch in enumerate(dataloader, 0):
            img = batch["img"].to(device=device, non_blocking=True)
            img = img[:, 0]  # Consider only the center crop
            dist_group = batch["dist_group"]
            dist_level = batch["dist_level"]
            mos_score = batch["mos"]

            with torch.cuda.amp.autocast(), torch.no_grad():
                output_features, _ = model(img)
            features = torch.cat((features, output_features.float().detach().cpu()), 0)

            for d_group, d_level, mos in zip(dist_group, dist_level, mos_score):
                mos_colors.append(f"rgba({mos_color_map}, {mos_shades[round(mos.item())]})")
                marker_color = dist_color_maps[d_group]
                marker_shade = dist_shades[d_level.item()]
                dist_colors.append(f"rgba({marker_color}, {marker_shade})")

        dist_colors = np.array(dist_colors)
        mos_colors = np.array(mos_colors)
        features = features.numpy()

        for method in methods:
            args = tsne_args if method == "T-SNE" else umap_args
            if method == "UMAP":
                features_embedded = UMAP(**args).fit_transform(features)
            elif method == "T-SNE":
                features_embedded = TSNE(**args).fit_transform(features)
            else:
                raise NotImplementedError(f"Method {method} not implemented")

            # Distortion visualization
            fig = go.Figure()
            for d_group, color in dist_color_maps.items():
                placeholder = features_embedded[np.where(dist_colors == f"rgba({color}, 1.0)")[0][0]]
                if args["n_components"] == 2:
                    trace = go.Scatter(x=[placeholder[0]], y=[placeholder[1]], mode='markers',
                                       marker=dict(size=5, color=f"rgba({color}, 1.0)"),
                                       name=d_group)
                elif args["n_components"] == 3:
                    trace = go.Scatter3d(x=[placeholder[0]], y=[placeholder[1]], z=[placeholder[2]], mode='markers',
                                         marker=dict(size=5, color=f"rgba({color}, 1.0)"),
                                         name=d_group)
                else:
                    raise ValueError(f"n_components parameter must be in [2, 3].")
                fig.add_trace(trace)

            if args["n_components"] == 2:
                fig.add_trace(go.Scatter(x=features_embedded[:, 0], y=features_embedded[:, 1],
                                         mode='markers', marker=dict(size=5, color=dist_colors), showlegend=False))
            elif args["n_components"] == 3:
                fig.add_trace(go.Scatter3d(x=features_embedded[:, 0], y=features_embedded[:, 1], z=features_embedded[:, 2],
                                           mode='markers', marker=dict(size=5, color=dist_colors), showlegend=False))

            fig.update_layout(title=f'{method} {model_name} - Distortion visualization',
                              scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                              legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
            figures[f"{model_name}_{method}_Distortion"] = fig

            # MOS visualization
            fig = go.Figure()
            for mos, opacity in mos_shades.items():
                placeholder = features_embedded[np.where(mos_colors == f"rgba({mos_color_map}, {opacity})")[0][0]]
                if args["n_components"] == 2:
                    trace = go.Scatter(x=[placeholder[0]], y=[placeholder[1]], mode='markers',
                                       marker=dict(size=5, color=f"rgba({mos_color_map}, {opacity})"),
                                       name=mos)
                elif args["n_components"] == 3:
                    trace = go.Scatter3d(x=[placeholder[0]], y=[placeholder[1]], z=[placeholder[2]], mode='markers',
                                         marker=dict(size=5, color=f"rgba({mos_color_map}, {opacity})"),
                                         name=mos)
                else:
                    raise ValueError(f"n_components parameter must be in [2, 3].")
                fig.add_trace(trace)

            if args["n_components"] == 2:
                fig.add_trace(go.Scatter(x=features_embedded[:, 0], y=features_embedded[:, 1],
                                         mode='markers', marker=dict(size=5, color=mos_colors), showlegend=False))
            elif args["n_components"] == 3:
                fig.add_trace(go.Scatter3d(x=features_embedded[:, 0], y=features_embedded[:, 1], z=features_embedded[:, 2],
                                           mode='markers', marker=dict(size=5, color=mos_colors), showlegend=False))

            fig.update_layout(title=f'{method} {model_name} - MOS visualization',
                              scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                              legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
            figures[f"{model_name}_{method}_MOS"] = fig

    return figures

""" 
두 모델의 특징을 시각화하여 각 모델의 성능을 시각적으로 비교

 """