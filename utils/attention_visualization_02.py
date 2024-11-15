from sklearn.manifold import TSNE
import plotly.graph_objects as go
import torch
import numpy as np
from umap import UMAP

def visualize_model_comparison_tsne_umap(model1, model2, dataloader, tsne_args, umap_args, device):
    """
    모델1과 모델2의 특징을 t-SNE 및 UMAP을 통해 시각화하고, 왜곡 유형과 MOS에 따른 색상 코드를 적용하여 비교합니다.
    
    Args:
        model1: 기존 ARNIQA 모델
        model2: 어텐션 메커니즘이 추가된 모델
        dataloader: 데이터 로더
        tsne_args: t-SNE 파라미터
        umap_args: UMAP 파라미터
        device: 장치 (GPU 또는 CPU)

    Returns:
        dict: 두 모델의 비교 시각화를 포함하는 Plotly 그래프 객체
    """
    print("모델 비교 시각화 생성 중...")

    methods = ["T-SNE", "UMAP"]

    # 왜곡 유형에 따른 색상 및 투명도 설정
    dist_color_maps = {"blur": "0, 80, 239",
                       "color_distortion": "227, 200, 0",
                       "jpeg": "216, 9, 168",
                       "noise": "245, 0, 56",
                       "brightness_change": "0, 204, 204",
                       "spatial_distortion": "160, 82, 45",
                       "sharpness_contrast": "96, 169, 23"}

    dist_shades = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

    mos_color_map = "138, 10, 10"
    mos_shades = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

    # 두 모델의 특징 추출 및 왜곡 레이블 생성
    def extract_features(model):
        features = torch.zeros((0, model.encoder.feat_dim))
        dist_colors = []
        mos_colors = []

        for batch in dataloader:
            img = batch["img"].to(device)
            img = img[:, 0]  # 중심 crop만 사용
            dist_group = batch["dist_group"]
            dist_level = batch["dist_level"]
            mos_score = batch["mos"]

            with torch.no_grad():
                output_features, _ = model(img)
            features = torch.cat((features, output_features.float().cpu()), 0)

            # 색상 및 투명도 설정
            for d_group, d_level, mos in zip(dist_group, dist_level, mos_score):
                mos_colors.append(f"rgba({mos_color_map}, {mos_shades[round(mos.item())]})")
                marker_color = dist_color_maps[d_group]
                marker_shade = dist_shades[d_level.item()]
                dist_colors.append(f"rgba({marker_color}, {marker_shade})")

        return features.numpy(), np.array(dist_colors), np.array(mos_colors)

    # 모델1 (기존 ARNIQA) 및 모델2 (어텐션 추가) 특징 추출
    features1, dist_colors1, mos_colors1 = extract_features(model1)
    features2, dist_colors2, mos_colors2 = extract_features(model2)

    figures = {}
    for method in methods:
        args = tsne_args if method == "T-SNE" else umap_args

        if method == "UMAP":
            features1_embedded = UMAP(**args).fit_transform(features1)
            features2_embedded = UMAP(**args).fit_transform(features2)
        elif method == "T-SNE":
            features1_embedded = TSNE(**args).fit_transform(features1)
            features2_embedded = TSNE(**args).fit_transform(features2)

        # 왜곡 시각화
        fig = go.Figure()

        # 모델1 (기존 ARNIQA) 시각화
        fig.add_trace(go.Scatter(x=features1_embedded[:, 0], y=features1_embedded[:, 1],
                                 mode='markers', marker=dict(size=5, color=dist_colors1), 
                                 name="기존 ARNIQA"))

        # 모델2 (어텐션 추가) 시각화
        fig.add_trace(go.Scatter(x=features2_embedded[:, 0], y=features2_embedded[:, 1],
                                 mode='markers', marker=dict(size=5, color=dist_colors2, symbol="triangle-up"),
                                 name="어텐션 추가 모델"))

        fig.update_layout(title=f'{method} - 모델 비교 (왜곡 유형)', xaxis_title="X", yaxis_title="Y")

        figures[f"{method}_Distortion_Comparison"] = fig

        # MOS 시각화
        fig = go.Figure()

        # 모델1 (기존 ARNIQA) 시각화
        fig.add_trace(go.Scatter(x=features1_embedded[:, 0], y=features1_embedded[:, 1],
                                 mode='markers', marker=dict(size=5, color=mos_colors1), 
                                 name="기존 ARNIQA"))

        # 모델2 (어텐션 추가) 시각화
        fig.add_trace(go.Scatter(x=features2_embedded[:, 0], y=features2_embedded[:, 1],
                                 mode='markers', marker=dict(size=5, color=mos_colors2, symbol="triangle-up"),
                                 name="어텐션 추가 모델"))

        fig.update_layout(title=f'{method} - 모델 비교 (MOS)', xaxis_title="X", yaxis_title="Y")

        figures[f"{method}_MOS_Comparison"] = fig

    return figures
