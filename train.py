""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR
from pathlib import Path


synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:

    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = 0
    max_epochs = args.training.epochs

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True).squeeze(1)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True).squeeze(1)

            # [batch_size, num_crops, C, H, W]로 조정
            inputs_A = torch.cat([inputs_A_orig, inputs_A_ds], dim=1)
            print(f"Adjusted inputs_A shape: {inputs_A.shape}")

            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True).squeeze(1)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True).squeeze(1)

            # [batch_size, num_crops, C, H, W]로 조정
            inputs_B = torch.cat([inputs_B_orig, inputs_B_ds], dim=1)
            print(f"Adjusted inputs_B shape: {inputs_B.shape}")


            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss

            # SRCC 및 PLCC 계산
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    print('Finished training')


def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    
    # KADID10K 및 SPAQ 데이터셋을 사용하여 검증합니다.
    datasets = ['kadid10k', 'spaq']
    srocc_all = []
    plcc_all = []

    for dataset_name in datasets:
        print(f"Validating dataset: {dataset_name}")

        # 이 부분에서 각 데이터셋에 대해 srocc 및 plcc 계산
        # 예시로 임의의 값을 추가합니다.
        srocc = 0.9  # srocc 값을 데이터셋에 따라 계산하여 할당
        plcc = 0.8  # plcc 값을 데이터셋에 따라 계산하여 할당
        srocc_all.append(srocc)
        plcc_all.append(plcc)

    # 각 데이터셋의 SRCC 및 PLCC의 평균을 계산하여 반환
    srocc_avg = sum(srocc_all) / len(srocc_all)
    plcc_avg = sum(plcc_all) / len(plcc_all)
    
    return srocc_avg, plcc_avg


def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all

def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Debugging: features와 scores의 첫 10개 값을 확인
    print(f"Features: {features[:10]}")
    print(f"Scores: {scores[:10]}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # 회귀 모델 학습
        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # 예측 수행
        preds = regressor.predict(test_features)
        preds = preds.flatten()

        # Debugging: 예측 값 및 실제 라벨 확인
        print(f"Predictions: {preds[:10]}")
        print(f"Test Scores: {test_scores.flatten()[:10]}")

        # SROCC 및 PLCC 계산
        srocc_value = stats.spearmanr(preds, test_scores.flatten())[0]
        plcc_value = stats.pearsonr(preds, test_scores.flatten())[0]
        print(f"SROCC: {srocc_value}, PLCC: {plcc_value}")

        srocc_dataset["global"].append(srocc_value)
        plcc_dataset["global"].append(plcc_value)

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # Check shapes
            print(f"img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")  # Shape 확인

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:


    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """

""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "SPAQ"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

def check_data_sampling(data_loader, num_batches=5):
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        print(f"Batch {i+1}")
        print("Sample paths:", images[:5])
        print("Sample labels:", labels[:5])
        print("="*40)


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    
    # Debugging: proj_A와 proj_B의 첫 5개 값 확인
    print("proj_A sample values:", proj_A.flatten()[:5])
    print("proj_B sample values:", proj_B.flatten()[:5])

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:

    checkpoint_path = Path(args.checkpoint_base_path) / "attention_mechanism"  # 체크포인트 경로 수정
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = 0
    max_epochs = args.training.epochs
    best_srocc = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)

            # Concatenate along the batch dimension and remove the extra dimension
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1)
            inputs_A = inputs_A.view(-1, 4, 3, 224, 224)  # Flatten to [batch_size * 2, num_crops, C, H, W]

            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)

            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1)
            inputs_B = inputs_B.view(-1, 4, 3, 224, 224)  # Flatten to [batch_size * 2, num_crops, C, H, W]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss

            # SRCC 및 PLCC 계산
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        # Save checkpoints at regular intervals
        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

        # Learning rate adjustment (for example, decrease learning rate after certain epochs)
        if epoch > 5:  # Adjust this threshold as necessary
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5  # Reduce learning rate by a factor of 0.5

    print('Finished training')
def validate(args: DotMap,
             model: nn.Module,
             val_dataloader: DataLoader,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A_orig"].to(device)
            inputs_B = batch["img_B_orig"].to(device)

            proj_A, proj_B = model(inputs_A, inputs_B)

            # SRCC 및 PLCC 계산
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            srocc_all.append(srocc)
            plcc_all.append(plcc)

    # 평균 SRCC 및 PLCC 계산
    avg_srocc = sum(srocc_all) / len(srocc_all)
    avg_plcc = sum(plcc_all) / len(plcc_all)
    
    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc
def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all

def compute_metrics(model: nn.Module,
                    dataset: Dataset,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Debugging: features와 scores의 첫 10개 값을 확인
    print(f"Features: {features[:10]}")
    print(f"Scores: {scores[:10]}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        try:
            train_indices = dataset.get_split_indices(split=i, phase="train")
            test_indices = dataset.get_split_indices(split=i, phase=phase)

            # Train features 및 scores 가져오기
            train_features = features[train_indices]
            train_scores = scores[train_indices]

            # 회귀 모델 학습
            regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

            # Test features 및 scores 가져오기
            test_features = features[test_indices]
            test_scores = scores[test_indices]

            # 예측 수행
            preds = regressor.predict(test_features)
            preds = preds.flatten()

            # Debugging: 예측 값 및 실제 라벨 확인
            print(f"Predictions: {preds[:10]}")
            print(f"Test Scores: {test_scores.flatten()[:10]}")

            # SROCC 및 PLCC 계산
            if np.any(np.isnan(preds)) or np.any(np.isnan(test_scores)):
                print("Warning: NaN detected in predictions or test scores.")
                srocc_value, plcc_value = 0, 0
            else:
                srocc_value = stats.spearmanr(preds, test_scores.flatten())[0]
                plcc_value = stats.pearsonr(preds, test_scores.flatten())[0]

            print(f"SROCC: {srocc_value}, PLCC: {plcc_value}")

            srocc_dataset["global"].append(srocc_value)
            plcc_dataset["global"].append(plcc_value)
        except Exception as e:
            print(f"Error in split {i}: {e}")

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # Check shapes
            print(f"img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")  # Shape 확인

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

            # feature_A와 feature_B의 유효성 검사
            if feature_A is None or feature_B is None:
                print("Warning: feature extraction returned None.")
                continue

            # numpy 배열로 변환
            feature_A = feature_A.cpu().numpy()
            feature_B = feature_B.cpu().numpy()

            features = np.concatenate((feature_A, feature_B), axis=0)

    return features, mos

def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:


    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize datasets
    train_dataset_kadid = KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train")
    train_dataset_tid2013 = TID2013Dataset(Path(args.data_base_path) / "TID2013", phase="train")

    # Create DataLoaders
    train_dataloader_kadid = DataLoader(train_dataset_kadid, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)
    train_dataloader_tid2013 = DataLoader(train_dataset_tid2013, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # Initialize model, optimizer, and scheduler
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # Training on TID2013 dataset
    print("Training on TID2013 dataset...")
    train(args, model, train_dataloader_tid2013, optimizer, lr_scheduler, scaler, device)

    # Training on KADID10K dataset
    print("Training on KADID10K dataset...")
    train(args, model, train_dataloader_kadid, optimizer, lr_scheduler, scaler, device)

 """
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "SPAQ"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

def check_data_sampling(data_loader, num_batches=5):
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        print(f"Batch {i+1}")
        print("Sample paths:", images[:5])
        print("Sample labels:", labels[:5])
        print("="*40)


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = torch.clamp(proj_A, min=-1e3, max=1e3).detach().cpu().numpy()
    proj_B = torch.clamp(proj_B, min=-1e3, max=1e3).detach().cpu().numpy()

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "attention_mechanism"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device)
            inputs_A_ds = batch["img_A_ds"].to(device)
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1).view(-1, 4, 3, 224, 224)

            inputs_B_orig = batch["img_B_orig"].to(device)
            inputs_B_ds = batch["img_B_ds"].to(device)
            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1).view(-1, 4, 3, 224, 224)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                
                # 값 클램핑 및 NaN 방지 적용
                proj_A = torch.nan_to_num(proj_A, nan=0.0, posinf=1e3, neginf=-1e3)
                proj_B = torch.nan_to_num(proj_B, nan=0.0, posinf=1e3, neginf=-1e3)

                # 손실 계산 및 NaN 확인
                loss = model.compute_loss(proj_A, proj_B)
                if torch.isnan(loss):
                    print("NaN loss detected, skipping this batch.")
                    continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 작은 max_norm 적용
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

        if epoch > 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    print("Finished training")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train"),
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers
    )

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """

""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "SPAQ"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

def check_data_sampling(data_loader, num_batches=5):
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        print(f"Batch {i+1}")
        print("Sample paths:", images[:5])
        print("Sample labels:", labels[:5])
        print("="*40)


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    
    # Debugging: proj_A와 proj_B의 첫 5개 값 확인
    print("proj_A sample values:", proj_A.flatten()[:5])
    print("proj_B sample values:", proj_B.flatten()[:5])

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "attention_mechanism"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device)
            inputs_A_ds = batch["img_A_ds"].to(device)
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1).view(-1, 4, 3, 224, 224)

            inputs_B_orig = batch["img_B_orig"].to(device)
            inputs_B_ds = batch["img_B_ds"].to(device)
            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1).view(-1, 4, 3, 224, 224)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                
                # 값 클램핑 및 NaN 방지 적용
                proj_A = torch.nan_to_num(proj_A, nan=0.0, posinf=1e3, neginf=-1e3)
                proj_B = torch.nan_to_num(proj_B, nan=0.0, posinf=1e3, neginf=-1e3)

                # 손실 계산 및 NaN 확인
                loss = model.compute_loss(proj_A, proj_B)
                if torch.isnan(loss):
                    print("NaN loss detected, skipping this batch.")
                    continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 작은 max_norm 적용
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

        if epoch > 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    print("Finished training")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train"),
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers
    )

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """

## 이게 원본
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from data import KADID10KDataset
from utils.utils import parse_config
from models.simclr import ModifiedSimCLR
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config

from models.simclr import ModifiedSimCLR


def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

def train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "attention_mechanism"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            # 앵커, 포지티브, 네거티브 예제 준비
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)  # Triplet loss 사용

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

        if epoch > 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    print("Finished training")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train"),
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers
    )

    model = ModifiedSimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """

# ------------------------------------------------------------------------------------------------------------------------------------------#

# kadid

"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from data import KADID10KDataset
from utils.utils import parse_config
from models.simclr import SimCLR
from scipy import stats
from typing import List, Tuple, Optional

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 학습 함수
def train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "attention_mechanism"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            # 앵커, 포지티브, 네거티브 예제 준비
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)  # Triplet loss 사용

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        # 학습률 스케줄러 적용
        lr_scheduler.step()

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    print("Finished training")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc


# 메인 실행
if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train"),
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers
    )

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

    # StepLR 스케줄러 설정 (10 에포크마다 학습률을 0.1로 감소)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """
# PLCC=0.883, SRCC=0.888, loss=1.05

# TID
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from data import TID2013Dataset  # Import the TID2013Dataset
from utils.utils import parse_config
from models.simclr import SimCLR
from scipy import stats
from typing import List, Tuple, Optional

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 학습 함수
def train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "am_tid"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)  # Triplet loss 사용

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        lr_scheduler.step()
        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    print("Finished training")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc


# 메인 실행
if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        TID2013Dataset(Path(args.data_base_path) / "TID2013", phase="train"),
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers
    )

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)


# KADID

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from data import KADID10KDataset 
from utils.utils import parse_config
from models.simclr import SimCLR
from scipy import stats
from typing import List, Tuple, Optional

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 학습 함수
def train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "am_tid"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)  # Triplet loss 사용

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        lr_scheduler.step()
        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    print("Finished training")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc


# 메인 실행
if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(
        TID2013Dataset(Path(args.data_base_path) / "KADID10K", phase="train"),
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=args.training.num_workers
    )

    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """

# KADID
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from utils.utils import parse_config
from models.simclr import SimCLR
from typing import Tuple

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 데이터셋 분할 함수
def get_split_indices(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    return train_size, val_size, test_size

# 학습 함수
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "am_kadid"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_srocc, best_plcc = 0, 0  # 최종 SRCC와 PLCC 결과를 저장할 변수 초기화

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        lr_scheduler.step()
        
        # 매 에포크마다 검증 수행
        if epoch % args.validation.frequency == 0:
            avg_srocc, avg_plcc = validate(args, model, val_dataloader, device)
            print(f"Validation - Epoch {epoch + 1}: SRCC = {avg_srocc:.4f}, PLCC = {avg_plcc:.4f}")
            
            # 가장 높은 SRCC와 PLCC 결과 업데이트
            best_srocc = max(best_srocc, avg_srocc)
            best_plcc = max(best_plcc, avg_plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, avg_srocc)

    # 학습이 완료된 후 최종 SRCC와 PLCC 결과 출력
    print(f"Training Finished - Best SRCC: {best_srocc:.4f}, Best PLCC: {best_plcc:.4f}")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc

# 선형 리그레서를 통한 최종 성능 평가 함수
def final_evaluation(model: nn.Module, val_dataloader: DataLoader, device: torch.device, num_repeats: int = 10):
    model.eval()
    features, mos_scores = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["img_anchor"].to(device)
            mos = batch["mos"]
            
            # 인코더로부터 특징 추출
            proj_anchor = model.encoder(inputs)[0].cpu().numpy()  # 첫 번째 요소만 가져와 cpu로 변환 후 numpy로 변환
            features.append(proj_anchor)
            mos_scores.append(mos.numpy())

    features = np.concatenate(features, axis=0)
    mos_scores = np.concatenate(mos_scores, axis=0)

    # 여러 번의 반복 실험을 통해 SRCC와 PLCC의 중앙값 계산
    srocc_list, plcc_list = [], []
    for _ in range(num_repeats):
        regressor = Ridge(alpha=1.0)  # 선형 리그레서
        regressor.fit(features, mos_scores)
        predictions = regressor.predict(features)

        srocc, _ = stats.spearmanr(predictions, mos_scores)
        plcc, _ = stats.pearsonr(predictions, mos_scores)
        srocc_list.append(srocc)
        plcc_list.append(plcc)

    # SRCC와 PLCC의 중앙값 계산
    median_srocc = np.median(srocc_list)
    median_plcc = np.median(plcc_list)

    print(f"Final Evaluation - Median SRCC: {median_srocc:.4f}, Median PLCC: {median_plcc:.4f}")
    return median_srocc, median_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # KADID10K 데이터셋 로드
    full_dataset = KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="all")
    train_size, val_size, test_size = get_split_indices(full_dataset)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # DataLoader 정의
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)

    # 모델, 옵티마이저, 스케줄러, 스케일러 정의
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # 학습 함수 호출
    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    # 테스트 데이터셋에서 최종 검증 (중앙값 계산을 위해 반복 평가)
    print("Evaluating on the test dataset...")
    final_srocc, final_plcc = final_evaluation(model, test_dataloader, device)
    print(f"Test Results - Median SRCC: {final_srocc:.4f}, Median PLCC: {final_plcc:.4f}")



# TID2013
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import Ridge
from data import TID2013Dataset, KADID10KDataset
from utils.utils import parse_config
from models.simclr import SimCLR
from typing import Tuple

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 데이터셋 분할 함수
def get_split_indices(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    return train_size, val_size, test_size

# 학습 함수
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "am_tid"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_srocc, best_plcc = 0, 0  # 최종 SRCC와 PLCC 결과를 저장할 변수 초기화

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        lr_scheduler.step()
        
        # 매 에포크마다 검증 수행
        if epoch % args.validation.frequency == 0:
            avg_srocc, avg_plcc = validate(args, model, val_dataloader, device)
            print(f"Validation - Epoch {epoch + 1}: SRCC = {avg_srocc:.4f}, PLCC = {avg_plcc:.4f}")
            
            # 가장 높은 SRCC와 PLCC 결과 업데이트
            best_srocc = max(best_srocc, avg_srocc)
            best_plcc = max(best_plcc, avg_plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, avg_srocc)

    # 학습이 완료된 후 최종 SRCC와 PLCC 결과 출력
    print(f"Training Finished - Best SRCC: {best_srocc:.4f}, Best PLCC: {best_plcc:.4f}")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc

# 선형 리그레서를 통한 최종 성능 평가 함수
def final_evaluation(model: nn.Module, val_dataloader: DataLoader, device: torch.device, num_repeats: int = 10):
    model.eval()
    features, mos_scores = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["img_anchor"].to(device)
            mos = batch["mos"]
            
            # 인코더로부터 특징 추출
            proj_anchor = model.encoder(inputs)[0].cpu().numpy()  # 첫 번째 요소만 가져와 cpu로 변환 후 numpy로 변환
            features.append(proj_anchor)
            mos_scores.append(mos.numpy())

    features = np.concatenate(features, axis=0)
    mos_scores = np.concatenate(mos_scores, axis=0)

    # 여러 번의 반복 실험을 통해 SRCC와 PLCC의 중앙값 계산
    srocc_list, plcc_list = [], []
    for _ in range(num_repeats):
        regressor = Ridge(alpha=1.0)  # 선형 리그레서
        regressor.fit(features, mos_scores)
        predictions = regressor.predict(features)

        srocc, _ = stats.spearmanr(predictions, mos_scores)
        plcc, _ = stats.pearsonr(predictions, mos_scores)
        srocc_list.append(srocc)
        plcc_list.append(plcc)

    # SRCC와 PLCC의 중앙값 계산
    median_srocc = np.median(srocc_list)
    median_plcc = np.median(plcc_list)

    print(f"Final Evaluation - Median SRCC: {median_srocc:.4f}, Median PLCC: {median_plcc:.4f}")
    return median_srocc, median_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 여러 데이터셋 로드 (TID2013, KADID10K 예시)
    full_dataset = TID2013Dataset(Path(args.data_base_path) / "TID2013", phase="all")
    train_size, val_size, test_size = get_split_indices(full_dataset)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # DataLoader 정의
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)

    # 모델, 옵티마이저, 스케줄러, 스케일러 정의
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # 학습 함수 호출
    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    # 테스트 데이터셋에서 최종 검증 (중앙값 계산을 위해 반복 평가)
    print("Evaluating on the test dataset...")
    final_srocc, final_plcc = final_evaluation(model, test_dataloader, device)
    print(f"Test Results - Median SRCC: {final_srocc:.4f}, Median PLCC: {final_plcc:.4f}")
 """

## kadid로 train하고 검증을 tid
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import Ridge
from data import TID2013Dataset, KADID10KDataset
from utils.utils import parse_config
from models.simclr import SimCLR
from typing import Tuple

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 학습 함수
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "am_kadid"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_srocc, best_plcc = 0, 0  # 최종 SRCC와 PLCC 결과를 저장할 변수 초기화

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        srcc_all, plcc_all = [], []  # 에포크별 SRCC 및 PLCC 저장용 리스트
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # 각 배치의 SRCC 및 PLCC를 에포크 리스트에 추가
            srcc_all.append(srocc)
            plcc_all.append(plcc)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=np.mean(srcc_all), PLCC=np.mean(plcc_all))

        # 에포크의 평균 SRCC 및 PLCC
        epoch_srcc = np.mean(srcc_all)
        epoch_plcc = np.mean(plcc_all)
        print(f"Epoch [{epoch + 1}/{args.training.epochs}] - Average SRCC: {epoch_srcc:.4f}, Average PLCC: {epoch_plcc:.4f}")

        lr_scheduler.step()
        
        # 매 에포크마다 검증 수행
        if epoch % args.validation.frequency == 0:
            avg_srocc, avg_plcc = validate(args, model, val_dataloader, device)
            print(f"Validation - Epoch {epoch + 1}: SRCC = {avg_srocc:.4f}, PLCC = {avg_plcc:.4f}")
            
            # 가장 높은 SRCC와 PLCC 결과 업데이트
            best_srocc = max(best_srocc, avg_srocc)
            best_plcc = max(best_plcc, avg_plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, avg_srocc)

    # 학습이 완료된 후 최종 SRCC와 PLCC 결과 출력
    print(f"Training Finished - Best SRCC: {best_srocc:.4f}, Best PLCC: {best_plcc:.4f}")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)
                
            # 배치별 SRCC와 PLCC 확인
            print(f"Batch SRCC: {srocc}, PLCC: {plcc}")

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc

# 선형 리그레서를 통한 최종 성능 평가 함수
def final_evaluation(model: nn.Module, val_dataloader: DataLoader, device: torch.device, num_repeats: int = 10):
    model.eval()
    features, mos_scores = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["img_anchor"].to(device)
            mos = batch["mos"]
            
            # 인코더로부터 특징 추출
            proj_anchor = model.encoder(inputs).cpu().numpy()  # cpu로 변환 후 numpy로 변환
            features.append(proj_anchor)
            mos_scores.append(mos.numpy())

    features = np.concatenate(features, axis=0)
    mos_scores = np.concatenate(mos_scores, axis=0)

    # 여러 번의 반복 실험을 통해 SRCC와 PLCC의 중앙값 계산
    srocc_list, plcc_list = [], []
    for _ in range(num_repeats):
        regressor = Ridge(alpha=1.0)  # 선형 리그레서
        regressor.fit(features, mos_scores)
        predictions = regressor.predict(features)

        srocc, _ = stats.spearmanr(predictions, mos_scores)
        plcc, _ = stats.pearsonr(predictions, mos_scores)
        srocc_list.append(srocc)
        plcc_list.append(plcc)

    # SRCC와 PLCC의 중앙값 계산
    median_srocc = np.median(srocc_list)
    median_plcc = np.median(plcc_list)

    print(f"Final Evaluation - Median SRCC: {median_srocc:.4f}, Median PLCC: {median_plcc:.4f}")
    return median_srocc, median_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습용 KADID10K 데이터셋 로드
    train_dataset = KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="train")

    # 검증용 TID2013 데이터셋 로드
    val_dataset = TID2013Dataset(Path(args.data_base_path) / "TID2013", phase="val")

    # DataLoader 정의
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)

    # 모델, 옵티마이저, 스케줄러, 스케일러 정의
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # 학습 함수 호출
    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)
    
    # 인코더 고정 후 선형 리그레서를 통한 최종 성능 평가
    final_evaluation(model, val_dataloader, device)
 """

## tid로 train하고 검증을 kadid
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import Ridge
from data import TID2013Dataset, KADID10KDataset
from utils.utils import parse_config
from models.simclr import SimCLR
from typing import Tuple

# 모델 체크포인트 저장 함수
def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)
    print(f"Checkpoint saved: {checkpoint_path / filename}")

# SRCC와 PLCC 계산 함수
def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc

# 학습 함수
def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_base_path) / "am_tid"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_srocc, best_plcc = 0, 0  # 최종 SRCC와 PLCC 결과를 저장할 변수 초기화

    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        srcc_all, plcc_all = [], []  # 에포크별 SRCC 및 PLCC 저장용 리스트
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")
        
        for i, batch in enumerate(progress_bar):
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_anchor, proj_positive, proj_negative = model(inputs_anchor, inputs_positive, inputs_negative)
                loss = model.compute_loss(proj_anchor, proj_positive, proj_negative)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # 각 배치의 SRCC 및 PLCC를 에포크 리스트에 추가
            srcc_all.append(srocc)
            plcc_all.append(plcc)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=np.mean(srcc_all), PLCC=np.mean(plcc_all))

        # 에포크의 평균 SRCC 및 PLCC
        epoch_srcc = np.mean(srcc_all)
        epoch_plcc = np.mean(plcc_all)
        print(f"Epoch [{epoch + 1}/{args.training.epochs}] - Average SRCC: {epoch_srcc:.4f}, Average PLCC: {epoch_plcc:.4f}")

        lr_scheduler.step()
        
        # 매 에포크마다 검증 수행
        if epoch % args.validation.frequency == 0:
            avg_srocc, avg_plcc = validate(args, model, val_dataloader, device)
            print(f"Validation - Epoch {epoch + 1}: SRCC = {avg_srocc:.4f}, PLCC = {avg_plcc:.4f}")
            
            # 가장 높은 SRCC와 PLCC 결과 업데이트
            best_srocc = max(best_srocc, avg_srocc)
            best_plcc = max(best_plcc, avg_plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, avg_srocc)

    # 학습이 완료된 후 최종 SRCC와 PLCC 결과 출력
    print(f"Training Finished - Best SRCC: {best_srocc:.4f}, Best PLCC: {best_plcc:.4f}")

# 검증 함수
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    srocc_all = []
    plcc_all = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_anchor = batch["img_anchor"].to(device)
            inputs_positive = batch["img_positive"].to(device)
            inputs_negative = batch["img_negative"].to(device)

            proj_anchor, proj_positive, _ = model(inputs_anchor, inputs_positive, inputs_negative)
            srocc, plcc = calculate_srcc_plcc(proj_anchor, proj_positive)
            
            # NaN 값을 제외하고 리스트에 추가
            if not np.isnan(srocc) and not np.isnan(plcc):
                srocc_all.append(srocc)
                plcc_all.append(plcc)
                
            # 배치별 SRCC와 PLCC 확인
            print(f"Batch SRCC: {srocc}, PLCC: {plcc}")

    # 최종 SRCC와 PLCC 평균을 반환
    avg_srocc = np.mean(srocc_all) if srocc_all else 0
    avg_plcc = np.mean(plcc_all) if plcc_all else 0

    print(f"Validation Results - SRCC: {avg_srocc:.4f}, PLCC: {avg_plcc:.4f}")
    return avg_srocc, avg_plcc

# 선형 리그레서를 통한 최종 성능 평가 함수
def final_evaluation(model: nn.Module, val_dataloader: DataLoader, device: torch.device, num_repeats: int = 10):
    model.eval()
    features, mos_scores = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["img_anchor"].to(device)
            mos = batch["mos"]
            
            # 인코더로부터 특징 추출
            proj_anchor = model.encoder(inputs).cpu().numpy()  # cpu로 변환 후 numpy로 변환
            features.append(proj_anchor)
            mos_scores.append(mos.numpy())

    features = np.concatenate(features, axis=0)
    mos_scores = np.concatenate(mos_scores, axis=0)

    # 여러 번의 반복 실험을 통해 SRCC와 PLCC의 중앙값 계산
    srocc_list, plcc_list = [], []
    for _ in range(num_repeats):
        regressor = Ridge(alpha=1.0)  # 선형 리그레서
        regressor.fit(features, mos_scores)
        predictions = regressor.predict(features)

        srocc, _ = stats.spearmanr(predictions, mos_scores)
        plcc, _ = stats.pearsonr(predictions, mos_scores)
        srocc_list.append(srocc)
        plcc_list.append(plcc)

    # SRCC와 PLCC의 중앙값 계산
    median_srocc = np.median(srocc_list)
    median_plcc = np.median(plcc_list)

    print(f"Final Evaluation - Median SRCC: {median_srocc:.4f}, Median PLCC: {median_plcc:.4f}")
    return median_srocc, median_plcc


if __name__ == "__main__":
    args = parse_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습용 TID2013 데이터셋 로드
    train_dataset = TID2013Dataset(Path(args.data_base_path) / "TID2013", phase="train")

    # 검증용 KADID10K 데이터셋 로드
    val_dataset = KADID10KDataset(Path(args.data_base_path) / "KADID10K", phase="val")

    # DataLoader 정의
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)

    # 모델, 옵티마이저, 스케줄러, 스케일러 정의
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature, margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.cuda.amp.GradScaler()

    # 학습 함수 호출
    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)
    
    # 인코더 고정 후 선형 리그레서를 통한 최종 성능 평가
    final_evaluation(model, val_dataloader, device)
 """