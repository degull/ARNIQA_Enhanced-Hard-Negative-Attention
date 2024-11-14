## KADID
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

 """


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

"""
import torch
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

"""
import torch
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