# src/train.py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np
import os

from data_loader import get_dataloader
from model import TransformerAutoencoder

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def find_best_f1_threshold(mse_errors, labels):
    """Validation 셋의 MSE 에러를 바탕으로 최적의 임계값과 F1 스코어를 탐색"""
    precisions, recalls, thresholds = precision_recall_curve(labels, mse_errors)
    # 0으로 나누기 방지
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx], thresholds[best_idx]

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() and config['env']['device'] == 'auto' else "cpu")
    print(f"Training on device: {device}")
    
    # 1. DataLoader 호출
    train_loader, scaler = get_dataloader(mode="train")
    # Validation 셋은 라벨(정상=0, 불량=1)을 만들기 위해 커스텀 로직이 약간 필요하지만, 
    # 여기서는 폴더 분할 방식을 가정한 로더를 불러옵니다.
    val_loader, _ = get_dataloader(mode="val", scaler=scaler) 
    
    # 임시로 Val 셋의 실제 라벨 생성 (Test01은 0, Test05는 1로 가정하여 매핑)
    # 실제 구현 시에는 data_loader.py에서 파일명에 'NG'가 있으면 1, 'OK'면 0을 반환하도록 수정해야 합니다.
    val_labels = np.array([0]*500 + [1]*500) # 예시용 더미 라벨
    
    # 2. 모델 및 학습 설정
    model = TransformerAutoencoder(
        input_dim=208, 
        window_size=config['data']['window_size']
    ).to(device)
    
    criterion = nn.MSELoss(reduction='none') # 샘플별 오차를 구하기 위해 none
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])
    
    best_val_f1 = 0.0
    best_threshold = 0.0
    patience_counter = 0
    os.makedirs("models", exist_ok=True)
    
    # 3. 학습 루프
    for epoch in range(config['train']['epochs']):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 4. Validation (F-score 평가)
        model.eval()
        val_mses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                # 시계열 차원과 센서 차원을 평균내어 샘플당 MSE 1개 추출
                mse = criterion(reconstructed, batch).mean(dim=[1, 2]).cpu().numpy()
                val_mses.extend(mse)
                
        # Val 데이터 길이와 라벨 길이를 맞춰서 최적 임계값 탐색
        val_mses = np.array(val_mses)[:len(val_labels)] 
        current_f1, current_thresh = find_best_f1_threshold(val_mses, val_labels[:len(val_mses)])
        
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}] - Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {current_f1:.4f} (Thresh: {current_thresh:.4f})")
        
        # 5. Checkpoint 저장 및 Early Stopping 로직
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_threshold = current_thresh
            torch.save(model.state_dict(), "models/best_model.pth")
            # 임계값도 함께 저장
            with open("models/threshold.txt", "w") as f:
                f.write(str(current_threshold))
            patience_counter = 0
            print("  --> 🌟 Best model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= config['train']['early_stopping_patience']:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train_model()