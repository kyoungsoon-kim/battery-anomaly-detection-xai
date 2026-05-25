# src/model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """시계열 데이터의 순서 정보를 Transformer에 주입하기 위한 위치 인코딩"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
    """PyTorch Native Transformer를 활용한 센서 이상탐지 모델"""
    def __init__(self, input_dim=208, window_size=30, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        
        # 1. Input Embedding (208개 센서 -> d_model 차원 확대)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)
        
        # 2. Transformer Encoder (HuggingFace DistilBert 대체)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True # (batch, seq, feature) 형태 유지
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Decoder / Reconstruction Layer (Latent -> 원본 208개 센서 복원)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, input_dim)
        )

    def forward(self, x):
        # x: (batch_size, window_size, 208)
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        
        latent = self.transformer_encoder(x_emb)
        reconstruction = self.decoder(latent)
        
        return reconstruction # 원본 데이터와 형태 동일 (batch, window, 208)