import torch
import torch.nn.functional as F
from src.model import RUABert
from transformers import BertTokenizer, AdamW
import pandas as pd

def robust_gnll_loss(mu, s, target):
    """
    Implementation of Variance Attenuation.
    mu: Predicted class scores
    s: Predicted log-variance
    """
    # Convert targets to one-hot for the residual calculation
    target_one_hot = F.one_hot(target, num_classes=mu.size(1)).float()
    
    # Core RUA-BERT Loss: 0.5 * exp(-s) * (residual^2) + 0.5 * s
    precision = torch.exp(-s)
    residual_sq = (target_one_hot - mu)**2
    loss = 0.5 * precision * residual_sq + 0.5 * s
    return loss.mean()

def train_rua(noise_level):
    model = RUABert().to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    df = pd.read_csv(f'data/processed/questions_noisy_{noise_level}.csv')
    

    model.train()
    for epoch in range(5): # RUA-BERT often needs slightly more epochs to calibrate variance
        for batch in loader:
            ids, mask, lbls = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            
            mu, s = model(ids, mask)
            loss = robust_gnll_loss(mu, s, lbls)
            
            loss.backward()
            optimizer.step()
        print(f"RUA-BERT Noise {noise_level} - Epoch {epoch+1} Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), f'models/rua_bert_{noise_level}.pt')