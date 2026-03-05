import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_baseline(noise_level):

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = AdamW(model.parameters(), lr=2e-5)
    

    

    inputs = tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    labels = torch.tensor(df['is_duplicate'].values) # Use the noisy label column
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(3):
        for batch in loader:
            ids, mask, lbls = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            
            outputs = model(ids, attention_mask=mask, labels=lbls)
            loss = outputs.loss # Standard Cross-Entropy
            
            loss.backward()
            optimizer.step()
        print(f"Baseline Noise {noise_level} - Epoch {epoch+1} Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), f'models/baseline_bert_{noise_level}.pt')

if __name__ == "__main__":
    for level in [0.0, 0.1, 0.2, 0.3]:
        train_baseline(level)