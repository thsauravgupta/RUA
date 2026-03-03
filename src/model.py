import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
class UncertaintyBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Predicts the Answer (Logits)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Predicts Uncertainty (Log Variance / Aleatoric)
        self.uncertainty = nn.Linear(config.hidden_size, 1)
        
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        # 1. Prediction Head
        logits = self.classifier(pooled_output)
        
        # 2. Uncertainty Head 
        log_var = self.uncertainty(pooled_output)
        
        loss = None
        if labels is not None:
            # Gaussian NLL Loss 
            loss_fct = CrossEntropyLoss(reduction='none')
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
    
            precision = torch.exp(-log_var).view(-1)
            
            # Loss = (Standard Loss * Precision) + Uncertainty Penalty
           
            loss = 0.5 * precision * ce_loss + 0.5 * log_var.view(-1)
            loss = loss.mean()
            
        return (loss, logits, log_var) if loss is not None else (logits, log_var)