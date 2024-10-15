import os
import torch
import torch.nn as nn
import gc
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold  # Use StratifiedKFold for binary classification
import torch_optimizer as optim
from sklearn.metrics import accuracy_score, f1_score

def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Ensure the label is either 0 or 1
        label = 1.0 if label > 0 else 0.0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float).unsqueeze(0)  # Add extra dimension
        }

def evaluate_metrics(preds, labels):
    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    binary_preds = (preds > 0.5).astype(int)
    labels = labels.astype(int)  # Ensure labels are also integers

    # Accuracy: Proportion of correct predictions for all labels
    accuracy = accuracy_score(labels, binary_preds)

    # Exact Match Accuracy (Accuracy All): Whether all predictions are correct
    correct_all = (binary_preds == labels).all(axis=1).astype(int)
    accuracy_all = correct_all.mean()

    # Multi-label Accuracy (IoU): Intersection over Union for all labels
    intersection = (binary_preds & labels).sum(axis=1)
    union = (binary_preds | labels).sum(axis=1)
    multi_label_accuracy = (intersection / (union + 1e-7)).mean()  # Add small epsilon to avoid division by zero

    # Label-wise Accuracy: Accuracy per individual label across all data points
    label_accuracy = (binary_preds == labels).mean(axis=0)

    return accuracy, accuracy_all, multi_label_accuracy, label_accuracy

def train_and_validate(model, train_loader, val_loader, criterion, device, optimizer, scheduler, results, num_epochs, fold):
    best_val_loss = float('inf')
    best_model_path = f'./best_model_fold_{fold}.pt'

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)

            probs = torch.sigmoid(outputs.logits)
            loss = criterion(probs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(inputs, attention_mask=attention_mask)

                probs = torch.sigmoid(outputs.logits)
                loss = criterion(probs, labels.float())
                total_val_loss += loss.item()

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_val_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Evaluate the metrics
        accuracy, accuracy_all, multi_label_accuracy, label_accuracy = evaluate_metrics(all_preds, all_labels)

        # Print logs
        print(f"Fold {fold}, Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, Accuracy All: {accuracy_all:.4f}, "
              f"Multi-label Accuracy: {multi_label_accuracy:.4f}, Label Accuracy: {label_accuracy}")

        # Save results
        results.append({
            'Fold': fold, 'Epoch': epoch, 'Train Loss': train_loss, 'Val Loss': val_loss,
            'Accuracy': accuracy, 'Accuracy All': accuracy_all, 'Multi-label Accuracy': multi_label_accuracy,
            'Label Accuracy': label_accuracy
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    return results

if __name__ == "__main__":
    # Set random seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load the new sentiment data
    df = pd.read_csv('/home/WenqiQiu/bilibili/bert_training/bert_training/1500合并.csv')
    
    # Drop rows with missing values in 'comment' or 'emotion_label'
    df = df.dropna(subset=['comment', 'emotion_label'])

    # Replace NaNs in labels with 0 or handle them
    labels = df['emotion_label'].values.astype(np.float32)  # Ensure labels are binary (0/1)
    texts = df['comment'].values

    # Tokenizer and model setup
    local_model_directory = '/home/WenqiQiu/.cache/modelscope/hub/iic/nlp_roberta_backbone_large_std'
    tokenizer = BertTokenizer.from_pretrained(local_model_directory)

    # K-Fold Cross-Validation setup
    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n========== Starting Fold {fold}/{num_folds} ==========")

        train_texts, val_texts = texts[train_idx], texts[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=128)

        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))

        model = BertForSequenceClassification.from_pretrained(local_model_directory, num_labels=1)  # Binary classification
        model = model.to('cuda')

        optimizer = optim.RAdam(model.parameters(), lr=1e-5, weight_decay=1e-3)
        criterion = nn.BCELoss()
 
        scheduler = get_scheduler(
            "cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=len(train_loader) * 20
        )

        # Train and validate for each fold
        results = train_and_validate(model, train_loader, val_loader, criterion, 'cuda', optimizer, scheduler, results, num_epochs=20, fold=fold)

        clean_memory()

    results_df = pd.DataFrame(results)
    results_df.to_csv('final_results_sentiment_kfold.csv', index=False)
    print("\nK-Fold Cross Validation Complete!")
