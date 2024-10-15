import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()  # 应用随机种子

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
            'attention_mask': encoding['attention_mask'].flatten()
        }

def predict(model, data_loader, device='cuda'):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_labels = torch.argmax(probs, dim=1)
            predictions.extend(pred_labels.cpu().numpy())

    return predictions

if __name__ == "__main__":
    # 加载数据
    df = pd.read_excel('/home/WenqiQiu/bilibili/bert_training/bert_training/emo-test-data.xlsx')
    texts = df['text'].dropna().values

    # 模型和tokenizer的路径
    local_model_directory = '/home/WenqiQiu/.cache/modelscope/hub/iic/nlp_roberta_backbone_large_std'
    tokenizer = BertTokenizer.from_pretrained(local_model_directory)

    # 准备数据集
    dataset = TextDataset(texts, tokenizer)
    data_loader = DataLoader(dataset, batch_size=16, sampler=SequentialSampler(dataset))

    # 加载模型
    model = BertForSequenceClassification.from_pretrained(local_model_directory, num_labels=5)
    model = model.to('cuda')

    # 进行预测
    predictions = predict(model, data_loader)

    # 映射预测结果到情绪标签
    sentiment_mapping = {0: '非常消极', 1: '消极', 2: '中立', 3: '积极', 4: '非常积极'}
    df['predicted_label'] = predictions
    df['predicted_sentiment'] = df['predicted_label'].map(sentiment_mapping)
    
    # 保存结果
    df.to_excel('/home/WenqiQiu/bilibili/bert_training/bert_training/result-emo-1014.xlsx', index=False)
    print("Classification complete. Results saved.")
