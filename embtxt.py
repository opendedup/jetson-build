import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]
print(embeddings.detach().numpy().tolist()[0])