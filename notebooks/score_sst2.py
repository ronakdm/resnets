import torch
import open_clip
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler

import pickle
import os
import time

import faiss
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

DATASET = "sst2"
DATA_PATH = f"/mnt/ssd/ronak/datasets/{DATASET}"
DEVICE = 'cuda:0'
MODEL_NAME = f"gpt2"
SEED = 11182023

train = load_dataset('sst2', split='train', cache_dir=DATA_PATH)

def get_param_shapes(parameters):
    return [torch.tensor(p.shape) for p in parameters]

def get_num_parameters(param_shapes):
    return torch.tensor([torch.prod(s) for s in param_shapes]).sum().item()

def flatten_gradient(grads):
    return torch.cat([p.reshape(-1) for p in grads])

print("loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
print(f"number of parameters in foundation model {MODEL_NAME}: {get_num_parameters(get_param_shapes(model.parameters()))}")

print("tokenizing text...")
texts = []
for sent in tqdm(train['sentence']):
    encoded_input = tokenizer(sent, return_tensors='pt')
    texts.append(encoded_input['input_ids'].to(DEVICE))


# generate weirstrass matrix
print("generating random projection matrix...")
n_components = 12
n_parameters = get_num_parameters(get_param_shapes(model.parameters()))
np.random.seed(123)
tic = time.time()
rand_project = np.random.normal(size=(n_components, n_parameters)).astype(np.float16) / np.sqrt(n_components)
rand_project = torch.from_numpy(rand_project).float().to(DEVICE)
toc = time.time()
print(f"time taken to generate random matrix 0f size {(n_components, n_parameters)} gradient projections: {toc - tic}")

n_samples = len(texts)
tic = time.time()
with torch.no_grad():
    grads = []
    for i, text in tqdm(enumerate(texts)):
        if i >= n_samples:
            break
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(input_ids=text, labels=text, output_hidden_states=False, output_attentions=False)
            grads.append(torch.matmul(rand_project, flatten_gradient(torch.autograd.grad(outputs=output.loss, inputs=model.parameters())).detach()).cpu().numpy())
toc = time.time()
print(f"time taken to collect {n_samples} gradient projections: {toc - tic}")

grads = np.stack(grads)
print(grads.shape)
np.save(os.path.join(DATA_PATH, f"{MODEL_NAME}_scores.npy"), grads)