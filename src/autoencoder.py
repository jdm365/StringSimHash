import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np

from typing import List


class VAE(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(VAE, self).__init__()

        # Initializing the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.decoder = nn.Sequential(
            nn.Linear(768, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 768)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = T.exp(0.5 * log_var) 
        eps = T.randn_like(std)
        return mu + eps * std

    def forward(self, text):
        input_ids = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
                )['input_ids'].to(self.device)
        outputs = self.encoder(input_ids=input_ids)

        # Let's suppose that the mean and log variance are the first and second half of the encoder output
        # You might want to use a more sophisticated way to compute these
        mu = outputs.last_hidden_state[:, :384]
        log_var = outputs.last_hidden_state[:, 384:]

        z = self.reparameterize(mu, log_var)
        
        reconstructed = self.decoder(z)

        return reconstructed, outputs.last_hidden_state


def train(items: List[str], model_name='bert-base-uncased'):
    model = VAE(model_name)
    loss_fn = nn.MSELoss()

    n_epochs = 10

    ## Torch Dataset
    dataset    = VAEDataset(items)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    progress_bar = tqdm(total=len(dataloader) * n_epochs, desc='Batches')
    losses = []
    for _ in range(n_epochs):
        for _, batch in enumerate(dataloader):
            input_ids = batch['text'].to(model.device)
            target = input_ids

            reconstructed,  = model(input_ids)

            loss = loss_fn(reconstructed, target)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            losses.append(loss.item())

            progress_bar.update(1)
            progress_bar.set_description(f'Loss: {np.mean(losses[-100:]):.4f}')


class VAEDataset(Dataset):
    def __init__(self, items: List[str]):
        super(VAEDataset, self).__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {'text': self.items[idx]}



if __name__ == '__main__':
    FILENAME = 'data/corrupted_companies_dedup.feather'
    data = pd.read_feather(FILENAME)['company'].tolist()

    train(data)
