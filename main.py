import os
import copy
import random
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from data_utils import CodonTokenizer, HybridDataset
from model_moe import AttnFusionMoE

BASE_PATH = ''
CONFIG = {
    'seed': 4066,
    'batch_size': 64,
    'accum_steps': 2,
    'bert_dim': 768,
    'bio_dim': 67,
    'd_model': 768,
    'n_experts': 16,
    'top_k': 4,
    'k_visiting': 8,
    'dropout': 0.3,
    'aux_loss_weight': 0.05,
    'rank_loss_weight': 0.2,
    'initial_lr': 1e-4,
    'max_epochs': 30,
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    'mouse_file': os.path.join(BASE_PATH, 'mouse.xlsx'),
    'human_file': os.path.join(BASE_PATH, 'human.xlsx'),
    'len_u5': 200, 'len_cds': 2200, 'len_u3': 2200
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger()



class RankLoss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred, y_true = y_pred.view(-1, 1), y_true.view(-1, 1)
        pred_diff = y_pred - y_pred.t()
        true_diff = y_true - y_true.t()
        mask = (torch.abs(true_diff) > 1e-4).float()
        loss = torch.relu(-torch.sign(true_diff) * pred_diff) * mask
        return loss.sum() / (mask.sum() + 1e-9)


def compute_aux_loss(gate_weights, n_experts):
    density = gate_weights.mean(dim=0)
    return torch.sum(density * density) * n_experts


def calculate_metrics(y_true_norm, y_pred_norm, mean, std):
    y_true = np.array(y_true_norm) * std + mean
    y_pred = np.array(y_pred_norm) * std + mean
    if len(y_true) < 2: return 0, 0, 0, 0
    return r2_score(y_true, y_pred), pearsonr(y_true, y_pred)[0], spearmanr(y_true, y_pred)[0], mean_squared_error(
        y_true, y_pred)


def train_one_epoch(model, loader, optimizer, criterion, device, accum_steps, epoch_idx, desc, scheduler=None,
                    rank_loss_fn=None):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{desc} Ep {epoch_idx}", leave=False)
    for i, (u5, cds, u3, bert_x, bio_x, targets) in pbar:
        u5, cds, u3, bert_x, bio_x, targets = u5.to(device), cds.to(device), u3.to(device), bert_x.to(device), bio_x.to(
            device), targets.to(device)
        outputs, gate_weights, _ = model(u5, cds, u3, bert_x, bio_x)
        loss = criterion(outputs, targets) + CONFIG['aux_loss_weight'] * compute_aux_loss(gate_weights,
                                                                                          CONFIG['n_experts'])
        if rank_loss_fn and "P3" in desc:
            loss += CONFIG['rank_loss_weight'] * rank_loss_fn(outputs, targets)
        loss = loss / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step();
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
    if scheduler: scheduler.step()
    return total_loss / len(loader)


def validate(model, loader, criterion, device, epoch_idx, desc):
    model.eval()
    preds, labels = [], []
    ds = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    with torch.no_grad():
        for u5, cds, u3, bert_x, bio_x, targets in loader:
            outputs, _, _ = model(u5.to(device), cds.to(device), u3.to(device), bert_x.to(device), bio_x.to(device))
            preds.extend(outputs.cpu().numpy());
            labels.extend(targets.cpu().numpy())
    r2, pcc, scc, mse = calculate_metrics(labels, preds, ds.mean, ds.std)
    logger.info(f"[{desc} Ep{epoch_idx}] R2:{r2:.4f} | PCC:{pcc:.4f} | SCC:{scc:.4f} | MSE:{mse:.4f}")
    return r2


def main():
    seed_everything(CONFIG['seed'])
    tokenizer = CodonTokenizer(CONFIG['len_u5'], CONFIG['len_cds'], CONFIG['len_u3'])
    ds_mouse = HybridDataset(CONFIG['mouse_file'], os.path.join(BASE_PATH, 'mouse_bert_feats.npy'), tokenizer, "Mouse")
    ds_human = HybridDataset(CONFIG['human_file'], os.path.join(BASE_PATH, 'human_bert_feats.npy'), tokenizer, "Human")

    m_train_idx, m_val_idx = train_test_split(np.arange(len(ds_mouse)), test_size=0.1, random_state=CONFIG['seed'])
    m_train_loader = DataLoader(Subset(ds_mouse, m_train_idx), batch_size=CONFIG['batch_size'], shuffle=True)
    m_val_loader = DataLoader(Subset(ds_mouse, m_val_idx), batch_size=CONFIG['batch_size'], shuffle=False)

    h_train_idx, h_test_idx = train_test_split(np.arange(len(ds_human)), test_size=0.2, random_state=CONFIG['seed'])
    h_train_loader = DataLoader(Subset(ds_human, h_train_idx), batch_size=CONFIG['batch_size'], shuffle=True)
    h_test_loader = DataLoader(Subset(ds_human, h_test_idx), batch_size=CONFIG['batch_size'], shuffle=False)

    model = AttnFusionMoE(tokenizer.vocab_size, CONFIG['bert_dim'], CONFIG['bio_dim'], CONFIG['d_model'],
                          CONFIG['n_experts'], CONFIG['top_k'], CONFIG['dropout']).to(CONFIG['device'])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['initial_lr'])
    criterion = nn.SmoothL1Loss()
    best_r2 = -float('inf')
    for epoch in range(CONFIG['max_epochs']):
        train_one_epoch(model, m_train_loader, optimizer, criterion, CONFIG['device'], CONFIG['accum_steps'], epoch + 1,
                        "P1 Mouse")
        r2 = validate(model, m_val_loader, criterion, CONFIG['device'], epoch + 1, "P1 Val")
        if r2 > best_r2:
            best_r2 = r2
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    logger.info(f"Phase 1 Done. Best R2: {best_r2:.4f}")


if __name__ == "__main__":
    main()