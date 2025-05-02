"""
Train a two-tower retrieval model and save the best model.
"""

INPUT_DIM_QUERY = 32
INPUT_DIM_PRODUCT = 160

HIDDEN_DIM_PRODUCT = 64
HIDDEN_DIM_QUERY = 64

EMBEDDING_DIM = 32
LR = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_SPLITS = 5
MODEL_DIR = 'models/two_towers'




import os
import time
import torch
from torch import nn, autocast
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold
from recsys.recall.two_towers.model import TwoTowerModel

def select_device() -> torch.device:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def create_dataloaders(qd, pd, lbl, train_idx, val_idx, batch_size=32):
    train_ds = TensorDataset(qd[train_idx], pd[train_idx], lbl[train_idx])
    val_ds   = TensorDataset(qd[val_idx],   pd[val_idx],   lbl[val_idx])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size)
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for q_batch, p_batch, y_batch in loader:
        q_batch, p_batch, y_batch = (
            q_batch.to(device), p_batch.to(device), y_batch.to(device)
        )
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model(q_batch, p_batch)
            loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * q_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for q_batch, p_batch, y_batch in loader:
            q_batch, p_batch, y_batch = (
                q_batch.to(device), p_batch.to(device), y_batch.to(device)
            )
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(q_batch, p_batch)
                loss = criterion(out, y_batch)
            total_loss += loss.item() * q_batch.size(0)
    return total_loss / len(loader.dataset)


def run_cross_validation(qd, pd, lbl, n_splits=5, batch_size=32, num_epochs=100, lr=1e-3, model_dir='models'):
    device = select_device()
    print(f"Training on {device}\nCV folds={n_splits}, epochs={num_epochs}, lr={lr}\n")
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    qd, pd, lbl = qd.float(), pd.float(), lbl.float()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_loss, best_path = float('inf'), ''

    for fold, (tr_i, val_i) in enumerate(kf.split(qd), 1):
        train_loader, val_loader = create_dataloaders(qd, pd, lbl, tr_i, val_i, batch_size)
        model = TwoTowerModel(qd.size(1), pd.size(1), 64, 64, 32).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=lr)

        for epoch in range(1, num_epochs+1):
            start = time.time()
            tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - start
            if epoch % 10 == 0 or epoch == num_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Fold {fold}, Ep {epoch}/{num_epochs}: TL={tl:.4f}, VL={vl:.4f}, LR={current_lr:.2e}, time={elapsed:.2f}s')

        if vl < best_loss:
            best_loss, best_path = vl, os.path.join(model_dir, f'best_fold{fold}.pt')
            torch.save(model.state_dict(), best_path)

    return best_path


def main(train_inputs, train_labels, val_inputs, val_labels):

    device = select_device()
    train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
    train_p = torch.tensor(train_inputs[1], dtype=torch.float32)
    train_y = torch.tensor(train_labels,    dtype=torch.float32)
    val_q   = torch.tensor(val_inputs[0],   dtype=torch.float32)
    val_p   = torch.tensor(val_inputs[1],   dtype=torch.float32)
    val_y   = torch.tensor(val_labels,      dtype=torch.float32)
    

    best = run_cross_validation(
        train_q, 
        train_p, 
        train_y, 
        n_splits=NUM_SPLITS, 
        batch_size=BATCH_SIZE, 
        num_epochs=NUM_EPOCHS, 
        lr=LR,
        model_dir=MODEL_DIR
        )

    # check if input dims match
    assert train_q.size(1) == INPUT_DIM_QUERY
    assert train_p.size(1) == INPUT_DIM_PRODUCT

    model = TwoTowerModel(
        q_in=INPUT_DIM_QUERY, 
        p_in=INPUT_DIM_PRODUCT, 
        q_hidden=HIDDEN_DIM_QUERY, 
        p_hidden=HIDDEN_DIM_PRODUCT, 
        emb=EMBEDDING_DIM
        ).to(device)
    model.load_state_dict(torch.load(best, map_location=device))
    test_loader = DataLoader(TensorDataset(val_q, val_p, val_y), batch_size=BATCH_SIZE)
    test_loss = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
    print(f'Final Test Loss: {test_loss:.4f}')
    # Remove the leading '/' from 'final_model.pt'
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final_model.pt'))

if __name__ == '__main__':
    from recsys.data.load_data import load_data
    train_inputs, train_labels, val_inputs, val_labels = load_data()
    main(train_inputs, train_labels, val_inputs, val_labels)
