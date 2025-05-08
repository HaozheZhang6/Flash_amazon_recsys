"""
Train a two-tower retrieval model and save the best model.
"""

INPUT_DIM_QUERY = 32
INPUT_DIM_PRODUCT = 160
DCN_FC_OUT_DIM = 32
EMBEDDING_DIM = 32
LR = 3e-3
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_SPLITS = 5
MODEL_DIR = 'models/ranking-dcn'



import os
import time
import torch
from torch import nn, autocast
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold
from recsys.ranking.dcn.model import DCN
from recsys.recall.two_towers.utils import select_device
from dataclasses import dataclass
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

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


def loss_fn(y_true, y_pred):
    
    return nn.BCEWithLogitsLoss()(y_true, y_pred)

def plot_losses(train_losses, val_losses, epoch, num_epochs):
    """Plot training and validation losses in real-time"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Loss vs Epoch (Current Epoch: {epoch}/{num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, len(train_losses), max(1, len(train_losses)//10)))
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def run_cross_validation(qd, pd, lbl, n_splits=5, batch_size=32, num_epochs=100, lr=1e-3, model_dir='models'):
    device = select_device()
    print(f"Training on {device}\nCV folds={n_splits}, epochs={num_epochs}, lr={lr}\n")
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    qd, pd, lbl = qd.float(), pd.float(), lbl.float()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_loss, best_path = float('inf'), ''

    # Initialize plot
    plt.ion()  # Turn on interactive mode
    train_losses = []
    val_losses = []

    for fold, (tr_i, val_i) in enumerate(kf.split(qd), 1):
        train_loader, val_loader = create_dataloaders(qd, pd, lbl, tr_i, val_i, batch_size)
        model = DCN(
            config=DCNConfig(),
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)

        # Reset losses for new fold
        train_losses = []
        val_losses = []

        for epoch in range(1, num_epochs+1):
            start = time.time()
            tl = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            vl = evaluate(model, val_loader, loss_fn, device)
            elapsed = time.time() - start

            # Store losses
            train_losses.append(tl)
            val_losses.append(vl)

            # Update plot
            plot_losses(train_losses, val_losses, epoch, num_epochs)

            if epoch % 10 == 0 or epoch == num_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Fold {fold}, Ep {epoch}/{num_epochs}: TL={tl:.4f}, VL={vl:.4f}, LR={current_lr:.2e}, time={elapsed:.2f}s')

        if vl < best_loss:
            best_loss, best_path = vl, os.path.join(model_dir, f'best_fold{fold}.pt')
            torch.save(model.state_dict(), best_path)

    plt.ioff()  # Turn off interactive mode
    plt.close()  # Close the plot
    return best_path

@dataclass
class DCNConfig:
    dcn_layers: int = 4
    dnn_layers: int = 4
    q_in: int = INPUT_DIM_QUERY
    p_in: int = INPUT_DIM_PRODUCT
    hidden_dim: int = EMBEDDING_DIM
    embed_dim: int = EMBEDDING_DIM

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

    model = DCN(
        config=DCNConfig(),
        ).to(device)

    model.load_state_dict(torch.load(best, map_location=device))
    test_loader = DataLoader(TensorDataset(val_q, val_p, val_y), batch_size=BATCH_SIZE)
    test_loss = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
    print(f'Final Test Loss: {test_loss:.4f}')
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final_model.pt'))

if __name__ == '__main__':
    from recsys.data.load_data import load_data
    train_inputs, train_labels, val_inputs, val_labels = load_data(usage="ranking")
    main(train_inputs, train_labels, val_inputs, val_labels)
