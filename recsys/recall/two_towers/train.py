"""
Train a two-tower retrieval model and save the best model.
"""

INPUT_DIM_QUERY = 32
INPUT_DIM_PRODUCT = 160

HIDDEN_DIM_PRODUCT = 64
HIDDEN_DIM_QUERY = 64

EMBEDDING_DIM = 32
LR = 1e-3
NUM_EPOCHS = 40
BATCH_SIZE = 512
NUM_SPLITS = 5
MODEL_DIR = 'models/two_towers'
CHUNK_SIZE = 32




import os
import time
import torch
from torch import nn, autocast
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold
from recsys.recall.two_towers.model import TwoTowerModel
from recsys.recall.two_towers.utils import select_device
import matplotlib.pyplot as plt
import numpy as np
from recsys.utils.training_utils import (
    plot_losses,
    setup_plot_directories,
    save_training_plots,
    save_final_evaluation_plot
)


def create_dataloaders(qd, pd, lbl, train_idx, val_idx, batch_size=32, training_method="point_wise"):
    if training_method == "point_wise":
        train_ds = TensorDataset(qd[train_idx], pd[train_idx], lbl[train_idx])
        val_ds = TensorDataset(qd[val_idx], pd[val_idx], lbl[val_idx])
    elif training_method == "pair_wise":
        # For pair-wise, pd should be a tuple of (positive_products, negative_products)
        train_ds = TensorDataset(qd[train_idx], pd[0][train_idx], pd[1][train_idx], lbl[train_idx])
        val_ds = TensorDataset(qd[val_idx], pd[0][val_idx], pd[1][val_idx], lbl[val_idx])
    elif training_method == "list_wise":
        # For list-wise, pd should be a tuple of (products, rewards, frequencies)
        train_ds = TensorDataset(qd[train_idx], pd[0][train_idx], pd[1][train_idx], pd[2][train_idx])
        val_ds = TensorDataset(qd[val_idx], pd[0][val_idx], pd[1][val_idx], pd[2][val_idx])
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size)
    )

def point_wise_loss(model, q_batch, p_batch, y_batch):
    """Point-wise loss using cross entropy on cosine similarity"""
    total_loss = 0.0
    num_chunks = (q_batch.size(0) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, q_batch.size(0))
        
        # Get chunks
        q_chunk = q_batch[start_idx:end_idx]
        p_chunk = p_batch[start_idx:end_idx]
        y_chunk = y_batch[start_idx:end_idx]
        
        # Calculate loss for this chunk
        out = model(q_chunk, p_chunk)
        chunk_loss = nn.BCEWithLogitsLoss()(out, y_chunk)
        total_loss += chunk_loss * (end_idx - start_idx)
    
    return total_loss / q_batch.size(0)

def pair_wise_loss(model, q_batch, p_pos_batch, p_neg_batch, margin=0.1):
    """Pair-wise loss with margin"""
    total_loss = 0.0
    num_chunks = (q_batch.size(0) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, q_batch.size(0))
        
        # Get chunks
        q_chunk = q_batch[start_idx:end_idx]
        p_pos_chunk = p_pos_batch[start_idx:end_idx]
        p_neg_chunk = p_neg_batch[start_idx:end_idx]
        
        # Calculate similarities for this chunk
        pos_sim = model(q_chunk, p_pos_chunk)
        neg_sim = model(q_chunk, p_neg_chunk)
        
        # Calculate loss for this chunk
        chunk_loss = torch.clamp(margin - (pos_sim - neg_sim), min=0)
        total_loss += chunk_loss.sum()
    
    return total_loss / q_batch.size(0)

def list_wise_loss(model, q_batch, p_batch, r_batch, freq_batch):
    """List-wise loss using softmax and cross entropy with frequency adjustment"""
    batch_size = q_batch.size(0)
    
    # 1. Calculate similarity matrix: each query with all products
    q_expanded = q_batch.unsqueeze(1).expand(-1, batch_size, -1)
    p_expanded = p_batch.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Reshape for model computation
    q_flat = q_expanded.reshape(-1, q_batch.size(-1))
    p_flat = p_expanded.reshape(-1, p_batch.size(-1))
    
    # Compute similarities
    sim = model(q_flat, p_flat)
    sim = sim.view(batch_size, batch_size)
    
    # Adjust similarities with frequency
    freq_adjustment = torch.log(freq_batch.float() + 1).unsqueeze(0).expand(batch_size, -1)
    sim = sim - freq_adjustment
    
    # Apply softmax along product dimension
    probs = torch.softmax(sim, dim=1)
    
    # Create one-hot matrix for positive products
    one_hot = torch.eye(batch_size, device=probs.device)
    
    # Calculate cross entropy loss and scale by rewards
    loss = -torch.sum(one_hot * torch.log(probs + 1e-8), dim=1) * r_batch
    
    return loss.mean()

def train_one_epoch(model, loader, optimizer, device, training_method="point_wise"):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        if training_method == "point_wise":
            q_batch, p_batch, y_batch = [b.to(device) for b in batch]
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = point_wise_loss(model, q_batch, p_batch, y_batch)
            
        elif training_method == "pair_wise":
            q_batch, p_pos_batch, p_neg_batch, y_batch = [b.to(device) for b in batch]
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = pair_wise_loss(model, q_batch, p_pos_batch, p_neg_batch)
            
        elif training_method == "list_wise":
            q_batch, p_batch, r_batch, freq_batch = [b.to(device) for b in batch]
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = list_wise_loss(model, q_batch, p_batch, r_batch, freq_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * q_batch.size(0)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, training_method="point_wise"):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            if training_method == "point_wise":
                q_batch, p_batch, y_batch = [b.to(device) for b in batch]
                if y_batch.dim() == 1:
                    y_batch = y_batch.unsqueeze(1)
                loss = point_wise_loss(model, q_batch, p_batch, y_batch)
                
            elif training_method == "pair_wise":
                q_batch, p_pos_batch, p_neg_batch, y_batch = [b.to(device) for b in batch]
                loss = pair_wise_loss(model, q_batch, p_pos_batch, p_neg_batch)
                
            elif training_method == "list_wise":
                q_batch, p_batch, r_batch, freq_batch = [b.to(device) for b in batch]
                # Create a zero tensor with the same shape as r_batch for validation
                freq_batch = torch.zeros_like(r_batch)
                loss = list_wise_loss(model, q_batch, p_batch, r_batch, freq_batch)
            
            total_loss += loss.item() * q_batch.size(0)
    
    return total_loss / len(loader.dataset)

def run_cross_validation(qd, pd, lbl, n_splits=5, batch_size=32, num_epochs=100, lr=1e-3, 
                        model_dir='models', training_method="point_wise"):
    device = select_device()
    print(f"Training on {device}\nCV folds={n_splits}, epochs={num_epochs}, lr={lr}\n")
    
    # Setup directories
    method_dir = os.path.join(model_dir, training_method)
    method_dir, plots_dir = setup_plot_directories(method_dir)
    
    # Convert to float based on training method
    qd = qd.float()
    if training_method == "pair_wise":
        pd = (pd[0].float(), pd[1].float())
    elif training_method == "list_wise":
        pd = (pd[0].float(), pd[1].float(), pd[2].float())
    else:
        pd = pd.float()
    lbl = lbl.float()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_loss, best_path = float('inf'), ''
    best_fold = 0
    best_train_losses = []
    best_val_losses = []
    
    # Initialize plot
    plt.ion()
    train_losses = []
    val_losses = []
    
    # Calculate log probabilities for list-wise training
    
    for fold, (tr_i, val_i) in enumerate(kf.split(qd), 1):
        train_loader, val_loader = create_dataloaders(
            qd, pd, lbl, tr_i, val_i, batch_size, training_method=training_method
        )
        model = TwoTowerModel(qd.size(1), pd[0].size(1) if training_method in ["pair_wise", "list_wise"] else pd.size(1), 
                            64, 64, 32).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        
        # Reset losses for new fold
        train_losses = []
        val_losses = []
        
        for epoch in range(1, num_epochs+1):
            start = time.time()
            tl = train_one_epoch(model, train_loader, optimizer, device, 
                               training_method)
            vl = evaluate(model, val_loader, device, training_method)
            
            elapsed = time.time() - start
            
            # Store losses
            train_losses.append(tl)
            val_losses.append(vl)
            
            # Update plot every 5 epochs
            if epoch % 5 == 0 or epoch == num_epochs:
                plot_losses(train_losses, val_losses, epoch, num_epochs)
            
            if epoch % 10 == 0 or epoch == num_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Fold {fold}, Ep {epoch}/{num_epochs}: TL={tl:.4f}, VL={vl:.4f}, '
                      f'LR={current_lr:.2e}, time={elapsed:.2f}s')
        
            # Save plots for this fold
            save_training_plots(
                train_losses,
                val_losses,
                epoch,
                num_epochs,
                plots_dir,
                fold,
                is_best=(vl < best_loss)
            )
        
        if vl < best_loss:
            best_loss = vl
            best_path = os.path.join(method_dir, f'best_fold{fold}.pt')
            best_fold = fold
            best_train_losses = train_losses.copy()
            best_val_losses = val_losses.copy()
            torch.save(model.state_dict(), best_path)
    
    # Save the best fold's plot with error handling
    try:
        best_plot_path = os.path.join(plots_dir, 'best_fold_losses.png')
        plot_losses(best_train_losses, best_val_losses, num_epochs, num_epochs, save_path=best_plot_path)
    except Exception as e:
        print(f"Warning: Could not save best fold plot: {str(e)}")
    
    plt.ioff()
    plt.close('all') 
    return best_path, best_fold, best_loss

def main(train_inputs, train_labels, val_inputs, val_labels, training_method="point_wise"):
    device = select_device()
    
    if training_method == "point_wise":
        train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
        train_p = torch.tensor(train_inputs[1], dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        val_q = torch.tensor(val_inputs[0], dtype=torch.float32)
        val_p = torch.tensor(val_inputs[1], dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.float32)
        
        best, best_fold, best_loss = run_cross_validation(
            train_q, train_p, train_y,
            n_splits=NUM_SPLITS,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            model_dir=MODEL_DIR,
            training_method=training_method
        )
        
    elif training_method == "pair_wise":
        train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
        train_p_pos = torch.tensor(train_inputs[1], dtype=torch.float32)
        train_p_neg = torch.tensor(train_inputs[2], dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        val_q = torch.tensor(val_inputs[0], dtype=torch.float32)
        val_p_pos = torch.tensor(val_inputs[1], dtype=torch.float32)
        val_p_neg = torch.tensor(val_inputs[2], dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.float32)
        
        # For pair-wise, we need to pass both positive and negative products
        train_p = (train_p_pos, train_p_neg)
        val_p = (val_p_pos, val_p_neg)
        
        best, best_fold, best_loss = run_cross_validation(
            train_q, train_p, train_y,
            n_splits=NUM_SPLITS,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            model_dir=MODEL_DIR,
            training_method=training_method
        )
        
    elif training_method == "list_wise":
        train_q = torch.tensor(train_inputs[0], dtype=torch.float32)
        train_p = torch.tensor(train_inputs[1], dtype=torch.float32)
        train_r = torch.tensor(train_inputs[2], dtype=torch.float32)
        train_freq = torch.tensor(train_inputs[3], dtype=torch.float32)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        
        val_q = torch.tensor(val_inputs[0], dtype=torch.float32)
        val_p = torch.tensor(val_inputs[1], dtype=torch.float32)
        val_r = torch.tensor(val_inputs[2], dtype=torch.float32)
        val_freq = torch.tensor(val_inputs[3], dtype=torch.float32)
        val_y = torch.tensor(val_labels, dtype=torch.float32)
    
        # For list-wise, we need to pass products, rewards, and frequencies
        train_p = (train_p, train_r, train_freq)
        val_p = (val_p, val_r, val_freq)
        
        best, best_fold, best_loss = run_cross_validation(
        train_q, train_p, train_y,
        n_splits=NUM_SPLITS,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        model_dir=MODEL_DIR,
        training_method=training_method
    )
    
    model = TwoTowerModel(
        q_in=INPUT_DIM_QUERY,
        p_in=INPUT_DIM_PRODUCT,
        q_hidden=HIDDEN_DIM_QUERY,
        p_hidden=HIDDEN_DIM_PRODUCT,
        emb=EMBEDDING_DIM
    ).to(device)
    
    model.load_state_dict(torch.load(best, map_location=device))
    
    # Create test loader based on training method
    if training_method == "point_wise":
        test_loader = DataLoader(TensorDataset(val_q, val_p, val_y), batch_size=BATCH_SIZE)
    elif training_method == "pair_wise":
        test_loader = DataLoader(TensorDataset(val_q, val_p[0], val_p[1], val_y), batch_size=BATCH_SIZE)
    elif training_method == "list_wise":
        test_loader = DataLoader(TensorDataset(val_q, val_p[0], val_p[1], val_p[2]), batch_size=BATCH_SIZE)
    
    test_loss = evaluate(model, test_loader, device, training_method)
    
    # After training, save the final evaluation plot
    save_final_evaluation_plot(test_loss, os.path.join(MODEL_DIR, training_method))
    
    print(f'Final Test Loss: {test_loss:.4f}')
    print(f'Best model was from fold {best_fold} with validation loss: {best_loss:.4f}')
    print(f'Plots saved in: {os.path.join(MODEL_DIR, training_method, "plots")}')
    
    # Save final model with training method in the name
    final_model_path = os.path.join(MODEL_DIR, training_method, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)

if __name__ == '__main__':
    from recsys.data.load_data import load_data
    method = "list_wise"
    train_inputs, train_labels, val_inputs, val_labels = load_data(training_method=method)
    print(f"Training method: {method}")
    main(train_inputs, train_labels, val_inputs, val_labels, training_method=method)
    # method = "pair_wise"
    # train_inputs, train_labels, val_inputs, val_labels = load_data(training_method=method)
    # print(f"Training method: {method}")
    # main(train_inputs, train_labels, val_inputs, val_labels, training_method=method)
    # method = "point_wise"
    # train_inputs, train_labels, val_inputs, val_labels = load_data(training_method=method)
    # print(f"Training method: {method}")
    # main(train_inputs, train_labels, val_inputs, val_labels, training_method=method)
