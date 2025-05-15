import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import List, Tuple, Optional

def select_device() -> torch.device:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    epoch: int,
    num_epochs: int,
    save_path: Optional[str] = None,
    title: str = 'Loss vs Epoch'
) -> None:
    """
    Plot training and validation losses with memory-efficient handling.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        epoch: Current epoch number
        num_epochs: Total number of epochs
        save_path: Optional path to save the plot
        title: Optional custom title for the plot
    """
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'{title} (Current Epoch: {epoch}/{num_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, len(train_losses), max(1, len(train_losses)//10)))
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=100)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save plot to {save_path}: {str(e)}")
    else:
        plt.draw()
        plt.pause(3)
        plt.close()

def setup_plot_directories(model_dir: str) -> Tuple[str, str]:
    """
    Set up directories for model and plot storage.
    
    Args:
        model_dir: Base directory for model storage
        
    Returns:
        Tuple of (model_dir, plots_dir)
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    return model_dir, plots_dir

def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    epoch: int,
    num_epochs: int,
    plots_dir: str,
    fold: int,
    is_best: bool = False
) -> None:
    """
    Save training plots for a fold and optionally the best fold.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        epoch: Current epoch number
        num_epochs: Total number of epochs
        plots_dir: Directory to save plots
        fold: Current fold number
        is_best: Whether this is the best fold
    """
    # Save fold plot
    try:
        fold_plot_path = os.path.join(plots_dir, f'fold_{fold}_losses.png')
        plot_losses(train_losses, val_losses, epoch, num_epochs, save_path=fold_plot_path)
    except Exception as e:
        print(f"Warning: Could not save plot for fold {fold}: {str(e)}")
    
    # Save best fold plot if this is the best fold
    if is_best:
        try:
            best_plot_path = os.path.join(plots_dir, 'best_fold_losses.png')
            plot_losses(train_losses, val_losses, epoch, num_epochs, save_path=best_plot_path)
        except Exception as e:
            print(f"Warning: Could not save best fold plot: {str(e)}")

def save_final_evaluation_plot(
    test_loss: float,
    model_dir: str,
    title: str = 'Final Evaluation'
) -> None:
    """
    Save the final evaluation plot.
    
    Args:
        test_loss: Final test loss value
        model_dir: Directory to save the plot
        title: Optional custom title for the plot
    """
    plots_dir = os.path.join(model_dir, 'plots')
    final_plot_path = os.path.join(plots_dir, 'final_evaluation.png')
    plot_losses([test_loss], [test_loss], 1, 1, save_path=final_plot_path, title=title)
