# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer classes for training logic.

This module contains the core classes for managing the training
and evaluation loops of a machine learning model.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lmt.tokenizer.base import BaseTokenizer
from lmt.training.config import BaseTrainingConfig


class BaseTrainer(ABC):
    """Base trainer class with common functionality.

    This class provides a foundation for training a PyTorch model. It
    handles model and data loader initialization, device placement,
    optimizer setup, and basic state tracking for training.

    Attributes:
            model (nn.Module): The PyTorch model to train, moved to the
                specified device.
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.
            config (BaseTrainingConfig): The configuration object for training.
            tokenizer (BaseTokenizer | None): The tokenizer instance, if
                provided.
            device (torch.device): The device (e.g., 'cuda', 'cpu') the model
                is on.
            optimizer (torch.optim.Optimizer): The optimizer for updating
                model weights.
            train_losses (list): A list to store training loss values per step.
            val_losses (list): A list to store validation loss values.
            global_step (int): A counter for the number of training steps
                completed.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: BaseTrainingConfig,
        tokenizer: BaseTokenizer | None = None,
    ):
        """Initializes the BaseTrainer.

        Args:
            model: The PyTorch model (nn.Module) to be trained.
            train_loader: A DataLoader for the training dataset.
            val_loader: A DataLoader for the validation dataset.
            config: A BaseTrainingConfig object containing training
                hyperparameters.
            tokenizer: An optional BaseTokenizer instance for text processing.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.tokenizer = tokenizer

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Initialize optimizer
        # TO DO: Generalize to other optimizers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.global_step = -1

    def save_model(
        self,
        save_subdir: Path = Path('model_weights'),
        save_name: Path = Path('model_and_optimizer.pth'),
    ):
        """Save model and optimizer state.

        Args:
            save_subdir (Path):  The directory where the weights will be saved.
            save_name (Path): File name
        """
        save_path = Path(self.config.save_dir).joinpath(save_subdir, save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f'Saving model and optimizer to {save_path}')
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__,
            },
            save_path,
        )

    def plot_losses(
        self,
        x_axis_data: list,
        save_subdir: Path = Path('plots'),
        save_name: Path = Path('loss_plot.png'),
    ):
        """Plots training and validation losses and saves the figure.

        Args:
            x_axis_data: Data for x-axis (e.g. number of training steps).
            save_subdir (Path): The directory where the plot will be saved.
            save_name (Path): File name
        """
        if not x_axis_data:
            print('No data to plot. Skipping loss plot generation.')
            return

        # Create a new figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot training and validation losses
        ax.plot(x_axis_data, self.train_losses, label='Training Loss')
        ax.plot(x_axis_data, self.val_losses, label='Validation Loss')

        # Set plot title and labels
        ax.set_title('Training & Validation Loss Over Time')
        ax.set_xlabel('Tokens Seen')
        ax.set_ylabel('Loss')

        # Add a legend and grid
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Save the plot to the specified directory
        save_path = Path(self.config.save_dir).joinpath(save_subdir, save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving loss plot to {save_path}')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory

    @abstractmethod
    def train_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> torch.Tensor:
        """Single training step - must be implemented by subclasses."""
        pass

    @abstractmethod
    def evaluate_step(self) -> tuple[float, float]:
        """Single evaluation step - must be implemented by subclasses."""
        pass

    def train(self) -> dict[str, Any]:
        """Main training loop."""
        print('Starting model training...')
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.train_step(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

                # Periodic evaluation
                if self.global_step % self.config.eval_freq == 0:
                    train_loss, val_loss = self.evaluate_step()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)

                    print(
                        f'Ep {epoch + 1} (Step {self.global_step:06d}): '
                        f'Train loss: {train_loss:.3f}, '
                        f'Val loss: {val_loss:.3f}'
                    )

        end_time = time.time()
        execution_time = (end_time - start_time) / 60
        print(f'Training completed in {execution_time:.2f} minutes.')

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'execution_time': execution_time,
        }
