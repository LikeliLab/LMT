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
"""Class for pretraining logic.

This module provides the PreTrainer class, which inherits from BaseTrainer
and implements specific training and evaluation steps for pretraining tasks.
"""

from typing import Any

import torch

from lmt.training.loss import calc_loss_batch, evaluate_model
from lmt.training.trainer.base import BaseTrainer


class PreTrainer(BaseTrainer):
    """Trainer for pretraining tasks.

    This class extends the `BaseTrainer` to handle the specific logic
    for pretraining models. It includes methods for a single training step,
    an evaluation step, and the full training loop with plotting.

    Attributes:
        tokens_seen (int): The total number of tokens processed by the model.
        track_tokens_seen (list[int]): A list to store `tokens_seen` at each
            evaluation step.
        track_global_steps (list[int]): A list to store `global_step` at each
            evaluation step.
        global_step (int): The current training step count.
            Inherited from BaseTrainer.
        config (Any): The configuration object for the trainer.
            Inherited from BaseTrainer.
        model (torch.nn.Module): The model to be trained.
            Inherited from BaseTrainer.
        train_loader (torch.utils.data.DataLoader): Data loader for the
            training set. Inherited from BaseTrainer.
        val_loader (torch.utils.data.DataLoader): Data loader for the
            validation set. Inherited from BaseTrainer.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the PreTrainer.

        Args:
            *args: Variable length argument list to be passed to the
                superclass.
            **kwargs: Arbitrary keyword arguments to be passed to the
                superclass.
        """
        super().__init__(*args, **kwargs)
        self.tokens_seen = 0
        self.track_tokens_seen = []
        self.track_global_steps = []

    def train_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single training step for pretraining.

        Calculates the loss for a batch, updates the `tokens_seen` and
        `global_step` counters.

        Args:
            input_batch: A `torch.Tensor` containing the input data for the
                batch.
            target_batch: A `torch.Tensor` containing the target data for the
                batch.

        Returns:
            A `torch.Tensor` representing the calculated loss for the batch.
        """
        loss = calc_loss_batch(
            input_batch, target_batch, self.model, self.device, 'pretraining'
        )
        self.tokens_seen += input_batch.numel()
        self.global_step += 1
        return loss

    def evaluate_step(self) -> tuple[float, float]:
        """Performs an evaluation step for pretraining.

        Evaluates the model on both the training and validation datasets,
        then records the current number of tokens seen and global steps.

        Returns:
            A tuple containing:
                - The average training loss (`float`).
                - The average validation loss (`float`).
        """
        train_loss, val_loss = evaluate_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.device,
            self.config.eval_iter,
            'pretraining',
        )
        self.track_tokens_seen.append(self.tokens_seen)
        self.track_global_steps.append(self.global_step)
        return train_loss, val_loss

    def train(self) -> dict[str, Any]:
        """Executes the full training loop and plots the results.

        Calls the superclass's `train` method to run the main training loop,
        then plots the losses against the number of tokens seen. Finally,
        it updates the results dictionary with the tokens seen and global
        steps.

        Returns:
            A `dict` containing the training results, including the tracked
            tokens seen and global steps.
        """
        results = super().train()
        self.plot_losses(self.track_tokens_seen)

        results['tokens_seen'] = self.track_tokens_seen
        results['global_steps'] = self.track_global_steps
        return results
