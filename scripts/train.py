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
"""Main script for training a language model.

This module provides functionality for training GPT-style language models with
support for both pretraining and classification tasks. It includes command-line
argument parsing, model configuration, data preparation, and training
orchestration.

The script supports various model architectures and training configurations
through command-line arguments, and can optionally download pretrained weights
for classification fine-tuning tasks.

Typical usage example:
    python train.py --task pretraining --num_epochs 20 --batch_size 4
    python train.py --task classification --download_model --learning_rate 1e-5
"""

import argparse

import torch

from lmt.models.config import ModelConfig
from lmt.models.gpt import GPT
from lmt.training.config import BaseTrainingConfig
from lmt.training.trainer import Trainer
from scripts.utils import (
    download_model,
    prepare_classification_model,
    prepare_data,
)


def create_base_parser():
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser()

    # Task
    task_group = parser.add_argument_group('Task')
    task_group.add_argument('--task', type=str, default='pretraining')

    # Data and save paths
    data_group = parser.add_argument_group('Data and Save Paths')
    data_group.add_argument('--data_dir', type=str, default='data')
    data_group.add_argument('--save_dir', type=str, default='scripts/runs')
    data_group.add_argument('--download_model', action='store_true')

    # Model hyperparameters
    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--context_length', type=int, default=1024)
    model_group.add_argument('--vocab_size', type=int, default=50257)
    model_group.add_argument('--num_layers', type=int, default=12)
    model_group.add_argument('--num_heads', type=int, default=12)
    model_group.add_argument('--embed_dim', type=int, default=768)
    model_group.add_argument('--dropout', type=float, default=0.1)
    model_group.add_argument('--qkv_bias', action='store_true', default=False)
    model_group.add_argument(
        '--start_context', type=str, default='Every effort moves you'
    )

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--batch_size', type=int, default=2)
    training_group.add_argument('--learning_rate', type=float, default=3e-4)
    training_group.add_argument('--weight_decay', type=float, default=0.1)
    training_group.add_argument('--num_epochs', type=int, default=10)
    training_group.add_argument('--eval_freq', type=int, default=5)
    training_group.add_argument('--eval_iter', type=int, default=5)
    training_group.add_argument('--device', type=str, default='mps')
    training_group.add_argument('--num_workers', type=int, default=0)

    training_group.add_argument('--train_ratio', type=float, default=0.7)
    training_group.add_argument('--val_ratio', type=float, default=0.2)

    return parser


def main(args):
    """Main function for training."""
    # Create training config
    training_config = BaseTrainingConfig(
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
        task=args.task,
        start_context=args.start_context,  # Additional config for pretraining
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Model configuration
    model_config = ModelConfig(
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        qkv_bias=args.qkv_bias,
        ff_network=None,
    )

    device = torch.device(args.device)
    model = GPT(model_config).to(device)

    print(f'Task = {args.task}')

    if args.task.lower() == 'classification':
        if args.download_model:
            print('\n -- Download Model ---')
            download_model(model_size='124M')

        print('\n--- Load Model Pretrained Weights ---')
        pretrained_model_dir = 'scripts/models/124M/'
        prepare_classification_model(
            model, model_config, pretrained_model_dir, device
        )

    # Get data
    dataloaders = prepare_data(args)

    # Initialize trainer and train
    trainer = Trainer(
        model,
        dataloaders['train_dataloader'],  # type: ignore
        dataloaders['val_dataloader'],  # type: ignore
        training_config,
    )
    results = trainer.train()
    trainer.plot_losses(x_axis_data=results['track_examples_seen'])
    trainer.save_model()

    return results


if __name__ == '__main__':
    parser = create_base_parser()
    args = parser.parse_args()
    main(args)
