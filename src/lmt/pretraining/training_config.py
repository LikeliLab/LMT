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
"""Configuration module for training and model parameters."""

from dataclasses import dataclass
from pathlib import Path

from lmt.models import ModelConfig


@dataclass
class TrainingParametersConfig:
    """Data class for training hyperparameters."""

    # --- Training Hyperparameters ---
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    device: str = 'mps'


@dataclass
class TrainingConfig:
    """Configuration class for training.

    Args:
        data_file (Path): Path to the data file used for training.
        save_dir (Path): Directory where training outputs will be saved.
        model_config (ModelConfig): Configuration for the model.
        training_hyperparameters (TrainingParametersConfig): Hyperparameters
            for training.
    """

    # --- Data & Model Paths ---
    data_file: Path = Path('data/text_corpus.txt')
    save_dir: Path = Path('runs')

    # --- Model Hyperparameters ---
    model_config: ModelConfig = ModelConfig()

    # --- Training Hyperparameters ---
    training_hyperparameters: TrainingParametersConfig = (
        TrainingParametersConfig()
    )


# Create an instance of the config
config = TrainingConfig()
