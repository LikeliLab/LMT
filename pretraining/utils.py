"""Functions needed for pretraining."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lmt.tokenizer import BaseTokenizer
from lmt.tokenizer.bpe import BPETokenizer
from pretraining.dataset import GPTDataset


def read_text_file(file_path: Path) -> str:
    """Reads the content from a .txt file.

    Args:
        file_path (Path): A Path object pointing to the .txt file.

    Raises:
        ValueError: If the file is not a .txt file.
        FileNotFoundError: If the file does not exist at the specified path.

    Returns:
        The content of the file as a string.
    """
    if file_path.suffix.lower() != '.txt':
        raise ValueError(
            f"Error: The file '{file_path.name}' is not a .txt file."
        )

    try:
        with file_path.open('r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Error: Could not find the file at path: '{file_path}'"
        ) from err
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        raise


def create_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: BaseTokenizer | None = None,
):
    """Creates a PyTorch DataLoader for text data.

    This function processes a raw text string by tokenizing it and wrapping it
    in a `GPTDataset` and `DataLoader`, making it ready for model training.

    Args:
        txt (str): The input text data as a single string.
        batch_size (int): The number of samples per batch. Defaults to 4.
        max_length (int): The maximum sequence length for tokenization.
            Defaults to 256.
        stride (int): The stride between consecutive sequences.
            Defaults to 128.
        shuffle (bool): If True, shuffles the data at every epoch.
            Defaults to True.
        drop_last (bool): If True, drops the last incomplete batch.
            Defaults to True.
        num_workers (int): The number of subprocesses to use for data loading.
            Defaults to 0.
        tokenizer (BaseTokenizer | None): The tokenizer to use. If None, a
            default `BPETokenizer` for 'gpt2' is instantiated.
                Defaults to None.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def calc_loss_batch(
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Calculates the cross-entropy loss for a model's predictions.

    This function moves the input and target tensors to the specified device,
    passes the input through the model to get logits, and then computes the
    cross-entropy loss between the model's predictions and the true targets.

    Args:
        input_ids (torch.Tensor): The input tensor containing token IDs for the
            model. Shape: (batch_size, sequence_length).
        target_ids (torch.Tensor): The target tensor containing the ground
            truth token IDs. Shape: (batch_size, sequence_length).
        model (torch.nn.Module): The neural network model that will produce
            the logits.
        device (torch.device): The device on which to perform the calculations.

    Returns:
        torch.Tensor: A scalar tensor representing the calculated
            cross-entropy loss.
    """
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_ids.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculates the average loss over a specified number of batches.

    This function iterates through a DataLoader for a given number of batches,
    computes the loss for each batch, and returns the average loss. It's
    useful for calculating validation or test loss without necessarily
    running through the entire dataset.

    Args:
        data_loader (DataLoader): The PyTorch DataLoader to iterate over.
        model (nn.Module): The model to be evaluated.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            perform the calculations.
        num_batches (Optional[int], optional): The number of batches to use for
            calculating the loss. If None, all batches in the data_loader
            are used. Defaults to None.

    Returns:
        float: The average loss per batch. Returns float("nan") if the
            data_loader is empty.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Evaluates the model on training and validation data subsets.

    This function sets the model to evaluation mode (`model.eval()`),
    calculates the loss on a specified number of batches from both the training
    and validation dataloaders, and then returns the model to training mode
    (`model.train()`). This is typically used for periodic evaluation during a
    training loop.

    Args:
        model (nn.Module): The model to be evaluated.
        train_dataloader (DataLoader): The DataLoader for the training set.
        val_dataloader (DataLoader): The DataLoader for the validation set.
        device (torch.device): The device on which to perform calculations.
        eval_iter (int): The number of batches to use for the evaluation from
            each dataloader.

    Returns:
        Tuple[float, float]: A tuple containing the average training loss and
            the average validation loss for the evaluated batches.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_dataloader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def generate_text(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int | torch.Tensor,
):
    """Generates a sequence of token IDs using greedy decoding.

    This function iteratively predicts the next token in a sequence by choosing
    the token with the highest probability (logit) at each step. It appends
    the predicted token ID to the sequence and uses the updated sequence as
    input for the next prediction. See:
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/previous_chapters.py

    Args:
        model (nn.Module): The transformer model to use for generation.
        idx (torch.Tensor): The initial sequence of token IDs, representing the
            prompt. Shape: (batch_size, sequence_length).
        max_new_tokens (int): The maximum number of new tokens to generate and
            append to the input sequence.
        context_size (int | torch.Tensor): The maximum number of tokens the
            model can use as context. The input sequence `idx` will be cropped
            to this size from the right if it's too long.

    Returns:
        torch.Tensor: The generated sequence of token IDs, which includes the
            original prompt plus the newly generated tokens.
            Shape: (batch_size, sequence_length + max_new_tokens).
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_and_print_sample(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    device: torch.device,
    start_context: str,
):
    """Generates and prints a sample of text from the model.

    This function sets the model to evaluation mode, generates a text sample
    based on a starting context, prints the result to the console, and then
    restores the model to training mode.

    Args:
        model (nn.Module): The transformer model to use for generation.
            It's assumed to have a `pos_embed` attribute.
        tokenizer (Any): The tokenizer used for encoding the start context
            and decoding the generated token IDs. The specific type can vary.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which
            to perform the generation.
        start_context (str): The initial string to prompt the model.
    """
    model.eval()
    context_size = model.pos_embed.weight.shape[0]  # type: ignore

    encoded = tokenizer.encode(start_context)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

        flat = token_ids.squeeze(0)  # remove batch dimension
        decoded_text = tokenizer.decode(flat.tolist())
        print(decoded_text.replace('\n', ' '))  # Compact print format
    model.train()


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: BaseTokenizer,
):
    """Trains a GPT-style model for causal language modeling.

    This function handles the complete training loop, including periodic
    evaluation on a validation set and generating sample text to monitor
    the model's progress.

    Args:
        model (torch.nn.Module): The GPT model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the
            training set.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the
            validation set.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): The device to run the model on (e.g., 'cuda'
            or 'cpu').
        num_epochs (int): Total number of epochs to train for.
        eval_freq (int): Frequency (in epochs) to perform evaluation.
        eval_iter (int): Number of batches from val_dataloader to use for
            evaluation.
        start_context (str): A starting string to prompt the model for
            text generation.
        tokenizer: (BaseTokenizer) The tokenizer used for encoding and
            decoding text.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_ids, target_ids in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_ids, target_ids, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_ids.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f'Ep {epoch + 1} (Step {global_step:06d}): '
                    f'Train loss: {train_loss:.3f} '
                    f'Val loss: {val_loss:.3f}'
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(save_dir, train_losses, val_losses, tokens_seen):
    """Plots training and validation losses and saves the figure.

    Args:
        save_dir (Path): The directory where the plot will be saved.
        train_losses (list[float]): A list of training losses.
        val_losses (list[float]): A list of validation losses.
        tokens_seen (list[int]): A list of tokens seen at each evaluation step.
    """
    if not tokens_seen:
        print('No data to plot. Skipping loss plot generation.')
        return

    # Create a new figure and axes for the plot 📈
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training and validation losses
    ax.plot(tokens_seen, train_losses, label='Training Loss')
    ax.plot(tokens_seen, val_losses, label='Validation Loss')

    # Set plot title and labels
    ax.set_title('Training & Validation Loss Over Time')
    ax.set_xlabel('Tokens Seen')
    ax.set_ylabel('Loss')

    # Add a legend and grid
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot to the specified directory
    plot_save_path = save_dir / 'loss_plot.png'
    print(f'Saving loss plot to {plot_save_path}')
    plt.savefig(plot_save_path)
    plt.close(fig)  # Close the figure to free up memory
