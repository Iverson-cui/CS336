import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from cs336_basics.building_blocks import (
    transformer_lm,
    AdamW,
    cross_entropy,
    clip_grad,
    learning_rate_schedule,
    data_loading,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train a transformer language model")

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument(
        "--context_length", type=int, default=1024, help="Context window length"
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=10000.0,
        help="Theta parameter for rotary positional embeddings",
    )
    parser.add_argument(
        "--d_k",
        type=int,
        default=None,
        help="Dimension of each attention head (defaults to d_model // num_heads)",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (lambda)"
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm"
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Learning rate warmup steps"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval"
    )
    parser.add_argument(
        "--val_interval", type=int, default=500, help="Validation interval"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Checkpoint saving interval",
    )

    # Data and I/O
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (numpy .npy file)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data (numpy .npy file)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint path"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps",
        help="Device (cuda or mps)",
    )

    return parser.parse_args()


def train_one_iteration(model, optimizer, train_data, args, iteration):
    """Train for one iteration."""
    # Load a random batch
    input_ids, target_ids = data_loading(
        train_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        devstr=args.device,
    )

    # Forward pass
    logits = model(input_ids)  # Shape: (batch_size, context_length, vocab_size)

    # Compute loss
    loss = cross_entropy(logits, target_ids)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    clip_grad(model.parameters(), args.max_grad_norm)

    # Update learning rate
    current_lr = learning_rate_schedule(
        step=iteration,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        annealing_steps=args.num_iterations,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    # Optimizer step
    optimizer.step()

    return loss.item(), current_lr


@torch.no_grad()
def validate(model, val_data, args, num_batches=10):
    """Compute validation loss over multiple batches."""
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        input_ids, target_ids = data_loading(
            val_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            devstr=args.device,
        )

        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)
        total_loss += loss.item()

    model.train()
    return total_loss / num_batches


def main():
    """Main training loop."""
    args = parse_args()

    # Setup
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Compute d_k if not provided
    if args.d_k is None:
        args.d_k = args.d_model // args.num_heads
        print(f"d_k not provided, using d_model // num_heads = {args.d_k}")

    print(f"Training configuration:")
    print(f"  Device: {device}")
    print(
        f"  Model: d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}, d_k={args.d_k}, theta={args.theta}"
    )
    print(
        f"  Optimizer: lr={args.lr}, min_lr={args.min_lr}, weight_decay={args.weight_decay}"
    )
    print(
        f"  Training: batch_size={args.batch_size}, num_iterations={args.num_iterations}"
    )
    print()

    # Load data
    print("Loading data...")
    train_data = np.load(args.train_data, allow_pickle=False)
    val_data = np.load(args.val_data, allow_pickle=False)
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Val data shape: {val_data.shape}")
    print()

    # Initialize model
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        theta=args.theta,
        d_k=args.d_k,
        device=device,
    )
    model = model.to(device)
    model.train()

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

    # Resume from checkpoint if provided
    start_iteration = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        print(f"  Resumed at iteration {start_iteration}")
        print()

    # Training loop
    print("Starting training...")
    print("-" * 80)

    for iteration in range(start_iteration, args.num_iterations):
        # Train one iteration
        train_loss, current_lr = train_one_iteration(
            model, optimizer, train_data, args, iteration
        )

        # Logging
        if (iteration + 1) % args.log_interval == 0:
            print(
                f"Iteration {iteration + 1:6d} | Loss: {train_loss:.4f} | LR: {current_lr:.2e}"
            )

        # Validation
        if (iteration + 1) % args.val_interval == 0:
            val_loss = validate(model, val_data, args, num_batches=10)
            print(f"             | Val Loss: {val_loss:.4f}")

        # Checkpoint
        if (iteration + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration + 1}.pt"
            save_checkpoint(model, optimizer, iteration + 1, str(checkpoint_path))
            print(f"             | Saved checkpoint: {checkpoint_path}")

    print("-" * 80)
    print("Training complete!")

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.num_iterations, str(final_checkpoint))
    print(f"Final checkpoint saved: {final_checkpoint}")


if __name__ == "__main__":
    main()


# Usage examples:
