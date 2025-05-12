"""
Simple training loop
"""
import os
import gc
import time
import torch
import wandb
import argparse
from tqdm import tqdm
from model import Model
from config import Config
from torch.optim import Adam
from torch.amp import autocast
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

def train_model(model, config, resume=False):
    """
    Training loop for Model with tokenized Hugging Face dataset.
    """
    start_epoch = 0
    if resume:
        try:
            checkpoints = [f for f in os.listdir(config.ckpt_dir) if f.startswith("checkpoint_epoch_")]
            if checkpoints:
                latest_epoch = max([int(f.split('_')[2].split('.')[0]) for f in checkpoints])
                checkpoint_path = os.path.join(config.ckpt_dir, f"checkpoint_epoch_{latest_epoch}.pt")
                checkpoint = torch.load(checkpoint_path, map_location=config.device)
                model.load_state_dict(checkpoint['model_state'])
                start_epoch = latest_epoch
                print(f"✅ Successfully loaded model checkpoint from epoch {start_epoch}")
                if os.path.exists(os.path.join(config.ckpt_dir, "wandb_id.txt")):
                    with open(os.path.join(config.ckpt_dir, "wandb_id.txt")) as f:
                        wandb_id = f.read().strip()
                        wandb.init(
                            project="small-gpt",
                            id=wandb_id,
                            resume="must"
                        )
                else:
                    raise Exception("W&B ID not found")
        except Exception as e:
            print(f"❌ Model checkpoint load failed: {str(e)}")
            raise e
    else:
        run = wandb.init(project="small-gpt", config=vars(config))
        print(f"✅ Successfully initialized new run: {run.id}")
        with open(os.path.join(config.ckpt_dir, "wandb_id.txt"), 'w') as f:
            f.write(run.id)

    # Load dataset with streaming
    dataset = load_dataset(config.dataset_name, split="train", streaming=True, columns=['text'])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=config.max_position_embeddings)
    
    tokenized_dataset = dataset \
        .map(tokenize_function, batched=True) \
        .remove_columns(['text', 'token_type_ids', 'attention_mask']) \
        .batch(config.batch_size+1) \
        .with_format("torch")

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # Log model parameters and gradients
    wandb.watch(model, log="all", log_freq=1000)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    for epoch in tqdm(range(start_epoch, config.num_epochs), desc="Epochs"):
        start_time = time.time()
        total_loss = 0.0
        batch_count = 0
        
        for batch in tokenized_dataset:
            if batch_count >= config.max_steps:
                break

            # Extract inputs and targets (for autoregressive models, targets are shifted inputs)
            inputs = batch["input_ids"][:-1].to(config.device)
            targets = batch["input_ids"][1:].to(config.device)
            
            # Forward pass
            with autocast(device_type=config.device, dtype=config.dtype):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Log metrics to wandb
            wandb.log({
                "loss": loss.item(),
                "memory": torch.mps.current_allocated_memory() / 1e9,
                "batch_time": time.time() - start_time
            })
            
            # Log progress
            if batch_count % config.log_every == 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch + 1} | Batch {batch_count} | Loss: {avg_loss:.4f}")
                wandb.log({
                    "avg_loss": avg_loss,
                    "epoch": epoch + 1,
                    "batch": batch_count
                })

            # Garbage collection
            if batch_count % 10 == 0:
                gc.collect()
                torch.mps.empty_cache()

        # Save checkpoint
        checkpoint_path = os.path.join(config.ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'config': vars(config)
        }, checkpoint_path)
        wandb.save(checkpoint_path)
        
        # Epoch-end logging
        avg_loss = total_loss / batch_count
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch_time": time.time() - start_time,
            "epoch": epoch + 1
        })
        
        # Save the best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(config.ckpt_dir, "best_checkpoint.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'config': vars(config)
            }, checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"Saved best checkpoint with loss: {best_loss:.4f}")

        print(torch.mps.current_allocated_memory() / 1e9, "GB")

    # Finish wandb run
    wandb.finish()

def load_best_checkpoint(model, checkpoint_path="best_checkpoint.pt"):
    """
    Load the best checkpoint and return the model.
    """
    checkpoint = torch.load(os.path.join(model.config.ckpt_dir, checkpoint_path))
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded best checkpoint from {checkpoint_path}")
    return model

if __name__ == "__main__":
    config = Config()
    model = Model(config)
    model.to(config.device)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train_model(model, config, args.resume)
    