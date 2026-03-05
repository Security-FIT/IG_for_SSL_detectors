#!/usr/bin/env python3

import argparse
import os
import config

# Set Hugging Face cache directory before importing transformers (via models)
os.environ["HF_HOME"] = config.HF_HOME

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

from models.wavlm_sls import WavLM_SLS
from models.wavlm_camhfa import WavLM_CAMHFA
from models.wavlm_aasist import WavLM_AASIST
from utils.asvspoof5_dataset import get_asvspoof5_dataloader
from utils.metrics import calculate_EER, calculate_minDCF
from utils.ddp_utils import setup_ddp, cleanup_ddp, gather_tensors

def train_epoch(model, dataloader, criterion, optimizer, device, sampler=None, epoch=0):
    if sampler:
        sampler.set_epoch(epoch)

    model.train()
    running_loss = 0.0
    
    all_labels = []
    all_scores = []
    
    # Only show progress bar on rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        iterator = dataloader
    else:
        iterator = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
    for filenames, waveforms, labels in iterator:
        waveforms = waveforms.to(device)
        labels_dev = labels.to(device).float().unsqueeze(1) # [Batch, 1]
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(waveforms)
        loss = criterion(logits, labels_dev)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item() * waveforms.size(0)
        
        # Collect for metrics
        all_labels.extend(labels.cpu().numpy())
        # Pass raw logits. Higher logit = Higher probability of Spoof (Class 1).
        all_scores.extend(logits.detach().cpu().numpy().flatten())
        
        if not (dist.is_initialized() and dist.get_rank() != 0):
            iterator.set_postfix({'loss': running_loss/len(all_labels)})
        
    # Gather results for global metrics
    all_labels_np, all_scores_np = gather_tensors(all_labels, all_scores, device)
    
    if dist.is_initialized():
        # Average loss across processes for reporting
        avg_loss = torch.tensor(running_loss / len(all_labels), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()
    else:
        avg_loss = running_loss / len(all_labels)

    # Compute metrics on rank 0 (or all, but result is same)
    eer = calculate_EER(all_labels_np, all_scores_np)
    min_dcf = calculate_minDCF(all_labels_np, all_scores_np)
        
    return avg_loss, eer, min_dcf

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    all_labels = []
    all_scores = []
    
    # Only show progress bar on rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        iterator = dataloader
    else:
        iterator = tqdm(dataloader, desc="Validation")
        
    with torch.no_grad():
        for filenames, waveforms, labels in iterator:
            waveforms = waveforms.to(device)
            labels_dev = labels.to(device).float().unsqueeze(1)
            
            logits = model(waveforms)
            loss = criterion(logits, labels_dev)
            
            running_loss += loss.item() * waveforms.size(0)
            
            # Collect for metrics
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(logits.cpu().numpy().flatten())
            
            if not (dist.is_initialized() and dist.get_rank() != 0):
                iterator.set_postfix({'loss': running_loss/len(all_labels)})
            
    # Gather results
    all_labels_np, all_scores_np = gather_tensors(all_labels, all_scores, device)

    if dist.is_initialized():
        avg_loss = torch.tensor(running_loss / len(all_labels), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()
    else:
        avg_loss = running_loss / len(all_labels)
        
    # Compute metrics
    eer = calculate_EER(all_labels_np, all_scores_np)
    min_dcf = calculate_minDCF(all_labels_np, all_scores_np)
            
    return avg_loss, eer, min_dcf


def main():
    parser = argparse.ArgumentParser(description="Train ASVspoof5 Models")
    parser.add_argument("--model", type=str, required=True, choices=["sls", "camhfa", "aasist"], help="Model architecture")
    # Arguments now default to config values
    parser.add_argument("--data_dir", type=str, default=config.DATA_DIR, help="Path to ASVspoof5 root directory")
    parser.add_argument("--train_protocol", type=str, default=config.TRAIN_PROTOCOL, help="Train protocol filename")
    parser.add_argument("--dev_protocol", type=str, default=config.DEV_PROTOCOL, help="Dev protocol filename (optional)")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR, help="Directory to save checkpoints")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (per GPU)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--freeze_wavlm", action="store_true", help="Freeze WavLM backbone")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of epochs to freeze WavLM before end-to-end training")
    
    args = parser.parse_args()
    
    # Setup DDP
    use_ddp, global_rank, local_rank = setup_ddp()
    
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        device = torch.device(args.device)
        print(f"Using device: {device}")
    
    # Only make dirs on main process
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Initialize Model
    if global_rank == 0:
        print(f"Initializing model: {args.model}")
    
    # If warmup is requested, ensure we start frozen
    initial_freeze = args.freeze_wavlm or (args.warmup_epochs > 0)
    
    if args.model == "sls":
        model = WavLM_SLS(freeze_wavlm=initial_freeze)
    elif args.model == "camhfa":
        model = WavLM_CAMHFA(freeze_wavlm=initial_freeze)
    elif args.model == "aasist":
        model = WavLM_AASIST(freeze_wavlm=initial_freeze)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Move model to device BEFORE DDP wrap
    model.to(device)
    
    if use_ddp:
        # Wrap model
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # Access underlying model for parameter groups if needed via model.module
        model_module = model.module
    else:
        model_module = model
    
    # 2. Data Loaders
    if global_rank == 0:
        print("Loading data...")
        
    train_loader = get_asvspoof5_dataloader(
        root_dir=args.data_dir,
        protocol_file_name=args.train_protocol,
        variant="train",
        batch_size=args.batch_size,
        augment=args.augment,
        num_workers=3,
        distributed=use_ddp
    )
    
    # Extract sampler if distributed
    train_sampler = train_loader.sampler if use_ddp else None
    
    val_loader = None
    if args.dev_protocol:
        val_loader = get_asvspoof5_dataloader(
            root_dir=args.data_dir,
            protocol_file_name=args.dev_protocol,
            variant="dev",
            batch_size=args.batch_size,
            augment=False,
            num_workers=3,
            distributed=use_ddp 
        )
        
    # 3. Optimizer & Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Separate parameters for different learning rates
    # Use model_module to access attributes
    wavlm_params = [p for p in model_module.wavlm.parameters() if p.requires_grad]
    classifier_params = [p for n, p in model_module.named_parameters() if not n.startswith("wavlm") and p.requires_grad]
    
    param_groups = []
    if wavlm_params:
        param_groups.append({'params': wavlm_params, 'lr': 1e-6})
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': 1e-3})
        
    optimizer = optim.AdamW(param_groups)
    
    # 4. Training Loop
    best_val_min_dcf = 1.0
    
    for epoch in range(args.epochs):
        if global_rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Check for warmup end
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs:
            if global_rank == 0:
                print("Warmup complete. Unfreezing WavLM backbone...")
            
            # We need to unfreeze on all processes
            model_module.wavlm.train()
            for param in model_module.wavlm.parameters():
                param.requires_grad = True
            
            wavlm_params = [p for p in model_module.wavlm.parameters() if p.requires_grad]
            optimizer.add_param_group({'params': wavlm_params, 'lr': 1e-6})
            
            if global_rank == 0:
                print(f"Added {len(wavlm_params)} WavLM parameters to optimizer.")
        
        train_loss, train_eer, train_min_dcf = train_epoch(model, train_loader, criterion, optimizer, device, sampler=train_sampler, epoch=epoch)
        
        if global_rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train EER: {train_eer*100:.3f}%, Train minDCF: {train_min_dcf:.6f}")
        
        if val_loader:
            val_loss, val_eer, val_min_dcf = validate(model, val_loader, criterion, device)
            
            if global_rank == 0:
                print(f"Val Loss: {val_loss:.4f}, Val EER: {val_eer*100:.3f}%, Val minDCF: {val_min_dcf:.6f}")
                
                if val_min_dcf < best_val_min_dcf:
                    best_val_min_dcf = val_min_dcf
                    save_path = os.path.join(args.output_dir, f"{args.model}_best.pt")
                    torch.save(model_module.state_dict(), save_path)
                    print(f"Saved best model to {save_path}")
        
        # Save latest
        if global_rank == 0:
            save_path = os.path.join(args.output_dir, f"{args.model}_epoch_{epoch+1}.pt")
            torch.save(model_module.state_dict(), save_path)

    if global_rank == 0:
        print("Training complete.")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()
