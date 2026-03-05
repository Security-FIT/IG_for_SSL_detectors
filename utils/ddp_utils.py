import os
import torch
import torch.distributed as dist
import builtins
import numpy as np

def setup_ddp():
    """
    Initialize Distributed Data Parallel (DDP) environment.
    Returns:
        use_ddp (bool): True if DDP is initialized
        rank (int): Global rank
        local_rank (int): Local rank
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        
        # Suppress printing on non-master ranks
        if rank != 0:
            def print_pass(*args, **kwargs):
                pass
            builtins.print = print_pass
            
        print(f"Initialized DDP environment (Rank {rank}/{world_size}, Local Rank {local_rank})")
        return True, rank, local_rank
    else:
        print("Not using Distributed Data Parallel")
        return False, 0, 0

def cleanup_ddp():
    """Cleanup DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def gather_tensors(labels_list, scores_list, device):
    """
    Gather tensors from all ranks. Used in training loop.
    Assumes equal size chunks on each rank (DistributedSampler default).
    Returns numpy arrays of concatenated labels and scores.
    """
    if not dist.is_initialized():
        return np.array(labels_list), np.array(scores_list)

    # Convert lists to tensors
    labels_tensor = torch.tensor(labels_list, device=device)
    scores_tensor = torch.tensor(scores_list, device=device)
    
    world_size = dist.get_world_size()
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
    gathered_scores = [torch.zeros_like(scores_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_labels, labels_tensor)
    dist.all_gather(gathered_scores, scores_tensor)
    
    # Concatenate
    all_labels = torch.cat(gathered_labels).cpu().numpy()
    all_scores = torch.cat(gathered_scores).cpu().numpy()
    
    return all_labels, all_scores

def gather_eval_results(labels_list, scores_list, filenames_list):
    """
    Gather results from all ranks to rank 0 using all_gather_object.
    Returns (all_labels, all_scores, all_filenames) on rank 0.
    Returns ([], [], []) on other ranks.
    """
    if not dist.is_initialized():
        return labels_list, scores_list, filenames_list
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Collect local data into a dictionary
    local_data = {
        'labels': labels_list,
        'scores': scores_list,
        'filenames': filenames_list
    }
    
    # Prepare list for gathered data
    gathered_data = [None for _ in range(world_size)]
    
    # Gather objects (this handles serialization of lists containing mixed types)
    dist.all_gather_object(gathered_data, local_data)
    
    if rank == 0:
        all_labels = []
        all_scores = []
        all_filenames = []
        
        for data in gathered_data:
            if data is not None:
                all_labels.extend(data['labels'])
                all_scores.extend(data['scores'])
                all_filenames.extend(data['filenames'])
        
        return all_labels, all_scores, all_filenames
    else:
        return [], [], []
