from typing import Literal
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import os
import pandas as pd
import numpy as np

from augmentation.Augment import Augmentor

class ASVspoof5Dataset(Dataset):
    """
    Base class for the ASVspoof5 dataset. This class should not be used directly, but rather subclassed.

    param root_dir: Path to the ASVspoof5 folder
    param protocol_file_name: Name of the protocol file to use
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    param augment: Whether to apply data augmentation (for training)
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["train", "dev", "eval"] = "train",
        augment=False,
    ):
        # Enable data augmentation base on the argument passed, but only for training
        self.augment = False if variant != "train" else augment
        if self.augment:
            self.augmentor = Augmentor()

        self.root_dir = root_dir

        protocol_file = os.path.join(self.root_dir, protocol_file_name)
        # Assuming space separated protocol file based on repository code
        self.protocol_df = pd.read_csv(protocol_file, sep=" ", header=None)

        subdir = ""
        if variant == "train":
            subdir = "flac_T"
        elif variant == "dev":
            subdir = "flac_D"
        elif variant == "eval":
            subdir = "flac_E_eval"

        self.protocol_df.columns = [
            "SPEAKER_ID",
            "AUDIO_FILE_NAME",
            "GENDER",
            "CODEC",
            "CODEC_Q",
            "CODEC_SEED",
            "ATTACK_TAG",
            "ATTACK_LABEL",
            "KEY",
            "-",
        ]
        self.rec_dir = os.path.join(self.root_dir, subdir)

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        """
        Returns tuples of the form (audio_file_name, waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        audio_name = os.path.join(self.rec_dir, f"{audio_file_name}.flac")
        waveform, _ = torchaudio.load(audio_name)

        label = 0 if self.protocol_df.loc[idx, "KEY"] == "bonafide" else 1

        if self.augment:
            waveform = self.augmentor.augment(waveform)

        return audio_file_name, waveform, label

    def get_labels(self) -> np.ndarray:
        """
        Returns an array of labels for the dataset, where 0 is genuine speech and 1 is spoofing speech
        Used for computing class weights for the loss function and weighted random sampling
        """
        return self.protocol_df["KEY"].map({"bonafide": 0, "spoof": 1}).to_numpy()

    def get_class_weights(self):
        """Returns an array of class weights for the dataset, where 0 is genuine speech and 1 is spoofing speech"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)


def pad_collate_fn(batch):
    """
    Collate function to pad waveforms to the longest in the batch.
    Batch is a list of tuples: (audio_file_name, waveform, label)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    filenames = [item[0] for item in batch]
    waveforms = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # Find max length
    max_len = max([w.shape[1] for w in waveforms])

    padded_waveforms = []
    for w in waveforms:
        if w.shape[1] < max_len:
            pad_amount = max_len - w.shape[1]
            # Pad with zeros at the end
            padded_w = torch.nn.functional.pad(w, (0, pad_amount))
            padded_waveforms.append(padded_w)
        else:
            padded_waveforms.append(w)

    return filenames, torch.stack(padded_waveforms), torch.tensor(labels)

def get_asvspoof5_dataloader(
    root_dir: str,
    protocol_file_name: str,
    variant: Literal["train", "dev", "eval"] = "train",
    batch_size: int = 4,
    augment: bool = False,
    num_workers: int = 0,
    distributed: bool = False
) -> DataLoader:
    """
    Creates and returns a DataLoader for the ASVspoof5 dataset.
    """
    dataset = ASVspoof5Dataset(
        root_dir=root_dir,
        protocol_file_name=protocol_file_name,
        variant=variant,
        augment=augment,
    )

    sampler = None
    shuffle = False

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(variant == "train"))
    elif variant == "train":
        # Weighted sampler for training to balance classes
        try:
            samples_weights = np.vectorize(dataset.get_class_weights().__getitem__)(
                dataset.get_labels()
            )
            sampler = WeightedRandomSampler(samples_weights, len(dataset))
        except Exception as e:
            print(f"Warning: Could not create WeightedRandomSampler: {e}")
            shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=pad_collate_fn,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
    )

    return dataloader
