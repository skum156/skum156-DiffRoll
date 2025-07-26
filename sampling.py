# Fixed sampling.py for Automatic Music Transcription (AMT) using DiffRoll

from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# --- IMPORTANT CHANGE: Explicitly import ClassifierFreeDiffRoll from diffwave.py ---
from model.diffwave import ClassifierFreeDiffRoll

import AudioLoader.music.amt as MusicDataset
from utils.custom_dataset import Custom

from omegaconf import OmegaConf, DictConfig # Import DictConfig for type hinting
import warnings
import os # Import os for debugging log file path
import sys # Import sys for printing traceback to stdout

@hydra.main(config_path="config", config_name="sampling")
def main(cfg: DictConfig): # Added type hint for cfg
    # Resolve paths to absolute paths
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.dataset.args.audio_path = to_absolute_path(cfg.dataset.args.audio_path)

    # Determine the number of samples to generate/process.
    # For Custom dataset, this is usually determined by the number of audio files.
    # For pure generation, it's explicitly set.
    S = cfg.dataset.num_samples

    # Initialize dummy 'x' for the DataLoader.
    # This 'x' represents the initial noise for the diffusion process.
    # Its size (S) is determined by cfg.dataset.num_samples.
    # The actual content of 'x' might be ignored by the model's forward pass
    # when conditioned on a real waveform, but it's needed for DataLoader compatibility.
    x = torch.randn(S, 1, 640, 88)

    dataset = None # Initialize dataset variable

    # --- REVISED DATASET LOADING LOGIC ---
    # This section now prioritizes loading the Custom dataset if specified,
    # regardless of the specific `task.sampling.type` used for the model's internal method.
    if cfg.dataset.name == 'Custom':
        print("Loading custom audio for transcription...", flush=True)
        # Instantiate the Custom dataset
        dataset = Custom(**OmegaConf.to_container(cfg.dataset.args, resolve=True),
                         sample_rate=cfg.sampling_rate) # Pass sample_rate to Custom dataset

        # If custom_audio_dataset's __len__ returns 0, it means no files were found.
        # This warning is crucial for debugging.
        if len(dataset) == 0:
            warnings.warn(f"Custom dataset found 0 audio clips in {cfg.dataset.args.audio_path}. Please check the path and audio_ext.")

    elif cfg.task.sampling.type == 'inpainting_ddpm_x0':
        if cfg.dataset.name in ['MAESTRO', 'MAPS']:
            dataset_class = getattr(MusicDataset, cfg.dataset.name)
            dataset = dataset_class(**OmegaConf.to_container(cfg.dataset.args, resolve=True))
            waveform = torch.empty(S, cfg.dataset.args.sequence_length)
            roll_labels = torch.empty(S, 640, 88)
            for i in range(S):
                sample = dataset[i]
                waveform[i] = sample['audio']
                roll_labels[i] = sample['frame']
            dataset = TensorDataset(x, waveform, roll_labels)
        else:
            raise ValueError(f"Inpainting task only supports MAESTRO, MAPS, or Custom datasets, but got {cfg.dataset.name}")

    elif cfg.task.sampling.type == 'generation_ddpm_x0':
        # This branch is for pure generation *not* from custom audio files.
        # If Custom dataset is used, it should be handled by the 'Custom' branch above.
        if cfg.dataset.name != 'Custom':
            waveform = torch.randn(S, 327680) # Dummy waveform for pure generation
            dataset = TensorDataset(x, waveform)
        else:
            raise ValueError("Pure generation task (generation_ddpm_x0) should not be used with 'Custom' dataset unless 'Custom' is specifically designed to provide dummy data for generation.")
    else:
        # Fallback for unsupported task types or dataset combinations
        raise ValueError(f"Unsupported sampling task type: {cfg.task.sampling.type} or dataset: {cfg.dataset.name}")

    # Ensure dataset is not None before proceeding
    if dataset is None:
        raise ValueError("Dataset could not be initialized. Check your config and dataset logic.")

    # Warn if batch size is larger than the actual dataset size.
    # This is common if you have only a few custom audio clips.
    if len(dataset) < cfg.dataloader.batch_size:
        warnings.warn(f"Batch size ({cfg.dataloader.batch_size}) is larger than total number of audio clips ({len(dataset)}). Forcing batch size to {len(dataset)})")
        # Adjust batch size to prevent DataLoader errors if it's too large for the dataset
        cfg.dataloader.batch_size = max(1, len(dataset)) # Ensure batch size is at least 1 if dataset is not empty

    loader = DataLoader(dataset, **OmegaConf.to_container(cfg.dataloader, resolve=True))

    # Model
    model_class = ClassifierFreeDiffRoll

    # --- CRITICAL CHANGE: Create a NEW DictConfig for the model's internal use ---
    # The model's __init__ (via SpecRollDiffusion) expects `sampling.type` to be a method name
    # like 'generation_ddpm_x0' or 'inpainting_ddpm_x0', not 'transcription'.
    # We use 'transcription' as a high-level task identifier in sampling.py,
    # but the model needs the underlying diffusion method name.
    
    # Make a deep copy to ensure it's a separate OmegaConf object
    # and then modify its 'type' attribute.
    model_sampling_config_for_model = OmegaConf.create(OmegaConf.to_container(cfg.task.sampling, resolve=True))
    model_sampling_config_for_model.type = 'generation_ddpm_x0' # Use the appropriate diffusion method name

    if cfg.task.frame_threshold is not None:
        model = model_class.load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                                 sampling=model_sampling_config_for_model, # Pass the modified DictConfig
                                                 frame_threshold=cfg.task.frame_threshold,
                                                 generation_filter=cfg.task.generation_filter,
                                                 inpainting_t=cfg.task.inpainting_t,
                                                 inpainting_f=cfg.task.inpainting_f)
    else:
        model = model_class.load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                                 sampling=model_sampling_config_for_model, # Pass the modified DictConfig
                                                 generation_filter=cfg.task.generation_filter,
                                                 inpainting_t=cfg.task.inpainting_t,
                                                 inpainting_f=cfg.task.inpainting_f)

    # Logger setup
    name = f"Transcription-{model.__class__.__name__}-{cfg.model.args.kernel_size if 'kernel_size' in cfg.model.args else 'N/A'}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)

    # PyTorch Lightning Trainer setup
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True),
                         logger=logger)

    print(f"Starting prediction with model: {model.__class__.__name__} and checkpoint: {cfg.checkpoint_path}", flush=True)
    print(f"Configured for task type: {cfg.task.sampling.type} (High-level task)", flush=True)
    print(f"Model's internal diffusion type: {model_sampling_config_for_model.type}", flush=True)

    # --- MANUAL DATALOADER ITERATION TEST ---
    print(f"DataLoader has {len(loader.dataset)} items and batch size {loader.batch_size}", flush=True)
    print("Attempting to iterate DataLoader manually:", flush=True)
    items_yielded = 0
    try:
        for i, batch in enumerate(loader):
            print(f"  Batch {i} yielded. Batch type: {type(batch)}", flush=True)
            if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
                print(f"  First element shape: {batch[0].shape}", flush=True)
            else:
                print(f"  First element not a tensor or batch is empty.", flush=True)
            items_yielded += 1
            if items_yielded > 5: # Limit for debugging
                print("  (Stopping manual iteration after 5 batches for brevity)", flush=True)
                break
    except Exception as e:
        print(f"  Error during manual DataLoader iteration: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout) # Print full traceback to stdout
    print(f"Manual DataLoader iteration finished. Total items yielded: {items_yielded}", flush=True)

    # Now proceed with trainer.predict if manual iteration was successful
    if items_yielded > 0:
        print("Proceeding with trainer.predict...", flush=True)
        trainer.predict(model, loader)
    else:
        print("DataLoader yielded no items. Skipping trainer.predict.", flush=True)

if __name__ == "__main__":
    main()