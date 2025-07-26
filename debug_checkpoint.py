import torch
import os

checkpoint_path = '/scratch/negishi/kumar809/skum156-DiffRoll/Pretrain_MAESTRO-retrain_Both-k=9.ckpt'

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
else:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print("\n--- Keys in checkpoint: ---")
        print(checkpoint.keys())

        if 'hyper_parameters' in checkpoint:
            print("\n--- Hyperparameters (from 'hyper_parameters' key): ---")
            print(checkpoint['hyper_parameters'])
        elif 'hparams' in checkpoint: # For older PyTorch Lightning versions
            print("\n--- Hyperparameters (from 'hparams' key): ---")
            print(checkpoint['hparams'])
        else:
            print("\n--- No common 'hyper_parameters' or 'hparams' key found. ---")
            print("\n--- Available keys in checkpoint: ---")
            print(checkpoint.keys()) # Show what keys are available

    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
