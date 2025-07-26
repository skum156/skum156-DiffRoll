import hydra
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
import logging
from typing import List, Any, Tuple
import time
from tqdm import tqdm

# --- IMPORTS FOR YOUR MODEL ---
import torch
import torchaudio
from model.unet import SpecUnet 

# Set up basic logging
log = logging.getLogger(__name__)

# --- CONFIGURATION ---
@dataclass
class ModelConfig:
    name: str = "SpecUnet"

@dataclass
class TrainerConfig:
    devices: int = 1
    accelerator: str = "cpu"
    # Keeping precision at 32 as Half precision caused issues on CPU
    precision: int = 32 

@dataclass
class InferenceConfig:
    # Shape of the spectrogram (Height, Width) for the UNet's 'x' input
    shape: Tuple[int, int] = (128, 16) # H, W
    batch_size: int = 1
    num_sampling_steps: int = 200

    # !!! IMPORTANT: These spec_args MUST match your model's training configuration !!!
    # If your SpecUnet was trained with different values, inference will likely fail.
    # Check your original training script/config for these values.
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 256
    n_mels: int = 229

@dataclass
class MainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    checkpoint_path: str = "/scratch/negishi/kumar809/skum156-DiffRoll/Pretrain_MAESTRO-retrain_Both-k=9.ckpt"

    # !!! IMPORTANT: This 'dim' MUST match the 'dim' argument used during training SpecUnet !!!
    # Common values are 64, 128, 256. Start by trying 64.
    # CHANGED: Adjusted model_dim to 32 to potentially match checkpoint's bottleneck channels
    model_dim: int = 32 # <--- CHANGED THIS LINE


# --- 2. Register configuration with ConfigStore ---
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)


# --- 3. Main function with @hydra.main ---
@hydra.main(config_path=None, config_name="config", version_base="1.1")
def main(cfg: MainConfig) -> None:
    log.info("DEBUG: Program started with the following resolved config:")
    log.info(OmegaConf.to_yaml(cfg))

    # Access parameters
    model_name = cfg.model.name
    checkpoint_path = cfg.checkpoint_path
    input_shape = cfg.inference.shape
    batch_size = cfg.inference.batch_size
    num_sampling_steps = cfg.inference.num_sampling_steps
    
    devices = cfg.trainer.devices
    accelerator = cfg.trainer.accelerator
    precision = cfg.trainer.precision
    
    model_dim = cfg.model_dim
    
    # Pack spectrogram arguments into a dictionary
    spec_args = {
        "sample_rate": cfg.inference.sample_rate,
        "n_fft": cfg.inference.n_fft,
        "hop_length": cfg.inference.hop_length,
        "n_mels": cfg.inference.n_mels,
    }

    log.info(f"DEBUG: Model Name: {model_name}")
    log.info(f"DEBUG: Checkpoint Path: {checkpoint_path}")
    log.info(f"DEBUG: Input Spectrogram Shape: {input_shape} (Batch Size: {batch_size})")
    log.info(f"DEBUG: Number of Sampling Steps: {num_sampling_steps}")
    log.info(f"DEBUG: Trainer: Accelerator={accelerator}, Devices={devices}, Precision={precision}")
    log.info(f"DEBUG: SpecUnet 'dim' parameter: {model_dim}")
    log.info(f"DEBUG: MelSpectrogram arguments: {spec_args}")


    # --- YOUR ACTUAL MODEL LOADING AND INFERENCE LOGIC ---
    try:
        device = torch.device(accelerator if torch.cuda.is_available() and accelerator == "gpu" else "cpu")
        log.info(f"DEBUG: Using device: {device}")

        # --- Inspect Checkpoint Hyperparameters ---
        log.info(f"DEBUG: Inspecting hyperparameters from checkpoint: {checkpoint_path}")
        try:
            ckpt_data = torch.load(checkpoint_path, map_location='cpu')
            if 'hyper_parameters' in ckpt_data:
                loaded_hparams = ckpt_data['hyper_parameters']
                log.info(f"DEBUG: Checkpoint Hyperparameters: {loaded_hparams}")
                # You can now look for 'dim', 'dim_mults', 'residual_channels', etc.
                # to confirm if model_dim=32 or if other params need adjustment.
            else:
                log.warning("WARNING: 'hyper_parameters' key not found in the checkpoint. Model might be saved differently.")
        except Exception as e:
            log.error(f"ERROR: Failed to load checkpoint to inspect hyperparameters: {e}")
        # --- End Checkpoint Inspection ---

        # --- Load the Model ---
        log.info(f"DEBUG: Attempting to load model '{model_name}' from: {checkpoint_path}")
        
        expected_output_channels = 3 

        model = SpecUnet.load_from_checkpoint(
            checkpoint_path=checkpoint_path, 
            dim=model_dim, # Will now be 32
            spec_args=spec_args,
            strict=False, 
            channels=3, 
            inference_frequency_dim=input_shape[1], 
            out_dim=expected_output_channels, 
            # If the checkpoint inspection above shows different 'dim_mults',
            # you might need to add them here, e.g., dim_mults=(1,2,4),
        )
        
        model.eval()
        model.to(device)
        log.info("DEBUG: Model loaded and set to evaluation mode successfully.")
        
        # --- Prepare Initial Input Tensors ---
        channels_for_x_input = 3 
        log.info(f"DEBUG: Preparing initial noisy input (x) of shape: ({batch_size}, {channels_for_x_input}, {input_shape[0]}, {input_shape[1]})")
        initial_input_x = torch.randn(batch_size, channels_for_x_input, input_shape[0], input_shape[1], device=device, dtype=torch.float32)
        
        # 2. Conditioning 'waveform' (raw audio)
        expected_time_steps = input_shape[0] 
        waveform_length = (expected_time_steps - 1) * spec_args['hop_length'] + spec_args['n_fft']

        log.info(f"DEBUG: Preparing conditioning waveform of shape: ({batch_size}, {waveform_length}) (sample_rate={spec_args['sample_rate']})")
        waveform_conditioning = torch.randn(batch_size, waveform_length, device=device, dtype=torch.float32) 
        
        log.info("DEBUG: Initial input tensors prepared.")


        # --- Run the Actual Sampling/Inference Loop ---
        log.info(f"DEBUG: Initiating actual sampling loop for {num_sampling_steps} steps.")
        
        generated_output = initial_input_x 

        with torch.no_grad():
            for step in tqdm(range(num_sampling_steps), desc="Sampling progress"):
                t = torch.tensor([step], device=device).long() 
                generated_output = model(generated_output, waveform_conditioning, t) 
                
        log.info("DEBUG: Actual sampling loop completed.")
        
        # --- Save the Output ---
        log.info("DEBUG: Detaching output tensor from graph and moving to CPU for saving.")
        import numpy as np
        output_filepath = "./outputs/generated_spectrogram.npy" 
        log.info(f"DEBUG: Saving generated spectrogram to: {output_filepath}")
        
        # Ensure the outputs directory exists
        import os
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        np.save(output_filepath, generated_output.cpu().numpy())


        log.info("DEBUG: Script execution finished successfully.")

    except ImportError as ie:
        log.error(f"ERROR: Model class import failed. Please check 'from model.unet import SpecUnet' and your file structure. Error: {ie}", exc_info=True)
        raise
    except FileNotFoundError as fnfe:
        log.error(f"ERROR: Model checkpoint not found at {checkpoint_path}. Error: {fnfe}", exc_info=True)
        raise
    except TypeError as te:
        log.error(f"ERROR: A TypeError occurred during model loading or inference. This usually means a function received an argument of the wrong type or missing arguments. Check 'dim', 'spec_args', and the arguments passed to model's forward. Error: {te}", exc_info=True)
        raise
    except Exception as e:
        log.error(f"ERROR: An unhandled error occurred during model loading or inference: {e}", exc_info=True)
        raise

# --- Entry point for the script ---
if __name__ == "__main__":
    main()