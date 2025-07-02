import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from model import EtaWavLMTransform

class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset loader with proper tensor handling for ECAPA-TDNN"""
    
    def __init__(self, librispeech_path, subset="dev-clean", max_files=1000, min_duration=1.0, max_duration=8.0):
        self.root_path = Path(librispeech_path) / "LibriSpeech" / subset
        self.audio_files = []
        self.speaker_ids = []
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Collect audio files with validation
        count = 0
        valid_files = 0
        for speaker_dir in self.root_path.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                    
                for audio_file in chapter_dir.glob("*.flac"):
                    # Validate audio file before adding
                    if self._validate_audio_file(audio_file):
                        self.audio_files.append(audio_file)
                        self.speaker_ids.append(speaker_id)
                        valid_files += 1
                    
                    count += 1
                    if valid_files >= max_files:
                        break
                if valid_files >= max_files:
                    break
            if valid_files >= max_files:
                break
        
        print(f"Found {count} total files, {valid_files} valid files loaded")
    
    def _validate_audio_file(self, audio_path):
        """Validate audio file to prevent dimension errors"""
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            # Filter by duration - important for ECAPA-TDNN STFT requirements
            if duration < self.min_duration or duration > self.max_duration:
                return False
                
            # Ensure file has valid properties
            return info.num_channels > 0 and info.sample_rate > 0 and info.num_frames > 8000
            
        except Exception:
            return False
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_ids[idx]
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Fix tensor dimensions for both WavLM and ECAPA-TDNN
            waveform = self._fix_audio_dimensions(waveform)
            
            if waveform is None or waveform.numel() == 0:
                return torch.zeros(16000), 16000, speaker_id
            
            return waveform, sample_rate, speaker_id
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(16000), 16000, speaker_id
    
    def _fix_audio_dimensions(self, waveform):
        """Fix audio dimensions for both WavLM and ECAPA-TDNN compatibility"""
        if waveform is None or waveform.numel() == 0:
            return None
            
        # Handle all dimension cases
        if waveform.dim() == 0:
            return None
        elif waveform.dim() == 1:
            pass  # Already correct
        elif waveform.dim() == 2:
            if waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            elif waveform.shape[1] == 1:
                waveform = waveform.squeeze(1)
            else:
                # Multi-channel: average all channels
                waveform = torch.mean(waveform, dim=0)
        else:
            # Higher dimensions: flatten
            waveform = waveform.view(-1)
        
        # Ensure minimum length for both WavLM and ECAPA-TDNN
        # ECAPA-TDNN needs at least 8000 samples for STFT
        min_samples = 8000  # 0.5 seconds at 16kHz
        if waveform.shape[0] < min_samples:
            padding = min_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
        
        # Limit maximum length to prevent memory issues
        max_samples = 128000  # 8 seconds
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]
        
        return waveform

def main():
    """Train Eta-WavLM transformation using WavLM + ECAPA-TDNN as per paper"""
    
    # Configuration
    librispeech_path = "./data"
    model_save_path = "./models/eta_wavlm_transform_2.pkl"
    batch_size = 1
    max_training_files = 200  # Adjust based on your hardware
    
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    # Download LibriSpeech if needed
    if not Path(librispeech_path).exists() or not any(Path(librispeech_path).iterdir()):
        print("Downloading LibriSpeech train-clean-100...")
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=librispeech_path,
            url="dev-clean", 
            download=True
        )
    
    # Initialize Eta-WavLM transform with WavLM + ECAPA-TDNN (as per paper)
    print("Initializing Eta-WavLM with WavLM-Large + ECAPA-TDNN...")
    eta_wavlm = EtaWavLMTransform(
        wavlm_model_name="microsoft/wavlm-large",
        speaker_encoder_name="speechbrain/spkrec-ecapa-voxceleb",
        pca_components=128,  # As used in paper
        layer_idx=6  # 6th layer of WavLM-Large as per paper
    )
    
    # Create dataset with validation
    dataset = LibriSpeechDataset(
        librispeech_path, 
        max_files=max_training_files,
        min_duration=1.0,  # Minimum 1 second for ECAPA-TDNN
        max_duration=6.0   # Maximum 6 seconds to prevent memory issues
    )
    
    if len(dataset) == 0:
        print("No valid audio files found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training on {len(dataset)} valid audio files...")
    print("Using WavLM-Large (6th layer) + ECAPA-TDNN as per Eta-WavLM paper")
    
    # Train the transform
    try:
        eta_wavlm.train_transform(dataloader, subsample_frames=50)#subsample is L as per the paper
        eta_wavlm.save_model(model_save_path)
        print(f"Training completed! Model saved to {model_save_path}")
        
        # Print model info
        print("\nModel Information:")
        print(f"WavLM Model: microsoft/wavlm-large (layer {eta_wavlm.layer_idx})")
        print(f"Speaker Encoder: ECAPA-TDNN")
        print(f"PCA Components: {eta_wavlm.pca_components}")
        print(f"A* matrix shape: {eta_wavlm.A_star.shape}")
        print(f"b* vector shape: {eta_wavlm.b_star.shape}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
