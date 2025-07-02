import torch
import torch.nn as nn
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.decomposition import PCA
import pickle
import torchaudio
from speechbrain.pretrained import EncoderClassifier

class EtaWavLMTransform:
    """
    Paper implenetation of,
    Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation
    """
    
    def __init__(self, 
                 wavlm_model_name="microsoft/wavlm-large",
                 speaker_encoder_name="speechbrain/spkrec-ecapa-voxceleb",
                 pca_components=128,
                 layer_idx=6):
        """
        Initialize Eta-WavLM transform
        
        Args:
            wavlm_model_name: HuggingFace WavLM model name
            speaker_encoder_name: SpeechBrain ECAPA-TDNN model name  
            pca_components: Number of PCA components for speaker embedding reduction
            layer_idx: WavLM layer to extract representations from (6th layer)
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.pca_components = pca_components
        self.layer_idx = layer_idx
        
        # Load pre-trained models
        print("Loading WavLM model...")
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
        self.wavlm_model = WavLMModel.from_pretrained(wavlm_model_name).to(self.device)
        self.wavlm_model.eval()
        
        print("Loading ECAPA-TDNN speaker encoder...")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source=speaker_encoder_name, 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        
        # Components to be learned during training
        self.A_star = None  # This is Latent basis matrix
        self.b_star = None  # This is Bias vector
        self.pca = None     # PCA transform for speaker embeddings
        self.is_trained = False
        
    def extract_wavlm_features(self, waveform, sample_rate=16000, inference=False):
        """Extract WavLM representations with dimension-safe preprocessing"""
        try:
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
            # Critical: Ensure proper tensor dimensions for WavLM
            waveform = self._prepare_wavlm_input(waveform, inference)
            
            if waveform is None:
                print('Error in waveform using dummy wavform.')
                return torch.zeros(100, 1024, device=self.device)
            
            # Use manual preprocessing to avoid processor dimension issues
            inputs = self._manual_audio_preprocessing(waveform)
            
            with torch.no_grad():
                outputs = self.wavlm_model(**inputs, output_hidden_states=True)
                features = outputs.hidden_states[self.layer_idx]
                
            return features.squeeze(0)  # [seq_len, 1024]
            
        except Exception as e:
            print(f"WavLM extraction error: {e}")
            return torch.zeros(100, 1024, device=self.device)

    def _prepare_wavlm_input(self, waveform, inference):
        """Prepare waveform to meet WavLM's strict dimension requirements"""
        if waveform is None or waveform.numel() == 0:
            return None
            
        # Ensure 1D tensor
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        if waveform.dim() != 1:
            return None
        
        if not inference:
            min_length = 16000   
            max_length = 96000 
            
            current_length = waveform.shape[0]
            
            if current_length < min_length:
                padding_needed = min_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
            elif current_length > max_length:
                waveform = waveform[:max_length]
            
            # Ensure length is compatible with WavLM's stride/kernel requirements
            target_length = ((waveform.shape[0] // 320) + 1) * 320
            if waveform.shape[0] < target_length:
                padding = target_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
            
            return waveform
        else:
            return waveform

    def _manual_audio_preprocessing(self, waveform):
        """Manual audio preprocessing that avoids dimension issues"""
        # Normalize audio
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-7)
        
        # WavLM expects input_values with batch dimension
        input_values = waveform.unsqueeze(0).to(self.device)
        
        return {"input_values": input_values}
    
    def extract_speaker_embedding(self, waveform, sample_rate=16000):
        """Extract ECAPA-TDNN speaker embedding with robust dimension handling"""
        try:
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Prepare waveform for SpeechBrain's ECAPA-TDNN
            waveform = self._prepare_speechbrain_input(waveform)
            
            if waveform is None:
                print('there is some error line using random ecapa embedding.')
                return torch.randn(192)
            
            # ECAPA-TDNN expects [batch_size, samples] format
            with torch.no_grad():
                embedding = self.speaker_encoder.encode_batch(waveform.unsqueeze(0))
            
            # Handle ECAPA-TDNN output dimensions robustly
            embedding = self._process_ecapa_output(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"ECAPA-TDNN speaker encoder error: {e}")
            # Return dummy embedding with correct dimensions
            return torch.randn(192)

    def _prepare_speechbrain_input(self, waveform):
        """Prepare waveform for SpeechBrain's ECAPA-TDNN requirements"""
        if waveform is None or waveform.numel() == 0:
            return None
        
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        
        if waveform.dim() != 1:
            return None
        
        # SpeechBrain's ECAPA-TDNN has specific length requirements for STFT
        min_length = 16000   # 1 seconds at 16kHz 
        max_length = 96000 # 6 seconds maximum
        
        current_length = waveform.shape[0]
        
        if current_length < min_length:
            padding_needed = min_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed), mode='constant', value=0)
        elif current_length > max_length:
            waveform = waveform[:max_length]
        
        # Ensure the length works well with STFT parameters
        hop_length = 160
        target_length = ((waveform.shape[0] // hop_length) + 1) * hop_length
        
        if waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
        
        return waveform

    def _process_ecapa_output(self, embedding):
        """Process ECAPA-TDNN output to ensure correct dimensions"""
        # Handle different possible output shapes from ECAPA-TDNN
        if embedding.dim() == 3:  # [batch, time, features] or [batch, 1, features]
            if embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)
            else:
                # Multiple time steps - take mean
                embedding = torch.mean(embedding, dim=1)
        
        if embedding.dim() == 2:  # [batch, features]
            embedding = embedding.squeeze(0)  # Remove batch dimension
        
        if embedding.dim() > 1:
            # Flatten if still multi-dimensional
            embedding = embedding.view(-1)
        
        # To check correct embedding size (192 for ECAPA-TDNN)
        if embedding.shape[0] != 192:
            if embedding.shape[0] < 192:
                # Pad to 192 dimensions
                print('padding to 192 dimensions line 209')
                padding = 192 - embedding.shape[0]
                embedding = torch.nn.functional.pad(embedding, (0, padding), mode='constant', value=0)
            else:
                # Reduce to 192 dimensions
                embedding = embedding[:192]
        
        return embedding
    
    def train_transform(self, dataset_loader, subsample_frames=50):
        """Train the linear transformation (A*, b*) on multi-speaker dataset"""
        print("Training Eta-WavLM linear transformation...")
        
        S_list = []  # SSL representations
        E_list = []  # Speaker embeddings
        
        for batch_idx, (waveform, sample_rate, speaker_id) in enumerate(dataset_loader):
            print(f"Processing batch {batch_idx + 1}/{len(dataset_loader)}")
            
            try:
                # Extract WavLM features
                wavlm_features = self.extract_wavlm_features(waveform, sample_rate, inference= False)
                seq_len, feature_dim = wavlm_features.shape
                
                # Randomly subsample L frames
                if seq_len > subsample_frames:
                    indices = torch.randperm(seq_len)[:subsample_frames]
                    wavlm_features = wavlm_features[indices]
                
                # Extract speaker embedding
                speaker_emb = self.extract_speaker_embedding(waveform, sample_rate)
                
                # Robust speaker embedding expansion
                speaker_emb_expanded = self._expand_speaker_embedding(speaker_emb, wavlm_features.shape[0])
                
                # Move to CPU for consistent computation
                S_list.append(wavlm_features.detach().cpu())
                E_list.append(speaker_emb_expanded.detach().cpu())
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if not S_list:
            raise ValueError("No valid data processed. Check your dataset and model setup.")
        
        # Concatenate all data on CPU
        S = torch.cat(S_list, dim=0)  # [N, Q] where N = U * L, Q = 1024 this is according to the paper 
        E = torch.cat(E_list, dim=0)  # [N, V] where V = 192 this is for speaker embedding
        
        print(f"Training data shape: S={S.shape}, E={E.shape}")
        
        # Apply PCA to speaker embeddings
        print("Applying PCA to speaker embeddings...")
        self.pca = PCA(n_components=self.pca_components) #the pca components are according to the paper.
        D = self.pca.fit_transform(E.numpy()) #the pca components are on the speaker embeddings.
        # print(self.pca.explained_variance_ratio_) # [N, P] where P = 128
        D = torch.from_numpy(D).float()
        
        # Solve linear system: S = D^T * A + 1_N * b^T
        N = D.shape[0]
        ones = torch.ones(N, 1)
        D_tilde = torch.cat([D, ones], dim=1)  # [N, P+1]
        
        # Solve using pseudo-inverse
        print("Solving linear system...")
        try:
            A_tilde_star = torch.linalg.pinv(D_tilde) @ S  # [P+1, Q]
        except Exception as e:
            print(f"Error in pseudo-inverse computation: {e}")
            U, s, Vt = torch.linalg.svd(D_tilde, full_matrices=False)
            s_inv = torch.where(s > 1e-10, 1.0 / s, torch.zeros_like(s))
            A_tilde_star = Vt.T @ torch.diag(s_inv) @ U.T @ S
        
        # Split A* and b* and move to device for inference
        self.A_star = A_tilde_star[:-1, :].to(self.device)  # [P, Q]
        self.b_star = A_tilde_star[-1, :].to(self.device)   # [Q]
        
        self.is_trained = True
        print("Training completed!")
        print(f"A* shape: {self.A_star.shape}")
        print(f"b* shape: {self.b_star.shape}")

    def _expand_speaker_embedding(self, speaker_emb, target_length):
        """Safely expand speaker embedding to target length"""
        try:
            # Ensure speaker_emb is 1D
            if speaker_emb.dim() == 0:
                speaker_emb = speaker_emb.unsqueeze(0)
            elif speaker_emb.dim() > 1:
                speaker_emb = speaker_emb.view(-1)
            
            # Expand to target length: [embedding_dim] -> [target_length, embedding_dim]
            speaker_emb_expanded = speaker_emb.unsqueeze(0).repeat(target_length, 1)
            
            return speaker_emb_expanded
            
        except Exception as e:
            print(f"Error expanding speaker embedding: {e}")
            # Fallback: create dummy expanded embedding
            return torch.randn(target_length, 192)
        
    def transform(self, waveform, sample_rate=16000):
        """Apply Eta-WavLM transformation to remove speaker identity"""
        if not self.is_trained:
            raise ValueError("Transform must be trained first!")
            
        # Extract WavLM features
        S = self.extract_wavlm_features(waveform, sample_rate, inference=True)  # [K, Q]
        
        # Extract and transform speaker embedding
        speaker_emb = self.extract_speaker_embedding(waveform, sample_rate)  # [V]
        
        # Handle device consistency for PCA transform
        speaker_emb_cpu = speaker_emb.detach().cpu().numpy()
        d = self.pca.transform(speaker_emb_cpu.reshape(1, -1))[0]  # [P]
        d = torch.from_numpy(d).float().to(self.device)
        
        # Apply transformation: η = S - 1_K(d^T A* + b*)
        K = S.shape[0]
        speaker_component = d @ self.A_star + self.b_star  # [Q]
        eta = S - speaker_component.unsqueeze(0).repeat(K, 1)  # [K, Q]
        
        return eta
    
    def save_model(self, path):
        """Save trained model components"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
            
        state = {
            'A_star': self.A_star,
            'b_star': self.b_star,
            'pca': self.pca,
            'pca_components': self.pca_components,
            'layer_idx': self.layer_idx,
            'model_info': {
                'wavlm_model_name': 'microsoft/wavlm-large',
                'speaker_encoder_name': 'speechbrain/spkrec-ecapa-voxceleb'
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load pre-trained model components"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        self.A_star = state['A_star']
        self.b_star = state['b_star'] 
        self.pca = state['pca']
        self.pca_components = state['pca_components']
        self.layer_idx = state['layer_idx']
        self.is_trained = True
        
        print(f"Model loaded from {path}")
        print(f"A* shape: {self.A_star.shape}")
        print(f"b* shape: {self.b_star.shape}")
