import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
from model import EtaWavLMTransform

class SimpleEtaWavLMInference:
    """Simple inference wrapper for Eta-WavLM model"""
    
    def __init__(self, model_path):
        """Initialize inference system"""
        print("Loading Eta-WavLM model with WavLM + ECAPA-TDNN...")
        self.eta_wavlm = EtaWavLMTransform(
            wavlm_model_name="microsoft/wavlm-large",
            speaker_encoder_name="speechbrain/spkrec-ecapa-voxceleb",
            pca_components=128,
            layer_idx=15
        )
        self.eta_wavlm.load_model(model_path)
        print(f"Loaded Eta-WavLM model from {model_path}")
        
    def process_audio(self, audio_path=None,waveform=None, sample_rate=16000, save_outputs=True, output_dir="./inference_output"):
        """
        Process a single audio file and extract speaker-independent features, original wavlm features, speaker dependent features and speaker embeddings.
        
        """
        if audio_path is not None:
            audio_path = Path(audio_path)
            print(f"Loading audio: {audio_path}")
            waveform, sample_rate = torchaudio.load(audio_path)
        if waveform is not None and sample_rate is not None:
            waveform=waveform
            sample_rate=sample_rate
        
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)
            
        print(f"Audio duration: {len(waveform) / sample_rate:.2f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
        
        # Extract original WavLM features (with speaker info)
        print("Extracting WavLM features...")
        original_features = self.eta_wavlm.extract_wavlm_features(waveform, sample_rate)
        
        # Extract ECAPA-TDNN speaker embedding
        print("Extracting ECAPA-TDNN speaker embedding...")
        speaker_embedding = self.eta_wavlm.extract_speaker_embedding(waveform, sample_rate)
        
        # Apply Eta-WavLM transformation (remove speaker info)
        print("Applying Eta-WavLM transformation...")
        eta_features = self.eta_wavlm.transform(waveform, sample_rate)
        
        # Compute speaker component that was removed
        d = self.eta_wavlm.pca.transform(speaker_embedding.cpu().numpy().reshape(1, -1))[0]
        d = torch.from_numpy(d).float()
        speaker_component = d @ self.eta_wavlm.A_star.cpu() + self.eta_wavlm.b_star.cpu()
        
        # Prepare results
        results = {
            'audio_path': str(audio_path) if audio_path else 'audio',
            'duration_seconds': len(waveform) / sample_rate,
            'sequence_length': original_features.shape[0],
            'feature_dimension': original_features.shape[1],
            'speaker_embedding_dimension': speaker_embedding.shape[0],
            'original_features': original_features,
            'eta_features': eta_features,
            'speaker_embedding': speaker_embedding,
            'speaker_component': speaker_component,
        }
        
        # Compute analysis metrics
        analysis = self._analyze_speaker_removal(results)
        results['analysis'] = analysis
        
        # Save outputs if requested
        if save_outputs:
            self._save_outputs(results, output_dir, 'audio_output')
        
        return results
    
    def _analyze_speaker_removal(self, results):
        """Analyze the effectiveness of speaker identity removal"""
        original_features = results['original_features']
        eta_features = results['eta_features']
        speaker_component = results['speaker_component']
        speaker_embedding = results['speaker_embedding']
        
        # Compute statistics
        original_norm = torch.norm(original_features, dim=1)
        eta_norm = torch.norm(eta_features, dim=1)
        speaker_norm = torch.norm(speaker_component)
        speaker_emb_norm = torch.norm(speaker_embedding)
        
        # Compute cosine similarity between original and eta features
        cos_sim = torch.cosine_similarity(original_features, eta_features, dim=1)
        
        # Compute feature variance (measure of information retention)
        original_var = torch.var(original_features, dim=0).mean()
        eta_var = torch.var(eta_features, dim=0).mean()
        
        # Compute speaker component contribution
        speaker_contribution = speaker_norm / original_norm.mean()
        
        analysis = {
            'speaker_component_norm': float(speaker_norm),
            'speaker_embedding_norm': float(speaker_emb_norm),
            'speaker_contribution_ratio': float(speaker_contribution),
            'original_feature_norm_mean': float(original_norm.mean()),
            'eta_feature_norm_mean': float(eta_norm.mean()),
            'cosine_similarity_mean': float(cos_sim.mean()),
            'cosine_similarity_std': float(cos_sim.std()),
            'original_feature_variance': float(original_var),
            'eta_feature_variance': float(eta_var),
            'variance_retention_ratio': float(eta_var / original_var),
            'speaker_removal_effectiveness': float(1.0 - cos_sim.mean())
        }
        
        print("\n=== Eta-WavLM Speaker Removal Analysis ===")
        print(f"Speaker embedding norm (ECAPA-TDNN): {analysis['speaker_embedding_norm']:.4f}")
        print(f"Speaker component norm: {analysis['speaker_component_norm']:.4f}")
        print(f"Speaker contribution ratio: {analysis['speaker_contribution_ratio']:.4f}")
        print(f"Original features norm (mean): {analysis['original_feature_norm_mean']:.4f}")
        print(f"Eta features norm (mean): {analysis['eta_feature_norm_mean']:.4f}")
        print(f"Cosine similarity (mean ± std): {analysis['cosine_similarity_mean']:.4f} ± {analysis['cosine_similarity_std']:.4f}")
        print(f"Variance retention ratio: {analysis['variance_retention_ratio']:.4f}")
        print(f"Speaker removal effectiveness: {analysis['speaker_removal_effectiveness']:.4f}")
        
        return analysis
    
    def _save_outputs(self, results, output_dir, audio_name):
        """Save inference outputs to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save features as numpy arrays
        np.save(output_dir / f"{audio_name}_original_features.npy", 
               results['original_features'].cpu().numpy())
        np.save(output_dir / f"{audio_name}_eta_features.npy", 
               results['eta_features'].cpu().numpy())
        np.save(output_dir / f"{audio_name}_speaker_embedding.npy", 
               results['speaker_embedding'].cpu().numpy())
        np.save(output_dir / f"{audio_name}_speaker_component.npy", 
               results['speaker_component'].cpu().numpy())
        
        # Save metadata and analysis
        metadata = {
            'audio_path': results['audio_path'],
            'duration_seconds': results['duration_seconds'],
            'sequence_length': results['sequence_length'],
            'feature_dimension': results['feature_dimension'],
            'speaker_embedding_dimension': results['speaker_embedding_dimension'],
            'analysis': results['analysis'],
            'model_info': {
                'wavlm_model': 'microsoft/wavlm-large',
                'wavlm_layer': 15,
                'speaker_encoder': 'ECAPA-TDNN',
                'A_star_shape': list(self.eta_wavlm.A_star.shape),
                'b_star_shape': list(self.eta_wavlm.b_star.shape)
            }
        }
        
        with open(output_dir / f"{audio_name}_results.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Outputs saved to {output_dir}")

def main():
    """Simple command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Eta-WavLM Inference")
    parser.add_argument("--model", required=True, help="Path to trained Eta-WavLM model (.pkl)")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output_dir", default="./inference_output", help="Output directory")
    parser.add_argument("--no_save", action="store_true", help="Don't save output files")
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference = SimpleEtaWavLMInference(args.model)
    
    # Process audio file
    results = inference.process_audio(
        args.audio, 
        save_outputs=not args.no_save,
        output_dir=args.output_dir
    )
    
    print(f"\nProcessing completed for: {args.audio}")
    print(f"Eta features shape: {results['eta_features'].shape}")
    print(f"Speaker removal effectiveness: {results['analysis']['speaker_removal_effectiveness']:.4f}")

if __name__ == "__main__":
    main()
