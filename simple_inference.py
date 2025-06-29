from inference_eta_wavlm import SimpleEtaWavLMInference

eta_wavlm_model = SimpleEtaWavLMInference("path_to_your_model")  # Replace with your model path

# Process a single audio file
results = eta_wavlm_model.process_audio(audio_path="path_to_your_audio.wav")

# Access the results
print(f"Original features shape: {results['original_features'].shape}")
print(f"Eta features shape: {results['eta_features'].shape}")
print(f"Speaker removal effectiveness: {results['analysis']['speaker_removal_effectiveness']:.4f}")

# Access individual components
eta_features = results['eta_features']  # Speaker-independent features
speaker_embedding = results['speaker_embedding']  # ECAPA-TDNN embedding
analysis = results['analysis']  # Detailed analysis metrics
