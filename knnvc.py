import torch, torchaudio
from inference_eta_wavlm import SimpleEtaWavLMInference


knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
eta_wavlm_model = SimpleEtaWavLMInference("path_to_your_model")


src_wav_path = 'path_to_your_source_audio.wav'  # Replace with your source audio file path
ref_wav_paths = ['path_to_your_ref_audio.wav',]


results = eta_wavlm_model.process_audio(audio_path=src_wav_path)
query_seq = results['eta_features'] # this is eta features


matching_set = knn_vc.get_matching_set(ref_wav_paths) 



out_wav = knn_vc.match(query_seq, matching_set, topk=4)
torchaudio.save('knnvc2_out.wav', out_wav[None], 16000)
print('audio saved')