import litserve as ls
import torch
import numpy as np
import torchaudio
from typing import Dict, Any
from inference_eta_wavlm import SimpleEtaWavLMInference
import base64
import io

class AudioInferenceAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')
        self.eta_wavlm = SimpleEtaWavLMInference("path_to_your_model")  # Replace with your model path

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        def decode_audio(b64str):
            audio_bytes = base64.b64decode(b64str)
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            return waveform, sr
        audio1, sr1 = decode_audio(request["audio1"])
        audio2, sr2 = decode_audio(request["audio2"])
        return {"audio1": (audio1, sr1), "audio2": (audio2, sr2)}

    def predict(self, inputs: Dict[str, Any]) -> Any:
        (src_waveform, src_sr) = inputs["audio1"]
        (ref_waveform, ref_sr) = inputs["audio2"]

        # Save reference audio to temp file for knn_vc
        ref_path = "/tmp/ref.wav"
        torchaudio.save(ref_path, ref_waveform, ref_sr)

        # Extract eta features from source
        results = self.eta_wavlm.process_audio(waveform=src_waveform, sample_rate=src_sr, save_outputs=False)
        query_seq = results['eta_features']

        # Get matching set from reference(s)
        matching_set = self.knn_vc.get_matching_set([ref_path])

        # Run KNN-VC
        out_wav = self.knn_vc.match(query_seq, matching_set, topk=4)

        
        buf = io.BytesIO()
        torchaudio.save(buf, out_wav[None], 16000, format="wav")
        buf.seek(0)
        out_wav_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"output_wav_b64": out_wav_b64}

    def encode_response(self, output: Any) -> Dict[str, Any]:
        return {
            "output_wav_b64": output["output_wav_b64"]
        }
    
    
if __name__ == "__main__":
    api = AudioInferenceAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)