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
        self.eta_wavlm = SimpleEtaWavLMInference("path_of_your_model")  # Replace with your model path

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        def decode_audio(b64str):
            audio_bytes = base64.b64decode(b64str)
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            return waveform, sr
        audio1, sr1 = decode_audio(request["audio1"])
        audio2, sr2 = decode_audio(request["audio2"])
        return {"audio1": (audio1, sr1), "audio2": (audio2, sr2)}

   
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
       
        (src_waveform, src_sr) = inputs["audio1"]
        (ref_waveform, ref_sr) = inputs["audio2"]

        # getting the matching set
        ref_path = "/tmp/ref.wav"
        torchaudio.save(ref_path, ref_waveform, ref_sr)
        matching_set = self.knn_vc.get_matching_set([ref_path])

        # getting the eta wavlm features for the source audio
        src_results = self.eta_wavlm.process_audio(
            waveform=src_waveform,
            sample_rate=src_sr,
            save_outputs=False
        )
        eta_src = src_results["eta_features"]

        
        # extracting the speaker embedding
        spk_emb_ref = self.eta_wavlm.eta_wavlm.extract_speaker_embedding(
            ref_waveform.squeeze(),  # ensure 1-D
            sample_rate=ref_sr
        )                                            

        #I am applying PCA to the reference speaker embedding
        d_ref = self.eta_wavlm.eta_wavlm.pca.transform(
            spk_emb_ref.cpu().numpy().reshape(1, -1)
        )[0]
        d_ref = torch.from_numpy(d_ref).float().to(self.device)

        A_star = self.eta_wavlm.eta_wavlm.A_star.to(self.device)   # 128 × 1024
        b_star = self.eta_wavlm.eta_wavlm.b_star.to(self.device)   # 1024
        #getting the speaker component from ref audio
        speaker_comp_ref = d_ref @ A_star + b_star                 # 1024 this checking in the end

        # Adding the ref speaker component to the source eta wavlm features
        K = eta_src.shape[0]            
        query_seq = (eta_src.to(self.device) +
                    speaker_comp_ref.unsqueeze(0).expand(K, -1))  #K×1024

        # Now i am matching in the similar feature space
        out_wav = self.knn_vc.match(query_seq, matching_set, topk=4)

        
        buf = io.BytesIO()
        torchaudio.save(buf, out_wav.unsqueeze(0), 16000, format="wav")
        buf.seek(0)
        out_b64 = base64.b64encode(buf.read()).decode("utf-8")

        #lastly checking all the shapes are correct
        print('------to check--------')
        print(f"speker_comp_ref shape: {speaker_comp_ref.shape}")
        print(f"query_seq shape: {query_seq.shape}")
        print(f"speaker embedding reference shape: {spk_emb_ref.shape}")
        return {"output_wav_b64": out_b64}


    
    
if __name__ == "__main__":
    api = AudioInferenceAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)
