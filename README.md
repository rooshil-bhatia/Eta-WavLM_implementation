# Eta-WavLM_implementation
Implementation of Eta-WavLM paper to get speaker independent features from WavLm features. I am using the Eta-WavLM Representations for the source utterance while inferencing from knn vc voice conversion model with litserve deployment code.
![Alt Text](https://github.com/rooshil-bhatia/Eta_WavLM_implementation/blob/main/knnvc_eta.jpeg)

## Setting Up the Environment
1) To clone the Repository
```bash
git clone https://github.com/rooshil-bhatia/Eta-WavLM_implementation.git
```
2) Create a Conda Environment with python 3.10>= and activate it.
```bash
conda create -n eta_wavlm python=3.10
```
```bash
conda activate eta_wavlm
```
3) Install all the dependencies.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
cd Eta_WavLM_implementation
```
```bash
pip install -r requirements.txt
```
## Folder Structure
```
Eta_WavLM_implementation/
├── data/                      --> LibriSpeech dataset train-clean-100
├── inference_output/          --> Will contain features inferenced from the file simple_inference.py\
├── models/                    --> Trained model files will be saved here (.pkl)
├── pretrained_models/         --> This will store ECAPA-TDNN model automatically while starting to train
├── client.py                  --> API client Example
├── inference_eta_wavlm.py     --> Full inference wrapper
├── knnvc.py                   --> kNN-VC inference with Eta-WavLm features of the source utterance
├── model.py                   --> Core Eta-WavLM implementation
├── server.py                  --> LitServe Voice Conversion API server
├── simple_inference.py        --> Inference example to get features from an audio
└── train_eta_wavlm.py         --> Training script
```
## Training 
After running
```bash
python train_eta_wavlm.py
```
This will:

1)Download LibriSpeech train-clean-100 dataset automatically to ./data (~5.95GB)

2)Extract WavLM features from the 15th layer (1024 dimensions)

3)Extract ECAPA-TDNN embeddings for all utterances and apply PCA reduction (192 → 128 dims)

4)Learn linear transformation (A*, b*) using pseudo-inverse solution (Moore-Penrose inverse)

5)Save trained model to ./models/eta_wavlm_transform.pkl

### Training Configuration
The training script uses these paper-compliant settings:
- WavLM Model: microsoft/wavlm-large (15th layer)
- Speaker Encoder: ECAPA-TDNN from SpeechBrain
- PCA Components: 128 (optimal per paper's ablation)
- Subsample Frames('L' in the paper): 50 frames per utterance (adjustable)
- Audio Duration: 1-6 seconds (just a filter can be adjusted)
- Max Training Files: 200 LibriSpeech utterances (adjustable)


## Inferencing

After running
```bash
python simple_inference.py
```
Make sure to load the Eta-WavLm checkpoint

This will Give 5 files in ./inference_output:
- audio_output_eta_features.npy
- audio_output_original_features.npy
- audio_output_results.json
- audio_output_speaker_component.npy
- audio_output_speaker_embedding.npy

The content for `audio_output_results.json` will look like:
<details>
<summary>Click to expand JSON output</summary>

<br>

<pre>
<code>{
  "audio_path": "/speech/suma/rooshil/sample1.wav",
  "duration_seconds": 7.25,
  "sequence_length": 300,
  "feature_dimension": 1024,
  "speaker_embedding_dimension": 192,
  "analysis": {
    "speaker_component_norm": 228.8349,
    "speaker_embedding_norm": 322.0620,
    "speaker_contribution_ratio": 0.8512,
    "original_feature_norm_mean": 268.8370,
    "eta_feature_norm_mean": 249.3773,
    "cosine_similarity_mean": 0.5927,
    "cosine_similarity_std": 0.1325,
    "original_feature_variance": 50.2814,
    "eta_feature_variance": 50.2814,
    "variance_retention_ratio": 1.0,
    "speaker_removal_effectiveness": 0.4073
  },
  "model_info": {
    "wavlm_model": "microsoft/wavlm-large",
    "wavlm_layer": 15,
    "speaker_encoder": "ECAPA-TDNN",
    "A_star_shape": [128, 1024],
    "b_star_shape": [1024]
  }
}
</code>
</pre>

</details>


