# Eta-WavLM_implementation
Implementation of Eta-WavLM paper. I am using the Eta-WavLM Representations for the source utterance while inferencing from knn vc voice conversion model with litserve deployment code.

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
├── model.py                   --> Core Eta-WavLM implementation
├── train_eta_wavlm.py         --> Training script
├── inference_eta_wavlm.py     --> Full inference wrapper
├── simple_inference.py        --> Inference example to get features from an audio
├── inference_output/          --> Will contain features inferenced from the file simple_inference.py
├── pretrained_models/          --> This will store ECAPA-TDNN model automatically while starting to train 
├── knnvc.py                   --> kNN-VC inference with Eta-WavLm features of the source utterance
├── server.py                  --> LitServe Voice Conversion API server
├── client.py                  --> API client Example
├── models/                    --> Trained model files will be saved here (.pkl)
└── data/                      --> LibriSpeech dataset train-clean-100
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

