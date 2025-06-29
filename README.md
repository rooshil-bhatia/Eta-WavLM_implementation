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

## File Structure

Eta_WavLM_implementation/
├── model.py                   --> Core Eta-WavLM implementation
├── train_eta_wavlm.py         --> Training script
├── inference_eta_wavlm.py     --> Full inference wrapper
├── simple_inference.py        --> Inference example to get features from an audio
├── knnvc.py                   --> kNN-VC inference with Eta-WavLm features of the source utterance
├── server.py                  --> LitServe Voice Conversion API server
├── client.py                  --> API client Example
├── models/                    # Trained model files will be saved here (.pkl)
└── data/                      # LibriSpeech dataset train-clean-100


