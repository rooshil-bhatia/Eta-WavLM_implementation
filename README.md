# Eta_WavLM_implementation
Implementation of Eta-WavLM paper to get speaker independent features from WavLm features. I am using the Eta-WavLM Representations for the source utterance while inferencing from knn vc voice conversion model with litserve deployment code.
![Alt Text](https://github.com/rooshil-bhatia/Eta_WavLM_implementation/blob/main/knnvc_eta.jpeg)

## Setting Up the Environment
1) To clone the Repository
```bash
git clone https://github.com/rooshil-bhatia/Eta_WavLM_implementation.git
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

## Litserve Deployment Details
1) To start the server. This will run the server `http://localhost:8000`.

```bash
python server.py
```

The input format is
```
{
  "audio1": "base64_encoded_source_audio_wav",
  "audio2": "base64_encoded_reference_audio_wav"
}

```
Field Descriptions:

audio1: Source speech audio (the content you want to convert, the source speech audio will go through Eta-WavLm to get speaker independent features.)

audio2: Reference speech audio (the target speaker voice characteristics)

Format: WAV files encoded as base64 strings. You can just give the path of the wav file and it will be automatically encoded.

Requirements: Any sample rate (auto-resampled to 16kHz), mono or stereo


The output format is 

```
{
  "output_wav_b64": "base64_encoded_converted_audio_wav"
}

```

This is the voice converted audio and it will be decoded at the client side and will get saved at the desired path.

2) To infer from the running server there is an example `client.py` file in which you can add path to the wav files in `src_wav_path= path_to_your_wav_file` and `ref_wav_path=path_to_your_wav_file` and you will get a voice converted wav file output saved at your desired path which you can set in file.


## Thought Process & Understanding

1) Eta-WavLm paper has proposed that speaker specific characteristics in WavLM representations can be removed through a linear transformation.
   
2) The training process reduces to solving a well-defined linear regression problem: S = D^T A + 1_N b^T, where the goal is to learn parameters A* and b* that best explain WavLM features in terms of PCA-reduced speaker embeddings.
   
3) This formulation transforms speaker identity removal from a complex neural network problem into a tractable linear algebra problem.
   
4) After a bit of research and reading results of the paper [Comparing the Moore-Penrose Pseudoinverse and Gradient Descent for Solving Linear Regression Problems: A Performance Analysis](https://arxiv.org/abs/2505.23552), I concluded that Pseudoinverse would be a faster, better and efficient way to solve the problem.
   
5) Choosing WavLM over HuBERT is a good choice as it is trained on a larger dataset is more robust to noise and can handle speech overlaps.

6) Tried to do some analysis between speaker independent and speaker dependent features to see effect of traning and change in the nature of the embeddings. Unlike the paper where they improved the performance of a voice conversion system to prove their research.

## Trade Offs

1) Used LibriSpeech `train-clean-100` subset of the data due to storage issues.

2) The paper states the their model has been trained on 1000 hours of data which might have seen more speakers and variablity than my configuration as I used 200 files to train my Eta_WavLM model.

3) The model architecture for the Voice Conversion that they used has been given in the paper but rather than training it from scratch I used pre trained Knn -vc checkpoint and replaced the source utterance WavLM features with Eta_WavLM features to test the voice conversion quality and make the whole pipeline working hence making a proof of concept.

## Future Plans

1) To train Eta_WavLM on bigger dataset.
2) To train an end to end voice conversion model using Eta_WavLM representations from scratch for getting the best output.
