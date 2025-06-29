import requests
import base64

def run_inference(server_url, src_wav_path, ref_wav_path):
    with open(src_wav_path, "rb") as f:
        src_wav_data = base64.b64encode(f.read()).decode('utf-8')
    
    with open(ref_wav_path, "rb") as f:
        ref_wav_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "audio1": src_wav_data,
        "audio2": ref_wav_data
    }
    response = requests.post(f"{server_url}/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        audio = result['output_wav_b64']
        audio_bytes = base64.b64decode(audio)
        output_path = "replace_to_your_desired_output_path"  # Replace with your desired output path
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f'Output saved to {output_path}')
    else:
        print("Request failed:", response.text)

if __name__ == "__main__":
    server_url = "http://localhost:8000"
    src_wav_path = "path_to_your_source_audio.wav"  # Replace with your source audio file path
    ref_wav_path = "path_to_your_reference_audio.wav"  # Replace with your reference audio file path
    run_inference(server_url, src_wav_path, ref_wav_path)