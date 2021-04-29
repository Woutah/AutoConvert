"""
Preprocess the VCTK dataset:
Selects mic1 and downsamples to 16000 Hz 
"""
import argparse
import os
import soundfile as sf
import librosa

import numpy as np

np.random.seed(42)

# Parse arguments
parser = argparse.ArgumentParser(description='Preprocess VCTK dataset')
parser.add_argument("input", default=None,
                    help="Source speaker folder")
parser.add_argument("output", default=None,
                    help="Source speaker folder")
args = parser.parse_args()



test_split = 0.1
path = args.input
out_path = args.output
train_path = os.path.join(out_path, "train")
test_path = os.path.join(out_path, "test")
text_dir = os.path.join(path, "txt")
audio_dir = os.path.join(path, "wav48_silence_trimmed")
mic = "mic1"


os.mkdir(args.output)
os.mkdir(train_path)
os.mkdir(test_path)

speaker_ids = sorted(os.listdir(text_dir))

for speaker_id in speaker_ids:
    print(f"Processing speaker {speaker_id}")
    
    out_speaker_train_path = os.path.join(train_path, speaker_id)
    out_speaker_test_path = os.path.join(test_path, speaker_id)
    
    if not os.path.exists(out_speaker_train_path):
        os.mkdir(out_speaker_train_path)
    
    if not os.path.exists(out_speaker_test_path):
        os.mkdir(out_speaker_test_path)
    
    utterance_dir = os.path.join(audio_dir, speaker_id)
    
    utterance_list = sorted(f for f in os.listdir(utterance_dir) if f.endswith(f"{mic}.flac"))
    utterance_list = np.random.permutation(utterance_list) # randomly permute file names
    for i, utterance_file in enumerate(utterance_list):
        if i > len(utterance_list) - int(test_split * len(utterance_list)):
            out = os.path.join(out_speaker_test_path, utterance_file[:-10] + ".wav")
        else:
            out = os.path.join(out_speaker_train_path, utterance_file[:-10] + ".wav")
        
        file_path = os.path.join(utterance_dir, utterance_file)
        
        data, sr = librosa.load(file_path, dtype="float32", sr=16000)

        sf.write(out, data, 16000)
        
        
        