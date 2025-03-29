import torch
import librosa
import librosa.display
import numpy as np
import argparse
import torch.nn as nn
import glob
import sys

def load_audio_with_skip(file_path, sr=22050, skip_seconds=10, min_remaining=35):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=sr)
    total_duration = len(audio) / sr
    # Check if skipping the first 20 seconds still leaves at least min_remaining seconds.
    if total_duration - skip_seconds >= min_remaining:
        audio = audio[int(skip_seconds * sr):]
    return audio, sr

# ---- CR2NN Model ----
class CR2NNModel(nn.Module):
    def __init__(self, num_classes=2, lr=1e-3):
        super(CR2NNModel, self).__init__()
        self.lr = lr
        
        # Convolutional feature extractor:
        # Input shape: (batch, 1, n_mels, time)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (batch, 32, n_mels, time)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),                        # (batch, 32, n_mels/2, time/2)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   # (batch, 64, n_mels/2, time/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))                         # (batch, 64, n_mels/4, time/4)
        )
        # For example, if n_mels=256 and fixed_length (time dimension) = 1400,
        # after pooling the output becomes roughly (batch, 64, 64, 350).
        # Averaging over the frequency dimension (n_mels/4) yields a sequence of length 350 with 64 features.
        
        # GRU: Now stacking two layers instead of one.
        # Using num_layers=2 applies dropout (if specified) between layers.
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.5)
        # The output of a bidirectional GRU with hidden_size=128 is 256 per time step.
        self.fc = nn.Linear(128 * 2, num_classes)  # Maps 256 features to num_classes

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        conv_out = self.conv(x)             # (batch, 64, n_mels/4, time/4)
        conv_out = conv_out.mean(dim=2)       # Average over the frequency axis: (batch, 64, time/4)
        conv_out = conv_out.permute(0, 2, 1)    # (batch, time/4, 64)
        gru_out, _ = self.gru(conv_out)       # (batch, time/4, 256)
        last_out = gru_out[:, -1, :]          # Use the output from the last time step: (batch, 256)
        logits = self.fc(last_out)            # (batch, num_classes)
        return logits
    
# Function to preprocess the audio file
def preprocess_audio(file_path, sr=22050, fixed_length=1400):
    # Load audio file
    # audio, _ = load_audio_with_skip(file_path, sr=sr, skip_seconds = 5)
    
    audio, _ = librosa.load(file_path, sr=sr)

    # Generate Mel spectrogram
    #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=4096,        # Higher frequency resolution
        hop_length=256,    # Higher time resolution
        n_mels=256,        # More Mel bins for finer frequency detail
        fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize the spectrogram
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    # Pad or truncate to fixed length
    if mel_spec_db.shape[1] < fixed_length:
        pad_width = fixed_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :fixed_length]

    # Add channel dimension and convert to PyTorch tensor
    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # Add channel dimension
    mel_spec_db = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return mel_spec_db

# Function to classify the audio file
def classify_audio(file_path, model, device):
    # Preprocess the audio file
    input_tensor = preprocess_audio(file_path).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    human_percent = probabilities[0,0].item() * 100
    ai_percent = probabilities[0,1].item() * 100

    # Map prediction to labels

    return {"Human" : human_percent, "AI" : ai_percent}

# Main function
if __name__ == "__main__":
    class Args:
        pass
    args = Args()

    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    directory_path = sys.argv[1]

    # Load the model
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CR2NNModel(num_classes=2)
    model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
    model.eval()
    device = "cpu"
    #model = CR2NNModel(num_classes=2)
    #checkpoint = torch.load(args.checkpoint, map_location=device)
    #model.load_state_dict(checkpoint["state_dict"])  # Adjust if using standard PyTorch checkpoint
    model = model.to(device)

    # List only .txt files
    audio_files = glob.glob(f"{directory_path}/*.mp3") + glob.glob(f"{directory_path}/*.wav")

    for file in audio_files:
        # Classify the audio file
        result = classify_audio(file, model, device)
        print(f"The audio file '{file}' is Human: {result['Human']:.2f}% AI: {result['AI']:.2f}%", flush=True)

