# extract audio features from a given audio file

import torch
import torchaudio
import torchaudio.transforms as transforms

# reads audio file/returns raw waveform amd sample_rate of loaded audio as torch tensor
def load_audio(audio_fp):
    # waveform shape: [channel, time] -> avg across channels -> channel=1
    waveform, sample_rate = torchaudio.load(audio_fp)
    waveform = (torch.sum(waveform, dim=0) / waveform.size(0)).unsqueeze(0)

    return waveform, sample_rate

# apply standard normalization to the given tensor
def normalize_features(features):
    mean = features.mean()
    sd = features.std()

    return (features - mean) / sd


# waveform shape: [channel, time]
# n_ffts -> sizes of FFT to use; default ~ 23ms, 46ms, 93ms
def extract_audio_feats(waveform, sample_rate, n_ffts=[1024, 2048, 4096], 
    hop_length=512, pad=0, n_mels=80):

    audio_feats = []
    
    for n_fft in n_ffts:
        # shape: [channel (1), n_mels, time]
        mel_specgram = transforms.MelSpectrogram(sample_rate=sample_rate,
            n_fft=n_fft, pad=pad, n_mels=n_mels, hop_length=hop_length)(waveform)
        mel_specgram = mel_specgram.squeeze(0)

        # scale outputs logarithmically [avoid -inf values by doing abs + adding 1e-16]
        log_mel_specgram = torch.log(torch.abs(mel_specgram) + 1e-16)

        audio_feats.append(mel_specgram)
    
    # shape: [n_mels, time, len(n_ffts)] ~ default: [80, ?, 3] -> [?, 80, 3] (post-transpose)
    audio_feats = torch.stack(audio_feats, dim=-1)
    audio_feats = audio_feats.transpose(0, 1)

    # standard normalization across frequency bands
    for timestep in range(audio_feats.size(0)):
        for freq_band in range(audio_feats.size(1)):
            audio_feats[timestep, freq_band] = normalize_features(audio_feats[timestep, freq_band])

    # transpose -> final shape: [3, ?, 80]
    audio_feats = audio_feats.transpose(0, 2).transpose(1, 2)

    return audio_feats