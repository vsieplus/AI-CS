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

# apply normalization to the given features tensor [timestep, channel]
def normalize_features(features):
    mean = features.mean()
    sd = features.std()

    if sd == 0:
        sd = 1

    return (features - features.mean()) / features.std()

# n_ffts -> sizes of FFT to use; default ~ 23ms, 46ms, 93ms
def extract_audio_feats(waveform, sample_rate, n_ffts=[1024, 2048, 4096], hop_length=512, pad=0, n_mels=80):
    audio_feats = []
    
    for n_fft in n_ffts:
        # shape: [channel (1), n_mels, time]
        mel_specgram = transforms.MelSpectrogram(sample_rate=sample_rate,
            n_fft=n_fft, pad=pad, n_mels=n_mels, hop_length=hop_length)(waveform)
        mel_specgram = mel_specgram.squeeze(0)

        # scale outputs logarithmically [avoid -inf values by doing abs + adding 1e-6]
        log_mel_specgram = torch.log(torch.abs(mel_specgram) + 1e-6)

        audio_feats.append(log_mel_specgram)
    
    # shape: [n_mels, time, len(n_ffts)] ~ default: [80, ?, 3]
    audio_feats = torch.stack(audio_feats, dim=-1)

    # standard normalization across frequency bands
    for freq_band in range(audio_feats.size(0)):
        audio_feats[freq_band] = normalize_features(audio_feats[freq_band])

    # transpose -> final shape: [3, ?, 80] (channel, timestep, frequency)
    audio_feats = audio_feats.transpose(0, 2)

    return audio_feats