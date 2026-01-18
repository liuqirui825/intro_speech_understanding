import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    raise RuntimeError("You need to change this part")

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    raise RuntimeError("You need to change this part")

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    N = len(waveform)
    num_frames = (N - frame_length) // step + 1
    frames = np.zeros((num_frames, frame_length))
    for m in range(num_frames):
        start = m * step
        frames[m, :] = waveform[start:start + frame_length]
    return frames

def frames_to_mstft(frames):
    N = frames.shape[1]
    stft = np.fft.fft(frames, axis=1)
    mstft = np.abs(stft[:, :N])
    return mstft

def mstft_to_spectrogram(mstft):
    max_val = np.amax(mstft)
    threshold = 0.001 * max_val
    clipped = np.maximum(threshold, mstft)
    spectrogram = 20 * np.log10(clipped)
    return spectrogram

def VAD(waveform, Fs):
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    frames = waveform_to_frames(waveform, frame_length, step)
    energy = np.sum(frames**2, axis=1)
    
    threshold = 0.1 * np.max(energy)
    mask = energy > threshold
    
    segments = []
    current_segment = []
    
    for m in range(len(mask)):
        if mask[m]:
            current_segment.append(m)
        elif current_segment:
            if len(current_segment) > 2:
                start_idx = current_segment[0]
                end_idx = current_segment[-1]
                start_sample = start_idx * step
                end_sample = min(end_idx * step + frame_length, len(waveform))
                segment = waveform[start_sample:end_sample]
                segments.append(segment)
            current_segment = []
    
    if current_segment and len(current_segment) > 2:
        start_idx = current_segment[0]
        end_idx = current_segment[-1]
        start_sample = start_idx * step
        end_sample = min(end_idx * step + frame_length, len(waveform))
        segment = waveform[start_sample:end_sample]
        segments.append(segment)
    
    return segments

def segments_to_models(segments, Fs):
    models = []
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    num_freqs = frame_length // 2
    
    for segment in segments:
        if len(segment) < frame_length:
            continue
        preemph = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
        frames = waveform_to_frames(preemph, frame_length, step)
        if len(frames) == 0:
            continue
        mstft = frames_to_mstft(frames)
        spectrogram = mstft_to_spectrogram(mstft)
        low_freq_spectra = spectrogram[:, :num_freqs]
        model = np.mean(low_freq_spectra, axis=0)
        models.append(model)
    
    return models

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def recognize_speech(testspeech, Fs, models, labels):
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    
    for y in range(Y):
        for k in range(K):
            if len(models[y]) == len(test_models[k]):
                sims[y, k] = cosine_similarity(models[y], test_models[k])
    
    test_outputs = []
    for k in range(K):
        best_model_idx = np.argmax(sims[:, k])
        test_outputs.append(labels[best_model_idx])
    
    return sims, test_outputs