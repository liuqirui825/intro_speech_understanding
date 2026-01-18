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

def get_features(waveform, Fs):
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    num_freqs = frame_length // 2
    
    preemph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    frames = waveform_to_frames(preemph, frame_length, step)
    mstft = frames_to_mstft(frames)
    features = mstft_to_spectrogram(mstft)
    features = features[:, :num_freqs]
    
    vad_frame_length = int(0.025 * Fs)
    vad_step = int(0.01 * Fs)
    vad_frames = waveform_to_frames(waveform, vad_frame_length, vad_step)
    energy = np.sum(vad_frames**2, axis=1)
    threshold = 0.1 * np.max(energy)
    mask = energy > threshold
    
    labels = np.zeros(len(frames), dtype=int)
    current_label = 0
    
    for i in range(len(frames)):
        vad_idx = min(i * step // vad_step, len(mask)-1)
        if mask[vad_idx]:
            labels[i] = current_label
        else:
            if i > 0 and labels[i-1] == current_label:
                current_label += 1
    
    if labels[-1] == current_label:
        current_label += 1
    
    if current_label < 6:
        labels[:] = np.arange(len(labels)) % 6
        current_label = 6
    
    repeated_labels = []
    for i in range(len(labels)):
        repeated_labels.extend([labels[i]] * 5)
    
    repeated_labels = np.array(repeated_labels[:len(features)])
    
    return features, repeated_labels

def train_neuralnet(features, labels, iterations):
    input_dim = features.shape[1]
    output_dim = 6
    
    weights = np.random.randn(input_dim, output_dim) * 0.01
    bias = np.zeros(output_dim)
    
    lossvalues = np.zeros(iterations)
    
    for i in range(iterations):
        linear_output = np.dot(features, weights) + bias
        
        exp_scores = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        one_hot_labels = np.zeros_like(probs)
        one_hot_labels[np.arange(len(labels)), labels] = 1
        
        loss = -np.sum(np.log(probs[np.arange(len(labels)), labels] + 1e-8)) / len(labels)
        lossvalues[i] = loss
        
        grad = probs - one_hot_labels
        grad /= len(labels)
        
        dw = np.dot(features.T, grad)
        db = np.sum(grad, axis=0)
        
        weights -= 0.01 * dw
        bias -= 0.01 * db
    
    model = {'weights': weights, 'bias': bias}
    
    return model, lossvalues

def test_neuralnet(model, features):
    weights = model['weights']
    bias = model['bias']
    
    linear_output = np.dot(features, weights) + bias
    
    exp_scores = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return probabilities