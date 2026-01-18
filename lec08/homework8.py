import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''
    raise RuntimeError("You need to change this part")

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
    raise RuntimeError("You need to change this part")

def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    stft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
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