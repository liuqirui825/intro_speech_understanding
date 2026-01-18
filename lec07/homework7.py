import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    raise RuntimeError("You need to write this part")

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    raise RuntimeError("You need to write this part")

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
import numpy as np

def major_chord(f, Fs):
    T = 0.5
    N = int(Fs * T)
    n = np.arange(N)
    
    root_freq = f
    third_freq = f * (2 ** (4/12))
    fifth_freq = f * (2 ** (7/12))
    
    omega_root = 2 * np.pi * root_freq / Fs
    omega_third = 2 * np.pi * third_freq / Fs
    omega_fifth = 2 * np.pi * fifth_freq / Fs
    
    x_root = np.cos(omega_root * n)
    x_third = np.cos(omega_third * n)
    x_fifth = np.cos(omega_fifth * n)
    
    x = x_root + x_third + x_fifth
    return x

def dft_matrix(N):
    n = np.arange(N)
    k = n.reshape(N, 1)
    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    N = len(x)
    X = np.fft.fft(x)
    magnitude = np.abs(X[:N//2])
    indices = np.argsort(magnitude)[-3:]
    frequencies = indices * Fs / N
    return tuple(sorted(frequencies))
