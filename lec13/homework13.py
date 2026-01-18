import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    raise RuntimeError("You need to write this part!")

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    raise RuntimeError("You need to write this part!")

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    nframes = (len(speech) - frame_length) // frame_skip

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        a = librosa.lpc(frame, order=order)
        A[i, :] = a

        e = np.zeros(frame_length)
        for n in range(order, frame_length):
            pred = 0.0
            for k in range(1, order + 1):
                pred -= a[k] * frame[n - k]
            e[n] = frame[n] - pred

        excitation[i, :] = e

    return A, excitation


def synthesize(e, A, frame_skip):
    nframes, order_plus1 = A.shape
    order = order_plus1 - 1
    length = nframes * frame_skip

    synthesis = np.zeros(length)

    for i in range(nframes):
        a = A[i]
        for n in range(frame_skip):
            idx = i * frame_skip + n
            synthesis[idx] = e[idx]
            for k in range(1, order + 1):
                if idx - k >= 0:
                    synthesis[idx] -= a[k] * synthesis[idx - k]

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    nframes, frame_length = excitation.shape

    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        gain[i] = np.sqrt(np.sum(excitation[i, -frame_skip:] ** 2) / frame_skip)

        for n in range(frame_skip):
            if n % T0 == 0:
                e_robot[i * frame_skip + n] = gain[i]

    return gain, e_robot