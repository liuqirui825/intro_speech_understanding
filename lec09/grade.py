import unittest, homework9
import numpy as np

np.random.seed(0)
Fs = 8000

def generate_vowel(fundamental, formants, duration=0.3):
    t = np.arange(0, duration, 1/Fs)
    signal = np.zeros_like(t)
    for formant in formants:
        for harmonic in range(1, 6):
            freq = fundamental * harmonic
            if abs(freq - formant) < 100:
                signal += 0.5 * np.sin(2 * np.pi * freq * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal / np.max(np.abs(signal))

speech = np.array([])
for vowel_params in [(200, [800, 1200]), (250, [400, 2000]), (180, [600, 1000]), (220, [500, 1500]), (190, [700, 1100])]:
    segment = generate_vowel(*vowel_params, duration=0.3)
    silence = np.zeros(int(0.1 * Fs))
    speech = np.concatenate([speech, segment, silence])

testspeech = np.array([])
test_pattern = [(250, [400, 2000]), (200, [800, 1200]), (180, [600, 1000]), 
                (220, [500, 1500]), (190, [700, 1100]), (200, [800, 1200]), (250, [400, 2000])]

for vowel_params in test_pattern:
    segment = generate_vowel(*vowel_params, duration=0.3)
    silence = np.zeros(int(0.05 * Fs))
    testspeech = np.concatenate([testspeech, segment, silence])

labels = ['a', 'i', 'u', 'e', 'o']

class Test(unittest.TestCase):
    def test_VAD(self):
        segments = homework9.VAD(speech, Fs)
        self.assertEqual(len(segments),5,'there should be five segments')
        self.assertGreater(len(segments[0]), 500, 'segments[0] should be more than 500 samples long!')

    def test_segments_to_models(self):
        frame_length = int(0.025*Fs)
        step = int(0.01*Fs)
        segments = homework9.VAD(speech, Fs)
        models = homework9.segments_to_models(segments, Fs)
        
        self.assertEqual(len(models), len(segments), 'there should be as many models as segment')
        N = int(0.004*Fs)
        self.assertEqual(len(models[0]), int(N/2), 'models[0] should have length %d'%(int(N/2)))
        
    def test_recognize_speech(self):
        segments = homework9.VAD(speech, Fs)
        models = homework9.segments_to_models(segments, Fs)
        sims, test_output = homework9.recognize_speech(testspeech, Fs, models, labels)  
        print(f"sims.shape: {sims.shape}") 
        print(f"len(test_output): {len(test_output)}")  
        self.assertEqual(sims.shape, (5,7), 'sims.shape should be (5,7)')
        self.assertGreater(sims[1,0], sims[0,0], 'sims[1,0] should be greater than sims[0,0]')
        self.assertEqual(len(test_output), 7, 'test_output should have length 7')
        self.assertEqual(test_output[0], 'i', 'test_output[0] should be i')
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)

n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))