import unittest, homework10
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
vowel_patterns = [
    (200, [800, 1200]), 
    (250, [400, 2000]),  
    (180, [600, 1000]), 
    (220, [500, 1500]),   
    (190, [700, 1100])  
]

for pattern in vowel_patterns:
    segment = generate_vowel(*pattern, duration=0.3)
    silence = np.zeros(int(0.1 * Fs))
    speech = np.concatenate([speech, segment, silence])

testspeech = np.array([])
test_pattern = [
    (250, [400, 2000]), 
    (200, [800, 1200]), 
    (180, [600, 1000]),  
    (220, [500, 1500]), 
    (190, [700, 1100]),
    (200, [800, 1200]),   
    (250, [400, 2000])  
]

for pattern in test_pattern:
    segment = generate_vowel(*pattern, duration=0.3)
    silence = np.zeros(int(0.05 * Fs))
    testspeech = np.concatenate([testspeech, segment, silence])

labels = ['a', 'i', 'u', 'e', 'o']

class Test(unittest.TestCase):
    def test_get_features(self):
        features, labels = homework10.get_features(speech, Fs)
        self.assertEqual(len(features.shape),2,'features should be a matrix')
        self.assertEqual(features.shape[0], len(labels), 'features and labels should have same number of frames')
        self.assertEqual(labels[0], 0, 'labels[0] should be 0')
        self.assertEqual(int(max(labels)), 5, 'max label should be 5')

    def test_train_neuralnet(self):
        features, labels = homework10.get_features(speech, Fs)
        model, lossvalues = homework10.train_neuralnet(features, labels, 1000)
        self.assertEqual(len(model), 2, 'model should have just 2 layers: LayerNorm and Linear')
        self.assertEqual(len(lossvalues), 1000, 'lossvalues should have length 1000')
        
    def test_test_neuralnet(self):
        features, labels = homework10.get_features(speech, Fs)
        model, lossvalues = homework10.train_neuralnet(features, labels, 1000)
        testfeatures, testlabels = homework10.get_features(testspeech, Fs)
        probabilities = homework10.test_neuralnet(model, testfeatures)
        self.assertEqual(probabilities.shape[0], len(testfeatures), 'probabilities and features should have same number of frames')
        self.assertEqual(probabilities.shape[1], 6, 'probabilities should have 6 columns, for 6 output classes')
        
suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
result = unittest.TextTestRunner().run(suite)
n_success = result.testsRun - len(result.errors) - len(result.failures)
print('%d successes out of %d tests run'%(n_success, result.testsRun))
print('Score: %d%%'%(int(100*(n_success/result.testsRun))))