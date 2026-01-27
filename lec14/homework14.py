import gtts, speech_recognition, librosa, soundfile

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    raise RuntimeError("You need to write this!")

def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
import os


def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognizer = sr.Recognizer()
    recognized_texts = []

    for text, lang, name in zip(texts, languages, filenames):
        mp3_file = name + ".mp3"
        wav_file = name + ".wav"

        # 1. Synthesize speech (MP3)
        synthesize(text, lang, mp3_file)

        # 2. Convert MP3 to WAV
        audio, sr_librosa = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, audio, sr_librosa)

        # 3. Speech recognition
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
            try:
                recognized_text = recognizer.recognize_google(audio_data, language=lang)
            except sr.UnknownValueError:
                recognized_text = ""
            except sr.RequestError:
                recognized_text = ""

        recognized_texts.append(recognized_text)

    return recognized_texts

