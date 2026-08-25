import os
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from pydub import AudioSegment
from scipy.io import wavfile
from PIL import ImageGrab
import pyttsx3
from google import genai
from google.genai import types

# --- Configuration Settings ---
TRIGGER_WORD = "debug" 
SAMPLE_RATE = 16000 
DURATION = 5 
AUDIO_FILE = "voice_command.wav"
SCREENSHOT_FILE = "screen_context.png"
AUDIO_PATH = AUDIO_FILE

client = genai.Client()

# This initializes the text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 165)

def print_ty_audio(audio_path):
    sound = AudioSegment.export(audio_path, format="wav")
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try: 
        text = recognizer.recognize_google(audio_data)
        print(f"Spoke: {text}")

    except sr.UnknownValueError:
        print(f"Sorry, could not uderstand")

    except sr.RequestError as e:
        print(f"API Error: {e}")

    return audio_path
    

def speak(text):
    print(f"\n[Assistant]: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def record_audio(filename, duration):
    print(f"🎙️ Listening for {duration} seconds... Speak your bug context.")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, SAMPLE_RATE, audio_data)

def capture_screen(filename):
    print("Snapshot captured; of your screen layout.")
    screenshot = ImageGrab.grab()
    screenshot.save(filename)

def analyze_with_gemini(audio_path, image_path):
    print("Processing context with Gemini Flash...")

    
    audio_file_ref = client.files.upload(file=audio_path)
    image_file_ref = client.files.upload(file=image_path)
    
    prompt = (
        "You are an expert pair-programmer helping a blind or eyes-free developer. "
        "Analyze the provided screenshot (which contains their code, editor, or terminal error) "
        "and listen to their audio question. Give a highly concise, 2-3 sentence answer "
        "explaining exactly what the bug is and how to fix it line-by-line. "
        "Do not output code blocks unless tiny; describe changes in plain, spoken words."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[audio_file_ref, image_file_ref, prompt]
    )
    
    # Deletes files after processing
    client.files.delete(name=audio_file_ref.name)
    client.files.delete(name=image_file_ref.name)
    
    return response.text

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    speak("Gedy's code reviewer initialized. Press Enter to simulate voice listener.")
    
    while True:
        try:
            # For a basic prototype, we use a keypress to trigger.
            input("\n[Press Enter when you want to ask a debugging question...]")
            
            capture_screen(SCREENSHOT_FILE)

            
            record_audio(AUDIO_FILE, DURATION)

            print_ty_audio(AUDIO_PATH)
            

            solution = analyze_with_gemini(AUDIO_FILE, SCREENSHOT_FILE)
            

            speak(solution)
            
        except KeyboardInterrupt:
            speak("Shutting down developer assistant. Happy coding.")
            break
        except Exception as e:
            print(f"Error encountered: {e}")
            speak("Sorry, I encountered an issue processing that code frame, please try again.")

if __name__ == "__main__":
    main()
