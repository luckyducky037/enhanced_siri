import warnings
import torchaudio
from gtts import gTTS
import os
import pyaudio
import wave
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
from transformers import pipeline
import pygame
import time

def play_sound(file_path):
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        print(f"Error playing sound: {e}")

    finally:
        pygame.mixer.quit()

memory = {}

def speech_to_text(filename):
  warnings.filterwarnings("ignore", message="Some weights of *")
  warnings.filterwarnings("ignore", message="You should *")
  model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
  processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

  ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

  audio_data, sr = torchaudio.load(filename)
  inputs = processor(audio_data[0].numpy(), sampling_rate=sr, return_tensors="pt")
  generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
  transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
  return transcription

def receive_audio(filename):
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 16000
  CHUNK = 1024
  RECORD_SECONDS = 5
  WAVE_OUTPUT_FILENAME = filename

  audio = pyaudio.PyAudio()

  stream = audio.open(format=FORMAT, channels=CHANNELS,
                  rate=RATE, input=True,
                  frames_per_buffer=CHUNK,
                  input_device_index=0,
                  output_device_index=0)

  print("Recording...")

  frames = []

  for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      data = stream.read(CHUNK)
      frames.append(data)

  print("Finished recording")

  stream.stop_stream()
  stream.close()
  audio.terminate()

  with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
      wf.setnchannels(CHANNELS)
      wf.setsampwidth(audio.get_sample_size(FORMAT))
      wf.setframerate(RATE)
      wf.writeframes(b''.join(frames))

import openai
openai.api_key = "sk-gWe7WmVvGvYhOIlIq5P34hkCNJdoG6dzT6K94CPGsxuAxulp"
openai.api_base = "https://api.goose.ai/v1"

def chat_with_gpt(prompt):
  engines = openai.Engine.list()
  for engine in engines.data:
    print(engine.id)

  completion = openai.Completion.create(
    engine="gpt-j-6b",
    prompt=prompt,
    max_tokens=160,
    stream=True)
  
  response = ""

  for c in completion:
    # print (c.choices[0].text, end = '')
    response = response+c.choices[0].text
  
  print("")

  return response

def chat_with_mistral(prompt, model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"):
    generator = pipeline("text-generation", model=model)
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'].strip()

"""
def update_memory(user_input, response):
  memory[user_input] = response

receive_audio("input.mp3")
user_input = speech_to_text("input.mp3")[0]
print("User: ", response)

response = chat_with_mistral(" ".join(sum(memory.values(), [])) + user_input)
print("Assistant: ", response)

def text_to_speech(text, filename):
  tts = gTTS(text=text, lang='en')
  tts.save(filename)
  os.system(filename)

text_to_speech(response, "response.mp3")
"""

def update_memory(user_input, response):
  memory[user_input] = response

for i in range(1):
  receive_audio("input"+str(i+1)+".mp3")
  user_input = speech_to_text("input"+str(i+1)+".mp3")[0]

  print("User: ", user_input)

  prompt = "Your input is: " + user_input

  response = chat_with_gpt(prompt)
  print("Assistant: ", response)

  def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    os.system(filename)

  text_to_speech(response, "response"+str(i+1)+".mp3")

  play_sound("response"+str(i+1)+".mp3")