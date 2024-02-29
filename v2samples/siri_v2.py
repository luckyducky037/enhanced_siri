from gtts import gTTS
import os
import pyaudio
import wave
import pygame

# Parameters
record_seconds = 5
sequence = 5

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

from openai import OpenAI

client = OpenAI(api_key="insert here")

m=[
	 {"role": "system", "content": "You are a helpful assistant that takes in audio and returns speech, like Siri."},
]

def chat_with_gpt(prompt):
	m.append({"role": "user", "content": prompt})
	
	completion = client.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=m,
	)
	
	response = completion.choices[0].message.content

	m.append({"role": "assistant", "content": response},)

	return response

def speech_to_text(filename):
	audio_file= open(filename, "rb")

	transcription = client.audio.transcriptions.create(
		model="whisper-1", 
		file=audio_file
	)

	return transcription.text

def receive_audio(filename):
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 16000
	CHUNK = 1024
	RECORD_SECONDS = record_seconds
	WAVE_OUTPUT_FILENAME = filename

	audio = pyaudio.PyAudio()

	stream = audio.open(format=FORMAT, channels=CHANNELS,
									rate=RATE, input=True,
									frames_per_buffer=CHUNK,
									input_device_index=0,
									output_device_index=0)

	print("Recording...")

	frames = []

	for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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

def text_to_speech(text, filename):
	tts = gTTS(text=text, lang='en')
	tts.save(filename)
	os.system(filename)

for i in range(sequence):
	receive_audio("input"+str(i+1)+".mp3")
	user_input = speech_to_text("input"+str(i+1)+".mp3")

	print("User: ", user_input)

	prompt = "Your input is: " + user_input

	response = chat_with_gpt(prompt)
	print("Assistant: ", response)

	text_to_speech(response, "response"+str(i+1)+".mp3")

	play_sound("response"+str(i+1)+".mp3")
