from gtts import gTTS
import os
import pyaudio
import wave
import pygame
from datetime import datetime
import webbrowser
import subprocess
import speech_recognition as sr

# Parameters
record_seconds = 10

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

client = OpenAI(api_key="")

with open("prompt.txt", 'r') as file:
    initial_prompt = file.read()

m=[
	 {"role": "system", "content": initial_prompt},
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

with open("index.txt", 'r') as file:
    i = int(file.read())

while True:
	def wait(keyword):
		recognizer = sr.Recognizer()
		recognizer.pause_threshold = 2
		with sr.Microphone() as source:
			recognizer.adjust_for_ambient_noise(source)
			try:
				audio = recognizer.listen(source, timeout=2, phrase_time_limit=1)
				command = recognizer.recognize_google(audio)
				if keyword.lower() in command.lower():
					return True
				else:
					return False
			except sr.WaitTimeoutError:
				return False
			except sr.UnknownValueError:
				return False
			except sr.RequestError:
				return False
	
	tick = 0

	while not wait("Hey George"): # Overly complicated period system
		tick += 1
		print('Available' + '.' * ((tick - 1) % 3 + 1) + ' ' * ((3 - tick) % 3), end='\r')

	play_sound("siri.mp3")

	receive_audio("input"+str(i)+".mp3")
	user_input = speech_to_text("input"+str(i)+".mp3")

	print("User: ", user_input)

	prompt = user_input

	response = chat_with_gpt(prompt)

	print("Assistant: ", response)

	if "<<<CALL:" in response:
		number = response[response.index("<<<CALL:") + 8 : response.index(">>>")]
		
		response = "Calling " + number
		text_to_speech(response, "response"+str(i)+".mp3")
		play_sound("response"+str(i)+".mp3")

		call_url = f'tel:{number}'
		webbrowser.open(call_url)
	
	elif "<<<TEXT:" in response:
		number = response[response.index("<<<TEXT:") + 8 : response.index(", message=")]
		message = response[response.index("message=\"") + 9 : response.index("\">>>")]

		response = "Messaging " + number + " with the message \"" + message + "\""
		text_to_speech(response, "response"+str(i)+".mp3")
		play_sound("response"+str(i)+".mp3")

		message_url = f'sms:{number}?body={message}'

		webbrowser.open(message_url)

	elif "<<<DATE>>>" in response or "<<<TIME>>>" in response:
		t = str(datetime.now()).split(" ")
		t[1] = t[1].split(".")[0]
		
		if "DATE" in response:
			response = "The date is " + t[0]
		else:
			response = "The time is " + t[1][:-3]

		text_to_speech(response, "response"+str(i)+".mp3")
		play_sound("response"+str(i)+".mp3")
	
	elif "<<<OPEN:" in response:
		url = response[response.index("<<<OPEN:") + 8 : response.index(">>>")]

		response = "Opening " + url
		text_to_speech(response, "response"+str(i)+".mp3")
		play_sound("response"+str(i)+".mp3")

		subprocess.run(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', url])
	
	elif "<<<GOOG:" in response:
		command = response[response.index("<<<GOOG:") + 8 : response.index(">>>")]
		url = "https://www.google.com/search?q=" + command

		response = "Opening a Google search for " + command
		text_to_speech(response, "response"+str(i)+".mp3")
		play_sound("response"+str(i)+".mp3")

		subprocess.run(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', url])

	else:
		text_to_speech(response, "response"+str(i)+".mp3")
		play_sound("response"+str(i)+".mp3")
	
	i += 1

	with open("index.txt", 'w') as file:
		file.write(str(i))