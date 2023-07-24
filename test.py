from gtts import gTTS
import pygame
import os

def speak_tts(text):
    tts = gTTS(text=text, lang='en')  # Substitua 'pt' pelo código do idioma desejado
    tts.save('temp.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load('temp.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.quit()
    os.remove('temp.mp3')

sla = 'assistant'
# Testar a fala
speak_tts(f'hello, i´m a alpha {sla}')
