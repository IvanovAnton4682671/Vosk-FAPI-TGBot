
from fastapi import FastAPI, File, UploadFile
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import wave
import os


app = FastAPI()   #приложение, являющееся экземпляром класса FastAPI


def print_file_info(file_path: str) -> None:
    """
    Данная функция печатает информацию об аудиофайле: количество каналов, ширину выборки, частоту и общее количестов фрагментов.

    Args:
         file_path (Str): Путь к файлу, который проверяем.
    """
    with wave.open(file_path, 'rb') as wf:                                                                                                  #открытие .wav файла в бинарном формате для чтения
        print(f'Channels: {wf.getnchannels()}, Sample Width: {wf.getsampwidth()}, Frame Rates: {wf.getframerate()}, Frames: {wf.getnframes()}')   #получение всех данных


def convert_wav_to_text(audio_file_path: str) -> str:
    """
    Данная функция конвертирует любой входящий аудиофайл в .wav формат, после чего распознаёт его с помощью модели Vosk и возвращает либо текст, либо ошибку.

    Args:
        audio_file_path (Str): Получаемый аудиофайл любого формата.
    Returns:
        text_from_audio (Str): Данные файла в текстовом формате.
        error (Str): Какая-то ошибка.
    """
    file_extension = audio_file_path.split('.')[-1].lower()                                                           #получение формата файла
    temp_wav_path = 'temp.wav'                                                                                        #создание временного файла для обработки
    try:                                                                                                              #блок для отлова возможных ошибок
        audio = AudioSegment.from_file(audio_file_path, format=file_extension)                                        #получение данных из файла
        audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(16000)                                       #принудительная установка параметров
        audio.export(temp_wav_path, format='wav')                                                                     #конвертация в .wav формат
        print_file_info(temp_wav_path)                                                                                #проверяем параметры файла
        if not os.path.exists('model/vosk-model-ru-0.42'):                                                            #если не существует папка с моделью
            raise FileNotFoundError('Проверьте наличие обученной модели Vosk в папке model!')                         #то показываем ошибку
        model = Model('model/vosk-model-ru-0.42')                                                                     #получаем предобученную модель для распознавания
        recognizer = KaldiRecognizer(model, 16000)                                                              #объявление объекта-распознавателя
        with wave.open(temp_wav_path, 'rb') as wave_file:                                                       #открытие .wav файла в бинарном формате для чтения
            if wave_file.getnchannels() != 1 or wave_file.getsampwidth() != 2 or wave_file.getframerate() != 16000:   #если файл не прошёл проверку параметров
                raise ValueError('Проверьте файл, он должен быть в формате .wav с 1 каналом, 16-bit и 16000 Hz!')     #то показываем ошибку
            recognizer.AcceptWaveform(wave_file.readframes(wave_file.getnframes()))                                   #распознавание файла
            result = recognizer.Result()                                                                              #получение результата распознавания
        text_from_audio = result.split('"text" : ')[1].strip('} \n"')                                                 #форматируем вывод, а то там словарь на выходе
        return text_from_audio                                                                                        #возвращаем текст сообщения
    except Exception as e:                                                                                            #если какая-то шелуха
        return f'Возникла неожиданная ошибка: {e}'                                                                    #то показываем ошибку
    finally:                                                                                                          #этот блок выполняется в любом случае
        if os.path.exists(temp_wav_path):                                                                             #если есть временный файл
            os.remove(temp_wav_path)                                                                                  #то сносим его нафиг, чтобы место не занимал
        if os.path.exists(audio_file_path):                                                                           #проверяем наличие исходного файла
            os.remove(audio_file_path)                                                                                #удаляем исходный файл после обработки


@app.post('/recognize')                                            #декоратор для обработки POST-запроса по адресу
async def recognize_audio(file: UploadFile = File(...)) -> dict:   #данная аннотация говорит о том, что мы работаем с загружаемым файлом
    """
    Данная функция получает на вход аудиофайл, который пробует распознать, и возвращает текстовое представление аудиофайла.

    Args:
        file (UploadFile): Входной файл является загружаемым.
    Return:
        text (Dict): Возвращаем текстовую информацию в виде словаря.
    """
    file_location = f'temp_{file.filename}'                        #создаём временное имя для файла
    with open(file_location, 'wb') as file_object:                 #открываем файл для записи в бинарном режиме
        file_object.write(file.file.read())                        #записываем содержимое полученного файла
    text_from_audio = convert_wav_to_text(file_location)           #получаем текст после распознавания
    if os.path.exists(file_location):                              #проверка существования временного файла
        os.remove(file_location)                                   #удаление временного файла
    return {'text': text_from_audio}                               #возвращаем результат
