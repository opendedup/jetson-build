import rclpy
from rclpy.node import Node
from chat_interfaces.msg import Chat

import traceback

import queue
import threading


from .services.stream import EnASRService
import numpy as np
import time
import pyaudio
from multiprocessing import Process, Queue, Event
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from whisper_trt.vad import load_vad
from .utils import find_audio_device_index
from .usb_4_mic_array import Tuning


def find_respeaker_audio_device_index():

    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")

    for i in range(num_devices):

        device_info = p.get_device_info_by_host_api_device_index(0, i)
        
        if "respeaker" in device_info.get("name").lower():

            device_index = i

    return device_index

@contextmanager
def get_respeaker_audio_stream(
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        bitwidth: int = 2
    ):

    if device_index is None:
        device_index = find_respeaker_audio_device_index()

    if device_index is None:
        raise RuntimeError("Could not find Respeaker device.")
    
    p = pyaudio.PyAudio()

    stream = p.open(
        rate=sample_rate,
        format=p.get_format_from_width(bitwidth),
        channels=channels,
        input=True,
        input_device_index=device_index
    )

    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()


def audio_numpy_from_bytes(audio_bytes: bytes):
    audio = np.fromstring(audio_bytes, dtype=np.int16)
    return audio


def audio_numpy_slice_channel(audio_numpy: np.ndarray, channel_index: int, 
                      num_channels: int = 6):
    return audio_numpy[channel_index::num_channels]


def audio_numpy_normalize(audio_numpy: np.ndarray):
    return audio_numpy.astype(np.float32) / 32768


@dataclass
class AudioChunk:
    audio_raw: bytes
    audio_numpy: np.ndarray
    audio_numpy_normalized: np.ndarray
    voice_prob: float | None = None


@dataclass
class AudioSegment:
    chunks: AudioChunk


class Microphone(Process):

    def __init__(self, 
                 output_queue: Queue, 
                 chunk_size: int = 3072,#1536,3072 
                 sound_device: str  = "respeaker",
                 use_channel: int = 0, 
                 num_channels: int = 1,
                 sample_rate: int = 16000):
        super().__init__()
        self.output_queue = output_queue
        self.chunk_size = chunk_size
        self.use_channel = use_channel
        self.num_channels = num_channels
        self.device_index = find_audio_device_index(sound_device)
        self.sample_rate = sample_rate

    def run(self):
        while True:
            with get_respeaker_audio_stream(sample_rate=self.sample_rate, 
                                            device_index=self.device_index, 
                                            channels=self.num_channels) as stream:
                while True:
                    try:
                        audio_raw = stream.read(self.chunk_size)
                        audio_numpy = audio_numpy_from_bytes(audio_raw)
                        audio_numpy = np.stack([audio_numpy_slice_channel(audio_numpy, i, self.num_channels) for i in range(self.num_channels)])
                        audio_numpy_normalized = audio_numpy_normalize(audio_numpy)

                        audio = AudioChunk(
                            audio_raw=audio_raw,
                            audio_numpy=audio_numpy,
                            audio_numpy_normalized=audio_numpy_normalized
                        )

                        self.output_queue.put(audio)
                    except:
                        print(traceback.format_exc())
                        break


class VAD(Process):

    def __init__(self,
            input_queue: Queue, 
            output_queue: Queue,
            sample_rate: int = 16000,
            use_channel: int = 0,
            speech_threshold: float = 0.6,
            max_filter_window: int = 1,
            trailing_silence_frames: int = 3,  # Number of silent frames to detect end of speech
            ready_flag = None,
            speech_start_flag = None,
            speech_end_flag = None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.use_channel = use_channel
        self.speech_threshold = speech_threshold
        self.max_filter_window = max_filter_window
        self.trailing_silence_frames = trailing_silence_frames
        self.ready_flag = ready_flag
        self.speech_start_flag = speech_start_flag
        self.speech_end_flag = speech_end_flag

    def run(self):

        vad = load_vad()
        # warmup run
        vad(np.zeros(1536, dtype=np.float32), sr=self.sample_rate)
        

        max_filter_window = deque(maxlen=self.max_filter_window)

        speech_chunks = []

        prev_is_voice = False
        silent_frame_count = 0 

        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:
            

            audio_chunk = self.input_queue.get()
            #print(f"""input_queue={self.input_queue.qsize()}""")

            voice_prob = float(vad(audio_chunk.audio_numpy_normalized[self.use_channel], sr=self.sample_rate).flatten()[0])
            chunk = AudioChunk(
                audio_raw=audio_chunk.audio_raw,
                audio_numpy=audio_chunk.audio_numpy,
                audio_numpy_normalized=audio_chunk.audio_numpy_normalized,
                voice_prob=voice_prob
            )

            max_filter_window.append(chunk)

            is_voice = any(c.voice_prob > self.speech_threshold for c in max_filter_window)
            
            if is_voice:
                speech_chunks.append(chunk)
                silent_frame_count = 0  # Reset silent frame count if voice is detected
                if self.speech_start_flag is not None and not prev_is_voice:
                    self.speech_start_flag.set()
            else:
                if speech_chunks: # if speech_chunks is not empty
                    silent_frame_count += 1
                    if silent_frame_count >= self.trailing_silence_frames:
                        segment = AudioSegment(chunks=speech_chunks)
                        self.output_queue.put(segment)
                        if self.speech_end_flag is not None:
                            self.speech_end_flag.set()
                        speech_chunks = []
                        silent_frame_count = 0
            prev_is_voice = is_voice


class StartEndMonitor(Process):

    def __init__(self, start_flag: Event, end_flag):
        super().__init__()
        self.start_flag = start_flag
        self.end_flag = end_flag

    def run(self):
        while True:
            self.start_flag.wait()
            self.start_flag.clear()
            #self.get_logger().debug(f"Speech started.")
            self.end_flag.wait()
            self.end_flag.clear()
            #self.get_logger().debug(f"Speech ended.")