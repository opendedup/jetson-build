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
from .utils import find_audio_device_index, adjust_angle
import torch
from transformers import Wav2Vec2ForSequenceClassification,Wav2Vec2FeatureExtractor
from .usb_4_mic_array import Tuning
import usb.core
import usb.util


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
                 chunk_size: int = 1536,#1536,3072 
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
            speech_threshold: float = 0.4,
            max_filter_window: int = 1,
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
            #print(f"""max_filter_window={len(max_filter_window)}""")

            is_voice = any(c.voice_prob > self.speech_threshold for c in max_filter_window)
            
            if is_voice > prev_is_voice:
                speech_chunks = [chunk for chunk in max_filter_window]
                # start voice
                speech_chunks.append(chunk)
                if self.speech_start_flag is not None:
                    self.speech_start_flag.set()
            elif is_voice < prev_is_voice:
                print(len(speech_chunks))
                # end voice
                segment = AudioSegment(chunks=speech_chunks)
                self.output_queue.put(segment)
                if self.speech_end_flag is not None:
                    self.speech_end_flag.set()
                speech_chunks = []
            elif is_voice:
                # continue voice
                speech_chunks.append(chunk)
            prev_is_voice = is_voice



class ASR(Process):

    def __init__(self, model: str, backend: str, input_queue,output_queue, use_channel: int = 0, ready_flag = None):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.backend = backend
        self.output_queue = output_queue
        self.model_name = "superb/wav2vec2-large-superb-sid"

    def run(self):
        
        if self.backend == "whisper_trt":
            from whisper_trt import load_trt_model
            model = load_trt_model(self.model)
        elif self.backend == "whisper":
            from whisper import load_model
            model = load_model(self.model)
        elif self.backend == "faster_whisper":
            from faster_whisper import WhisperModel
            class FasterWhisperWrapper:
                def __init__(self, model):
                    self.asr_options_template = {"hotwords": "Shimmy, Shimmel, Schimmel",
                                "language":"en","initial_prompt":"Shimmy Likes Cheese"}
                    self.model = model
                def transcribe(self, audio):
                    segs, info = self.model.transcribe(audio,**self.asr_options_template)
                    text = "".join([seg.text for seg in segs])
                    return {"text": text}
                
            model = FasterWhisperWrapper(WhisperModel(self.model))
            
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        embmodel = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name).to("cuda")
        
        # warmup
        model.transcribe(np.zeros(1536, dtype=np.float32))
        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:

            speech_segment = self.input_queue.get()

            t0 = time.perf_counter_ns()
            audio = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])

            text = model.transcribe(audio)['text']

            i= feature_extractor(audio, return_tensors="pt", sampling_rate=16000, padding=True).to("cuda")
            with torch.no_grad():
                o= embmodel(i.input_values, output_hidden_states=True)
            emb = o.logits.cpu().detach().numpy().tolist()[0]

            t1 = time.perf_counter_ns()
            if len(text) > 0:
                self.output_queue.put({
                    "text":text,
                    "emb":emb,
                    "time":(t1 - t0) / 1e9
                })


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


class WhisperASRPublisher(Node):

    def __init__(self):
        super().__init__('whisper_asr_publisher')
        self.publisher_ = self.create_publisher(Chat, 'asr', 10)
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        self.r_mic = Tuning(dev)
        t = threading.Thread(target=self.lworker)
        t.start()


    def lworker(self):
        try:
            audio_chunks = Queue()
            speech_segments = Queue()
            output_queue = Queue()
            vad_ready = Event()
            asr_ready = Event()
            speech_start = Event()
            speech_end = Event()


            asr = ASR("base.en", "whisper_trt", speech_segments,output_queue, ready_flag=asr_ready)

            vad = VAD(audio_chunks, speech_segments, max_filter_window=10, ready_flag=vad_ready, speech_start_flag=speech_start, speech_end_flag=speech_end)

            mic = Microphone(audio_chunks)
            mon = StartEndMonitor(speech_start, speech_end)

            vad.start()
            asr.start()
            mon.start()

            vad_ready.wait()
            asr_ready.wait()

            mic.start()
            while True:
                direction = adjust_angle(self.r_mic.direction)
                self.get_logger().info("Direction %d" % (direction))
                item = output_queue.get()
                msg = Chat()
                msg.chat_text = item["text"]
                msg.sid_embedding = item["emb"]
                msg.direction = direction
                self.publisher_.publish(msg)
                self.get_logger().info('Publishing: "%s"' % msg.chat_text)
        except:
            self.get_logger().error('%s' % traceback.format_exc())
        
        


def main(args=None):
    rclpy.init(args=args)
    asr_publisher = WhisperASRPublisher()
    

    rclpy.spin(asr_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    asr_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
