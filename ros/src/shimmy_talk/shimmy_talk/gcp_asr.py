import rclpy
from rclpy.node import Node
from chat_interfaces.msg import Chat

import traceback

import queue
import threading

import torch
import numpy as np
import time
from multiprocessing import Process, Queue, Event
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from .utils import adjust_angle
from transformers import Wav2Vec2ForSequenceClassification,Wav2Vec2FeatureExtractor
from .usb_4_mic_array import Tuning
import usb.core
import usb.util
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

import scipy.io.wavfile as wav
import io
from .whisper_asr import Microphone,VAD,StartEndMonitor




class GCPASR(Process):

    def __init__(self, input_queue,output_queue, use_channel: int = 0, ready_flag = None):
        super().__init__()
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.output_queue = output_queue
        self.model_name = "superb/wav2vec2-large-superb-sid"
        self.project_id = "lemmingsinthewind"
        self.config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="long",
        )
        self.client = SpeechClient()

    def run(self):
            
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        embmodel = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name).to("cuda")
        
        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:
            try:
                speech_segment = self.input_queue.get()
                raw_audio_bytes_array = bytearray()
                for chunk in speech_segment.chunks:
                    raw_audio_bytes_array.extend(chunk.audio_raw)
                audio_samples = np.frombuffer(raw_audio_bytes_array, dtype=np.int16)
                buffer = io.BytesIO()
                wav.write(buffer, 16000, audio_samples)
                t0 = time.perf_counter_ns()
                audio = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])
                request = cloud_speech.RecognizeRequest(
                    recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
                    config=self.config,
                    content=buffer.read(),
                )
                response = self.client.recognize(request=request)
                text = ""
                for result in response.results:
                    # The first alternative is the most likely one for this portion.
                    if len(result.alternatives) > 0:
                        text += result.alternatives[0].transcript
                
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
            except:
                print('Failed turning sound into text')
                print('%s' % traceback.format_exc())



class GCPASRPublisher(Node):

    def __init__(self):
        super().__init__('gcp_asr_publisher')
        self.publisher_ = self.create_publisher(Chat, 'asr', 10)
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        self.r_mic = Tuning(dev)
        t = threading.Thread(target=self.lworker)
        t.start()


    def lworker(self):
        try:
            audio_chunks = Queue(maxsize=200)
            speech_segments = Queue(maxsize=200)
            output_queue = Queue(maxsize=200)
            vad_ready = Event()
            asr_ready = Event()
            speech_start = Event()
            speech_end = Event()


            asr = GCPASR(speech_segments,output_queue, ready_flag=asr_ready)

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
    asr_publisher = GCPASRPublisher()
    

    rclpy.spin(asr_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    asr_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
