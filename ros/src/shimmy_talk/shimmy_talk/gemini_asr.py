import rclpy
from rclpy.node import Node
from chat_interfaces.msg import Chat

import traceback

import queue
import threading

import numpy as np
import time
from multiprocessing import Process, Queue, Event
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
from .utils import adjust_angle
from .usb_4_mic_array import Tuning
import usb.core
import usb.util

import scipy.io.wavfile as wav
import io
from .whisper_asr import Microphone,VAD,StartEndMonitor
from pydub import AudioSegment

import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.generative_models as generative_models



class GEMINIASR(Process):

    def __init__(self, input_queue,output_queue, use_channel: int = 0, ready_flag = None):
        super().__init__()
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.output_queue = output_queue
        self.model_name = "superb/wav2vec2-large-superb-sid"
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        system_prompt = """You are Shimmy, a robot with advanced audio processing capabilities. Your primary task is to:

1. **Transcribe Audio:**  Convert spoken audio into text. If you only hear noise, return an empty string for `chat_text`.
2. **Analyze Conversation:**  Identify conversational features, focusing on adjacency pairs. 
3. **Track Speakers:**  Understand voices based on history and introductions to remember and identify speakers. Keep track of your voice as well. 

## Adjacency Pair Analysis

An adjacency pair is a two-part exchange in a conversation where the second utterance is directly related to the first. Here are the most common types relevant to you: 

* **Question/Answer:** "What's the weather like today?" - "It's sunny."
* **Request/Response:** "Could you turn on the lights?" - "Sure, turning them on now."
* **Greeting/Greeting:** "Hello!" - "Hi there!"

Set the `adjacency_pairs` field to `true` if the current utterance is part of an adjacency pair related to the ongoing conversation with Shimmy. Otherwise, set it to `false`. 

## Output Format

Always output the following JSON format:

```json
{
  "chat_text": "The transcribed text from the audio.",
  "tone": "The general tone of the speaker (e.g., happy, sad, angry).",
  "number_of_persons": "The number of people speaking. If unknown, return -1.",
  "person_talking": "The name of the speaker, based on introductions and conversation history. If unknown, return 'unknown'.",
  "adjacency_pairs": true  // or false
} 
```
Example
Audio: "Hey Shimmy, what time is it?"
JSON Output:
{
  "chat_text": "Hey Shimmy, what time is it?",
  "tone": "neutral",
  "number_of_persons": 1,
  "person_talking": "unknown", // You don't know this speaker yet.
  "adjacency_pairs": false
}
"""
        
        self.audio_model = GenerativeModel("gemini-flash-experimental",system_instruction=[system_prompt])
        self.audio_chat = self.audio_model.start_chat()
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.config = {
            "max_output_tokens": 8192,
            "temperature": 0,
            "top_p": 0.95,
        }

    def convert_linear16_to_flac(self, linear16_data, sample_rate=16000):
        """Converts LINEAR16 audio data to FLAC format.

        Args:
            linear16_data: The LINEAR16 audio data as a bytes object.
            sample_rate: The sample rate of the audio data in Hz.

        Returns:
            The FLAC-encoded audio data as a bytes object.
        """

        audio_segment = AudioSegment(
            data=linear16_data,
            sample_width=2,  # 2 bytes per sample for 16-bit audio
            frame_rate=sample_rate,
            channels=1  # Assuming mono audio
        )
        flac_buffer = io.BytesIO()
        audio_segment.export(flac_buffer, format="flac")  # Export directly to the buffer
        flac_buffer.seek(0)  # Reset the buffer position to the beginning
        return flac_buffer
    
    def run(self):
            
        
        
        if self.ready_flag is not None:
            self.ready_flag.set()

        while True:
            try:
                speech_segment = self.input_queue.get()
                raw_audio_bytes_array = bytearray()
                for chunk in speech_segment.chunks:
                    raw_audio_bytes_array.extend(chunk.audio_raw)
                flac_audio_data = self.convert_linear16_to_flac(raw_audio_bytes_array)
                #audio_samples = np.frombuffer(raw_audio_bytes_array, dtype=np.int16)
                #buffer = io.BytesIO()
                #wav.write(buffer, 16000, audio_samples)
                t0 = time.perf_counter_ns()
                
                
                
                ipart = Part.from_data(data=flac_audio_data.getvalue(),mime_type="audio/flac")
                response = self.audio_chat.send_message(["here is the audio. only return the json.",ipart],
                                generation_config=self.config,stream=False, safety_settings=self.safety_settings)
                dialog = json.loads(response.text.replace("```json","").replace("```",""))
                print(dialog)
                if len(dialog["chat_text"]) > 0:
                    t1 = time.perf_counter_ns()
                    dialog["time"] = (t1 - t0) / 1e9
                    t1 = time.perf_counter_ns()
                    self.output_queue.put(dialog)
            except:
                print('Failed turning sound into text')
                print('%s' % traceback.format_exc())



class GEMINIPublisher(Node):
    def __init__(self):
        super().__init__('gemini_asr_publisher')
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


            asr = GEMINIASR(speech_segments,output_queue, ready_flag=asr_ready)

            vad = VAD(audio_chunks, speech_segments, max_filter_window=1, ready_flag=vad_ready, speech_start_flag=speech_start, speech_end_flag=speech_end)

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
                msg.chat_text = item["chat_text"]
                msg.tone = item["tone"]
                msg.num_of_persons = int(item["number_of_persons"])
                msg.adjacency_pairs = bool(item["adjacency_pairs"])
                msg.direction = direction
                msg.person = item["person_talking"]
                self.publisher_.publish(msg)
                self.get_logger().info(f'Publishing: {msg.chat_text} adj:{msg.adjacency_pairs} tone:{msg.tone} num_persons:{msg.num_of_persons}')
        except:
            self.get_logger().error('%s' % traceback.format_exc())
        
        


def main(args=None):
    rclpy.init(args=args)
    asr_publisher = GEMINIPublisher()
    

    rclpy.spin(asr_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    asr_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
