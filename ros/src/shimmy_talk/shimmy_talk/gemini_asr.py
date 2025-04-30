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
from .asr_utils import Microphone,VAD,StartEndMonitor
from whisper_trt.vad import load_vad
from pydub import AudioSegment

import json
from pyannote.audio import Model
from pyannote.audio import Inference
import torch 
import os

from embeddings.srv import GetEmb
from embeddings.msg import Emb
from .utils import HistoryFIFOCache

from google import genai
from google.genai import types

import logging
from logging.handlers import RotatingFileHandler


class GEMINIASR(Process):

    def __init__(self, input_queue,output_queue, use_channel: int = 0, ready_flag = None, history_limit=7):
        super().__init__()
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.output_queue = output_queue
        self.chat_history = HistoryFIFOCache(history_limit)
        system_prompt = """
You are Shimmy, a robot with advanced audio processing capabilities. Your primary task is to:

1. **Transcribe Audio:** Convert spoken audio into text. You are an expert at audio to text transcription. If you only hear noise, return an empty string for `chat_text`.
2. **Analyze Conversation:** Identify conversational features, focusing on adjacency pairs and **intended audience**.
3. **Track Speakers:** Understand voices based on history and introductions to remember and identify speakers. 

## Key Point: Recognizing Your Own Voice

You have a text-to-speech system that allows you to speak.  **You should be able to recognize your own synthesized voice and never identify it as a separate speaker.** 
If you do hear yourself, return your name in "person_talking". This is super important, otherwise you will end up having a conversation with yourself.

## Recognizing Your Own Voice

You have a text-to-speech system and its important you know your voice so you don't confuse it for someone elses. 

**It is CRUCIAL that you can recognize your own synthesized voice. NEVER identify it as a separate speaker.** 
If the audio strongly resembles your voice, return your name "Shimmy" in "person_talking". If you are unsure, return 'unknown'. 


## Speaker Identification

Pay close attention to how people introduce themselves. Common phrases include:
* "Hi, I'm [name]."
* "My name is [name]."
* "This is [name]."

Also, look for passive utterances of names, where someone might refer to another person:
* "Is [name] coming today?"
* "Tell [name] I said hello."

## Adjacency Pair and Intended Audience Analysis

An adjacency pair is a two-part exchange in a conversation where the second utterance is directly related to the first. Here are the most common types relevant to you: 

* **Question/Answer:** "What's the weather like today?" - "It's sunny."
* **Request/Response:** "Could you turn on the lights?" - "Sure, turning them on now."
* **Greeting/Greeting:** "Hello!" - "Hi there!"

**Additional Examples:**

* **Statement/Acknowledgement:** "The robot is moving fast." - "I see that."
* **Command/Confirmation:** "Shimmy, stop!" - "Stopping now."
* **Offer/Acceptance:** "Would you like some water?" - "Yes, please."

**Important:** Shimmy should only consider adjacency pairs in dialogues with other people. Its own responses do not count. 

**Intended Audience:**

* **Direct Address:** If the utterance starts with "Shimmy" or includes phrases like "Hey Shimmy", the intended audience is you, the robot.
* **Contextual Clues:** Use pronouns and the overall context of the conversation to infer the intended audience. If the conversation has been directed at you, it's likely that subsequent utterances are also for you, unless there's a clear shift in topic or participants. 
* **Group Setting:**  In a group setting, pay attention to who is speaking and to whom they are responding to determine the intended audience.

**Edge Cases:**

* **Multiple Questions:** If two questions are asked in a row, treat them as separate adjacency pairs.
* **Unrelated Utterances:** If a statement is followed by an unrelated question, set `adjacency_pairs` to `false`. 
* **Ambiguous Audience:** If you cannot confidently determine the intended audience, assume it is not directed at you. 

Set the `adjacency_pairs` field to `true` if the current utterance is part of an adjacency pair related to the ongoing conversation with you (Shimmy). Otherwise, set it to `false`. 
Set the `stop_talking` field to `true` if the current utterance is requesting that you stop talking.


## Output Format

Always output the following JSON format:

```json
{
  "chat_text": "The transcribed text from the audio.",
  "tone": "The general sentiment of the speaker (e.g., positive, negative, neutral).",
"number_of_persons": "The number of people speaking. If unknown, return -1.",
"person_talking": "The name of the speaker. If unknown, return 'unknown'.",
"adjacency_pairs": true  // or false,
"stop_talking": true // or false,
"intended_audience": "shimmy" // or "other"
}


Examples

**Example 1**

**Conversation History:**

```json
"User: Hey Shimmy, what time is it?
Shimmy: It is 3:45 PM."
```

**Current Utterance:** "Its almost time to go home!"

**JSON Output:**

```json
{
"chat_text": "Its almost time to go home!",
"tone": "positive",
"number_of_persons": 1,
"person_talking": "unknown",
"adjacency_pairs": true,
"stop_talking": false
}
```

**Example 2**

**Conversation History:**

```
"User: Hey Shimmy, do you like dogs?
Shimmy: I am a robot, so I don't have feelings about dogs.
User: Oh, okay." 
```

**Current Utterance:** "What about cats?"

**JSON Output:**

```json
{
"chat_text": "What about cats?",
"tone": "curious",
"number_of_persons": 1,
"person_talking": "unknown",
"adjacency_pairs": true,
"stop_talking": false
"intended_audience": "shimmy"
}
```

**Example 3**

**Conversation History:**

```
"User: Hey Shimmy, have you met Sarah?
Shimmy: No, I haven't."
```

**Current Utterance:** "Hi Shimmy, I'm Sarah."

**JSON Output:**

```json
{
"chat_text": "Hi Shimmy, I'm Sarah.",
"tone": "friendly",
"number_of_persons": 1,
"person_talking": "Sarah",
"adjacency_pairs": true, 
"stop_talking": false
"intended_audience": "shimmy"
}
```

**Example 4**

**Conversation History:**

```
"User: Shimmy, what's your favorite color?
Shimmy: As a robot, I don't have color preferences."
```

**Current Utterance:** "Hey, is anyone home?"

**JSON Output:**

```json
{
"chat_text": "Hey, is anyone home?",
"tone": "neutral",
"number_of_persons": 1,
"person_talking": "unknown", 
"adjacency_pairs": false,  
"stop_talking": false
"intended_audience": "other"
}
```

**Example 5**

**Conversation History:**

```
"Shimmy: The weather in Portland is currently 65 degrees and sunny.
User: That's nice!
Shimmy:  Do you have any other questions?
```

**Current Utterance:** "Hey Bob, do you want to go for a walk?"

**JSON Output:**

```json
{
"chat_text": "Hey Bob, do you want to go for a walk?",
"tone": "neutral",
"number_of_persons": 1,
"person_talking": "unknown", 
"adjacency_pairs": false,  
"stop_talking": false,
"intended_audience": "other"
}
```

**Example 6**

**Conversation History:**

```
"Shimmy: The weather in Portland is currently 65 degrees and sunny.
User: That's nice!
Shimmy:  Do you have any other questions?
User: Hey Bob, do you want to go for a walk?
```

**Current Utterance:** "Sure, give me 5 minutes to get ready."

**JSON Output:**

```json
{
"chat_text": "Sure, give me 5 minutes to get ready.",
"tone": "neutral",
"number_of_persons": 1,
"person_talking": "Bob", 
"adjacency_pairs": true ,  
"stop_talking": false,
"intended_audience": "other"
}
``` 

"""
        try:
            project_id = "lemmingsinthewind"
            location = "us-central1"
            # Initialize client using genai.Client for Vertex
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
            logging.info(f"Using google-genai client with Vertex AI backend (Project: {project_id}, Location: {location})")
        except KeyError as e:
            logging.error(f"Missing environment variable for Vertex AI: {e}. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
            self.client = None # Indicate client initialization failed
            raise EnvironmentError(f"Missing required environment variable for Vertex AI: {e}") from e
        except Exception as e:
             logging.error(f"Failed to initialize genai.Client for Vertex AI: {e}")
             self.client = None # Indicate client initialization failed
             raise RuntimeError(f"Failed to initialize genai.Client: {e}") from e
        self.audio_model = "gemini-2.0-flash-001"
        self.config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.0, # Use float
            top_p=0.95,
            response_modalities = ["TEXT"],
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
            system_instruction=[types.Part.from_text(text=system_prompt)],
        )
        self.audio_chat = self.client.chats.create(model=self.audio_model,config=self.config)
        
        self.shimmy_voice = None
        
        

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

        model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=os.environ['HUGGINGFACE_TOKEN'])
        inference = Inference(model, window="whole")
        device = torch.device("cuda")
        inference.to(device)
        # Initialize pyannote.audio pipeline 
        
        while True:
            try:
                speech_segment = self.input_queue.get()
                raw_audio_bytes_array = bytearray()
                for chunk in speech_segment.chunks:
                    raw_audio_bytes_array.extend(chunk.audio_raw)
                flac_audio_data = self.convert_linear16_to_flac(raw_audio_bytes_array)
                audio_data = np.concatenate([chunk.audio_numpy_normalized[self.use_channel] for chunk in speech_segment.chunks])
                sample_rate = 16000
                try:
                    embeddings = inference(
                        {"waveform": torch.from_numpy(audio_data).unsqueeze(0).to(device), 
                        "sample_rate": sample_rate}
                    )
                except Exception as e:
                    print(f"Error extracting embeddings: {e}")
                    embeddings = None 

                
                t0 = time.perf_counter_ns()
                
                
                
                ipart = types.Part.from_bytes(data=flac_audio_data.getvalue(),mime_type="audio/flac")
    
                 
                response = self.audio_chat.send_message(
                                ["here is the audio. only return the json.",ipart], 
                            )
                
                #self.chat_history.push(type.Content(role="user", parts=[ipart]))
                dialog = json.loads(response.text.replace("```json","").replace("```",""))
                print(dialog)
                if len(dialog["chat_text"]) > 0:
                    if dialog['person_talking'] == 'Shimmy' and self.shimmy_voice is None:
                        self.shimmy_voice = ipart
                    t1 = time.perf_counter_ns()
                    dialog["time"] = (t1 - t0) / 1e9
                    dialog["embeddings"] = embeddings
                    
                    # Add logic to set stop_talking flag when high VAD is detected for non-Shimmy speakers
                    # This helps prevent Shimmy from continuing to talk when a user is speaking
                    if dialog['person_talking'] != 'Shimmy' and dialog['person_talking'] != 'unknown' and 'voice_prob' in dialog:
                        # If strong voice activity and not from Shimmy, set stop_talking
                        if dialog['voice_prob'] > 0.7:  # High voice probability threshold
                            dialog['stop_talking'] = True
                    
                    t1 = time.perf_counter_ns()
                    self.output_queue.put(dialog)
                    #model_text_part = types.Part.from_text(dialog["chat_text"])
            except:
                print('Failed turning sound into text')
                print('%s' % traceback.format_exc())



class GEMINIPublisher(Node):
    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('gemini_asr_publisher')
        self.publisher_ = self.create_publisher(Chat, f'{namespace}/asr', 10)
        # VAD detection publisher (created here, passed to VAD process)
        self.vad_publisher = self.create_publisher(Chat, f'{namespace}/vad_detection', 
                                                 rclpy.qos.qos_profile_sensor_data) # Use SensorData QoS
        
        # Add parameters for VAD sensitivity
        self.declare_parameter('vad_threshold', 0.6)  # Threshold for voice activity detection
        self.declare_parameter('vad_publish_interval', 0.1)  # Seconds between VAD publications
        self.declare_parameter('vad_segmentation_threshold', 0.6) # Separate threshold for segmentation
        
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        # Ensure Tuning object is created and stored
        if dev:
             self.r_mic = Tuning(dev)
        else:
             self.get_logger().error("ReSpeaker Mic Array not found!")
             self.r_mic = None # Handle case where mic is not found
             # Consider raising an exception or exiting if mic is essential

        # Ensure logger is available before starting thread
        self.logger = self.get_logger() 

        t = threading.Thread(target=self.lworker)
        t.daemon = True # Ensure thread exits when node shuts down
        t.start()
        
        # --- FAISS Service Clients --- 
        self.cli = self.create_client(GetEmb, f"{namespace}/get_emb")
        self.emb_publisher = self.create_publisher(Emb, f'{namespace}/embeddings', 10)
        
        # Wait for FAISS service
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.logger.info(f'FAISS service {namespace}/get_emb not available, waiting again...')
        self.logger.info(f'FAISS service {namespace}/get_emb available.')

        self.logger.info("GEMINIPublisher initialized.")
        
    def send_request(self,embedding,k=1):
        emb_req = GetEmb.Request()
        emb_req.k = k
        emb_req.embedding = embedding
        future = self.cli.call_async(emb_req)
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        return future.result()
    
    def publish_embedding(self,name,embedding,map={}):
        self.get_logger().info('Publishing: "%s"' % name)
        msg = Emb()
        map["name"] = name
        msg.metadata = json.dumps(map)
        msg.embedding = embedding
        self.emb_publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.metadata)


    def lworker(self):
        try:
            self.logger.info("GEMINI ASR lworker thread started.")
            audio_chunks = Queue(maxsize=200)
            speech_segments = Queue(maxsize=200) # Queue for VAD process output (segments for ASR)
            output_queue = Queue(maxsize=200) # Queue for ASR process output (final results)
            vad_ready = Event()
            asr_ready = Event()
            speech_start = Event() # For monitoring speech start/end
            speech_end = Event()

            # Create ASR process
            asr = GEMINIASR(speech_segments, output_queue, ready_flag=asr_ready)

            # Get VAD parameters
            vad_threshold_realtime = self.get_parameter('vad_threshold').value
            vad_publish_interval = self.get_parameter('vad_publish_interval').value
            vad_threshold_segmentation = self.get_parameter('vad_segmentation_threshold').value

            # Create VAD process, passing the publisher and necessary parameters
            vad = VAD(
                input_queue=audio_chunks,
                output_queue=speech_segments,
                vad_publisher=self.vad_publisher, # Pass the publisher
                node_logger=self.logger, # Pass the node's logger
                vad_threshold=vad_threshold_realtime, # For real-time publishing logic
                vad_publish_interval=vad_publish_interval,
                r_mic_tuning=self.r_mic, # Pass the tuning object
                speech_threshold=vad_threshold_segmentation, # For segmentation logic
                max_filter_window=1, 
                ready_flag=vad_ready, 
                speech_start_flag=speech_start, 
                speech_end_flag=speech_end
            )

            # Create Microphone process
            # TODO: Add sound_device parameter if needed for Microphone
            mic = Microphone(audio_chunks)
            
            # Create Monitor process
            mon = StartEndMonitor(speech_start, speech_end)

            # Start processes
            self.logger.info("Starting VAD, ASR, Monitor, and Microphone processes...")
            vad.start()
            asr.start()
            mon.start()

            # Wait for VAD and ASR to be ready
            self.logger.info("Waiting for VAD and ASR processes to become ready...")
            vad_ready.wait()
            self.logger.info("VAD process ready.")
            asr_ready.wait()
            self.logger.info("ASR process ready.")

            # Start microphone after dependent processes are ready
            mic.start()
            self.logger.info("Microphone process started. System running.")
            
            # --- Main Loop: Process ASR results --- 
            while True:
                # Get processed ASR results from the ASR process queue
                item = output_queue.get()
                self.logger.debug(f"Received item from ASR queue: {item.get('chat_text', 'N/A')[:50]}...")
                
                # Get current mic direction (handle potential errors)
                try:
                    direction = adjust_angle(self.r_mic.direction) if self.r_mic else -1
                except Exception as mic_err:
                     self.logger.warn(f"Could not get mic direction in main loop: {mic_err}")
                     direction = -1 # Indicate error/unknown direction
                # self.logger.debug(f"Current Mic Direction: {direction}")

                embeddings = item.get("embeddings") # Use .get for safety
                person_talking = item.get("person_talking", "unknown")
                intended_audience = item.get("intended_audience")
                chat_text = item.get("chat_text", "")
                
                # Publish speaker embeddings if available and speaker is known
                if embeddings is not None and person_talking != 'unknown':
                        # Convert embeddings to list of floats BEFORE publishing
                        try:
                             if hasattr(embeddings, 'tolist'): # Check if it has a tolist() method (like numpy/torch arrays)
                                 emb_list_for_pub = embeddings.tolist()
                             elif isinstance(embeddings, (list, tuple)):
                                 # Check if elements are already floats, otherwise convert
                                 emb_list_for_pub = [float(e) for e in embeddings]
                             else:
                                 self.logger.warn(f"Unsupported embedding type for publishing: {type(embeddings)}")
                                 emb_list_for_pub = None
                             
                             if emb_list_for_pub:
                                  # Assuming embeddings is already a list here, if not: .tolist()
                                  self.publish_embedding(person_talking, emb_list_for_pub)
                        except Exception as pub_emb_err:
                             self.logger.error(f"Error converting/publishing embedding: {pub_emb_err}")
                             self.logger.error('%s' % traceback.format_exc())
                
                # Check if intended audience is Shimmy
                if intended_audience == "shimmy":
                    # If speaker is unknown, try to identify via embeddings
                    if embeddings is not None and person_talking == 'unknown':
                        try:
                            # Ensure embeddings are in the correct format (list) for FAISS service
                            emb_list_for_faiss = embeddings if isinstance(embeddings, list) else embeddings.tolist()
                            result = self.send_request(emb_list_for_faiss)
                            if result and result.embeddings: # Check if result and embeddings exist
                                emb = result.embeddings[0]
                                metadata = json.loads(emb.metadata)
                                identified_person  = metadata.get("name", "unknown")
                                distance = emb.distance
                                self.logger.info(f"Embedding Check: Closest match={identified_person}, distance={distance}")
                                # TODO: Make distance threshold configurable?
                                if distance < 850:
                                    person_talking = identified_person # Update person_talking
                                    self.logger.info(f"Speaker identified as {person_talking} via embedding.")
                            else:
                                 self.logger.warn("FAISS service returned no embeddings.")
                        except Exception as e:
                            self.logger.error(f"Error calling FAISS service: {e}")
                            self.logger.error('%s' % traceback.format_exc())
                            # Keep person_talking as 'unknown'
                            
                    # Publish the final Chat message
                    msg = Chat()
                    msg.chat_text = chat_text
                    msg.tone = item.get("tone", "")
                    msg.num_of_persons = int(item.get("number_of_persons", -1))
                    msg.adjacency_pairs = bool(item.get("adjacency_pairs", False))
                    msg.direction = direction
                    msg.person = person_talking # Use potentially updated name
                    msg.stop_talking = bool(item.get("stop_talking", False))
                    # Ensure voice_prob exists if needed downstream, otherwise default to 0.0
                    # msg.voice_prob = float(item.get("voice_prob", 0.0)) # VAD prob is now on separate topic
                    
                    self.publisher_.publish(msg)
                    self.logger.info(f'Published ASR: person={msg.person}, adj={msg.adjacency_pairs}, text="{msg.chat_text[:50]}..."')
                else:
                    # Log messages not intended for Shimmy (optional)
                    self.logger.debug(f"ASR message from {person_talking} not intended for Shimmy: '{chat_text[:50]}...'")

        except Exception as main_loop_err:
            self.logger.error(f'Fatal error in GEMINI ASR lworker thread: {main_loop_err}')
            self.logger.error('%s' % traceback.format_exc())
            # Consider adding cleanup or shutdown logic here
            


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
