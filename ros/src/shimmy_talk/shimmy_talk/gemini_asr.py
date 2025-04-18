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
from pydub import AudioSegment

import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
import vertexai.generative_models as generative_models
from pyannote.audio import Model
from pyannote.audio import Inference
import torch 
import os

from embeddings.srv import GetEmb
from embeddings.msg import Emb
from .utils import HistoryFIFOCache




class GEMINIASR(Process):

    def __init__(self, input_queue,output_queue, use_channel: int = 0, ready_flag = None, history_limit=7):
        super().__init__()
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.output_queue = output_queue
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        self.chat_history = HistoryFIFOCache(history_limit)
        system_prompt = """
You are Shimmy, a robot with advanced audio processing capabilities. Your primary task is to:

1. **Transcribe Audio:** Convert spoken audio into text. You are an expert at transcription. If you only hear noise, return an empty string for `chat_text`.
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
        
        self.audio_model = GenerativeModel("gemini-1.5-flash-002",system_instruction=[system_prompt])
        #self.audio_chat = self.audio_model.start_chat()
        
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
                
                
                
                ipart = Part.from_data(data=flac_audio_data.getvalue(),mime_type="audio/flac")
    
                history_content = self.chat_history.get_history()
                if self.shimmy_voice is not None:
                    history_content.append(Content(role="user", parts=[Part.from_text("This is what shimmy's voice sounds like."),self.shimmy_voice]))
                audio_chat = self.audio_model.start_chat(history=history_content)
                response = audio_chat.send_message(
                                ["here is the audio. only return the json.",ipart], 
                                generation_config=self.config,
                                stream=False, 
                                safety_settings=self.safety_settings
                            )
                
                self.chat_history.push(Content(role="user", parts=[ipart]))
                dialog = json.loads(response.text.replace("```json","").replace("```",""))
                print(dialog)
                if len(dialog["chat_text"]) > 0:
                    if dialog['person_talking'] == 'Shimmy' and self.shimmy_voice is None:
                        self.shimmy_voice = ipart
                    t1 = time.perf_counter_ns()
                    dialog["time"] = (t1 - t0) / 1e9
                    dialog["embeddings"] = embeddings
                    t1 = time.perf_counter_ns()
                    self.output_queue.put(dialog)
                    model_text_part = Part.from_text(dialog["chat_text"])
                    self.chat_history.push(Content(role="model", parts=[model_text_part])) 
            except:
                print('Failed turning sound into text')
                print('%s' % traceback.format_exc())



class GEMINIPublisher(Node):
    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('gemini_asr_publisher')
        self.publisher_ = self.create_publisher(Chat, f'{namespace}/asr', 10)
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        self.r_mic = Tuning(dev)
        t = threading.Thread(target=self.lworker)
        t.start()
        self.cli = self.create_client(GetEmb, f"{namespace}/get_emb")
        self.emb_publisher = self.create_publisher(Emb, f'{namespace}/embeddings', 10)
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'faiss service {namespace} not available, waiting again...')
        self.emb_publisher
        
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
                embeddings = item["embeddings"]
                if embeddings is not None and item['person_talking'] != 'unknown':
                        self.publish_embedding(item['person_talking'],embeddings.tolist())
                if item["intended_audience"] == "shimmy":
                    
                    if embeddings is not None and item['person_talking'] == 'unknown':
                        try:
                            result = self.send_request(embeddings.tolist())
                            emb = result.embeddings[0]
                            metadata = json.loads(emb.metadata)
                            person  = metadata["name"]
                            distance = emb.distance
                            self.get_logger().info("Checking %s match is %d" %(person,distance))
                            if distance < 850:
                                item["person_talking"] = person
                        except Exception as e:
                            print(f"Error extracting embeddings: {e}")
                            embeddings = None 
                    msg = Chat()
                    msg.chat_text = item["chat_text"]
                    msg.tone = item["tone"]
                    msg.num_of_persons = int(item["number_of_persons"])
                    msg.adjacency_pairs = bool(item["adjacency_pairs"])
                    msg.direction = direction
                    msg.person = item["person_talking"]
                    msg.stop_talking = item["stop_talking"]
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
