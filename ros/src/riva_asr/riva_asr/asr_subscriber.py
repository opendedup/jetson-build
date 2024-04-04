# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import riva.client
import numpy as np
import sounddevice as sd

from chat_interfaces.msg import Chat
from embeddings.srv import GetEmb
from rclpy.executors import MultiThreadedExecutor

import queue
import threading
import time
import json
import spacy


class RivaASR_Subscriber(Node):

    def __init__(self):
        super().__init__('riva_asr_subscriber')
        self.subscription = self.create_subscription(
            Chat,
            'asr',
            self.listener_callback,
            10)
        self.declare_parameter('sound_device',"Sound BlasterX G1: USB Audio (hw:0,0)")
        self.subscription  # prevent unused variable warning
        auth = riva.client.Auth(
            uri="localhost:50051",  # Replace with your Riva server address
        )
        self.tts_service = riva.client.SpeechSynthesisService(auth)
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        model = GenerativeModel("gemini-1.5-pro-preview-0215")
        self.chat = model.start_chat()
        self.config = {
            "max_output_tokens": 1024
        }
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.chat.send_message("""Your name is Shimmy, which is short for Shimmel or Schimmel.Your are humorous and have a dry wit. You love helping people and making them laugh. 
During our conversation keep the following in mind:
    * Do not use special charaters such as  * or # in your response.  
    * Do not use markdown format in your responses during our conversation.
    * Keep the dialog conversational.


""", 
                               generation_config=self.config, safety_settings=self.safety_settings)
        devices = sd.query_devices()
        # Find the device that we want to use.
        device = None
        for d in devices:
            #print(d)
            if d["name"] == self.get_parameter("sound_device").value:
                device = d
                break
        if device is not None:
            sd.default.device = device["name"]
        self.sample_rate_hz = 44100
        self.cli = self.create_client(GetEmb, 'get_emb')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.q = queue.Queue()
        
        t = threading.Thread(target=self.tworker)
        t.start()
    
    def tworker(self):
        resp_text = ""
        nlp = spacy.load("en_core_web_sm")
        while True:
            try:
                txt = self.q.get(block=True, timeout=.5)
                if txt is not None:
                    resp_text += txt
                if len(resp_text) > 100:
                    doc = nlp(resp_text)
                    sentences = [sent.text for sent in doc.sents]
                    for index, sentence in enumerate(sentences):
                        if index < len(sentences)-1:
                            req = { 
                                "language_code"  : "en-US",
                                "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
                                "sample_rate_hz" : self.sample_rate_hz,                          # Generate 44.1KHz audio
                                "voice_name"     : "English-US.Female-1",                    # The name of the voice to generate
                                "text": sentence.replace("*", " ")
                            }
                            resp = self.tts_service.synthesize(**req)
                            audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
                            sd.play(audio_samples, self.sample_rate_hz)
                            sd.wait()
                            self.get_logger().info(sentence)
                        else:
                            resp_text = sentence
                    
            except queue.Empty:
                if len(resp_text) > 0:
                    
                    doc = nlp(resp_text)
                    sentences = [sent.text for sent in doc.sents]

                    for sentence in sentences:
                        req = { 
                            "language_code"  : "en-US",
                            "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
                            "sample_rate_hz" : self.sample_rate_hz,                          # Generate 44.1KHz audio
                            "voice_name"     : "English-US.Female-1",                    # The name of the voice to generate
                            "text": sentence.replace("*", " ")
                        }
                        resp = self.tts_service.synthesize(**req)
                        audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
                        sd.play(audio_samples, self.sample_rate_hz)
                        sd.wait()
                        self.get_logger().info(sentence)
                    resp_text = ""
                    


    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.chat_text)
        emb_req = GetEmb.Request()
        emb_req.k = 1
        emb_req.embedding = msg.embedding
        future = self.cli.call_async(emb_req)
        threading.Thread(target=read_input, args=[future,self,msg]).start()
        

def read_input(future, obj,msg):
    while not future.done():
        time.sleep(0.1)
    obj.get_logger().info(future.result().embeddings[0].metadata)
    if len(future.result().embeddings) > 0:
        emb = future.result().embeddings[0]
        metadata = json.loads(emb.metadata)
        person  = metadata["name"]
        distance = emb.distance
        if distance < 1000 and person.startswith("robo"):
            obj.get_logger().info("Ignoring %s match is %d" %(person,distance))
        else:
            if not person.startswith("robo"):
                responses = obj.chat.send_message(msg.chat_text,
                                           generation_config=obj.config,stream=True, safety_settings=obj.safety_settings)
        
                for response in responses:
                    obj.q.put(response.text)
                    obj.get_logger().info(response.text)
    
    

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=8)


    riva_subscriber = RivaASR_Subscriber()

    executor.add_node(riva_subscriber)

    executor.spin()

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
