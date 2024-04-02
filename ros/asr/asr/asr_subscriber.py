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

from std_msgs.msg import String

import queue
import threading


class RivaASR_Subscriber(Node):

    def __init__(self):
        super().__init__('riva_asr_subscriber')
        self.subscription = self.create_subscription(
            String,
            'asr',
            self.listener_callback,
            10)
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
            if d["name"] == "Sound BlasterX G1: USB Audio (hw:0,0)":
                device = d
                break
        if device is not None:
            sd.default.device = device["name"]
        self.sample_rate_hz = 44100
        self.q = queue.Queue()
        t = threading.Thread(target=self.tworker)
        t.start()
    
    def tworker(self):
        resp_text = ""
        while True:
            try:
                txt = self.q.get(block=True, timeout=.1)
                self.get_logger().info(txt)
                if len(resp_text) > 100:
                    req = { 
                        "language_code"  : "en-US",
                        "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
                        "sample_rate_hz" : self.sample_rate_hz,                          # Generate 44.1KHz audio
                        "voice_name"     : "English-US.Female-1",                    # The name of the voice to generate
                        "text": resp_text
                    }
                    resp = self.tts_service.synthesize(**req)
                    audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
                    sd.play(audio_samples, self.sample_rate_hz)
                    sd.wait()
                    self.get_logger().info(resp_text)
                    resp_text = ""
                if txt is not None:
                    resp_text += txt
            except queue.Empty:
                if len(resp_text) > 0:
                    req = { 
                        "language_code"  : "en-US",
                        "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
                        "sample_rate_hz" : self.sample_rate_hz,                          # Generate 44.1KHz audio
                        "voice_name"     : "English-US.Female-1",                    # The name of the voice to generate
                        "text": resp_text
                    }
                    resp = self.tts_service.synthesize(**req)
                    audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
                    sd.play(audio_samples, self.sample_rate_hz)
                    sd.wait()
                    self.get_logger().info(resp_text)
                    resp_text = ""
                    


    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        responses = self.chat.send_message(msg.data,
                                           generation_config=self.config,stream=True, safety_settings=self.safety_settings)
        
        for response in responses:
            self.q.put(response.text)
            self.get_logger().info(response.text)


def main(args=None):
    rclpy.init(args=args)

    riva_subscriber = RivaASR_Subscriber()

    rclpy.spin(riva_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    riva_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
