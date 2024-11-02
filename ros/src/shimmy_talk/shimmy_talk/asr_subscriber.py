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
from vertexai.generative_models import (
    GenerativeModel,
    Part,
)
import vertexai.generative_models as generative_models
import riva.client
import sounddevice as sd

from chat_interfaces.msg import Chat
from rclpy.executors import MultiThreadedExecutor

import queue
import threading
import json
import spacy
import queue
import string


from .skills.agents import robot_agents,AgentRunner


class RivaASR_Subscriber(Node):

    def __init__(self):
        super().__init__('riva_asr_subscriber')
        self.subscription = self.create_subscription(
            Chat,
            'asr',
            self.listener_callback,
            10)
        self.declare_parameter('sound_device',"Sound BlasterX G1: USB Audio (hw:0,0)")
        self.declare_parameter('robot_names',["Schimmel", "Schimmy","Shimmy"])
        self.subscription  # prevent unused variable warning
        auth = riva.client.Auth(
            uri="localhost:50051",  # Replace with your Riva server address
        )
        self.tts_service = riva.client.SpeechSynthesisService(auth)
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        model = GenerativeModel("gemini-1.5-flash-002",tools=[robot_agents])
        self.robot_names = self.get_parameter("robot_names").value
        self.chat = model.start_chat()
        self.config = {
            "max_output_tokens": 1024,
            "temperature": 0.0,
        }
        

        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
        self.chat.send_message("""Your name is Shimmy, which is short for Shimmel or Schimmel.
Some Facts about you can use in context when answering questions:
    * You live with a family of 6 people.
    * You live in a house in Portland Oregon. This can be helpful when answering questions about weather and time.

""", 
                               generation_config=self.config, safety_settings=self.safety_settings)
        devices = sd.query_devices()
        # Find the device that we want to use.
        device = None
        for d in devices:
            print(d["name"])
            if d["name"].startswith(self.get_parameter("sound_device").value):
                device = d
                break
        if device is not None:
            sd.default.device = device["name"]
        self.sample_rate_hz = 44100
        
        self.q = queue.Queue()
        self.robot_runner = AgentRunner()
        t = threading.Thread(target=self.tworker)
        t.start()
        
        
    def check(self,sentence, words):
        self.get_logger().info(sentence)
        """
        Check if an array of words are in a sentence.

        Args:
            sentence: The sentence to check.
            words: The array of words to check.

        Returns:
            True if all of the words in the array are in the sentence, False otherwise.
        """
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()

        # Check if all of the words in the array are in the sentence.
        for word in words:
            if word.lower() in sentence:
                return True

        # If all of the words in the array are in the sentence, return True.
        return False
    
    def read_input(self,msg):
        result = self.robot_runner.voice_emb_client.send_request(msg.sid_embedding)
        #while not future.done():
        #    time.sleep(0.05)
        self.get_logger().info(result.embeddings[0].metadata)
        if len(result.embeddings) > 0:
            emb = result.embeddings[0]
            metadata = json.loads(emb.metadata)
            person  = metadata["name"]
            distance = emb.distance
            if distance < 1500 and person.startswith("robo"):
                self.get_logger().info("Ignoring %s match is %d" %(person,distance))
            else:
                if distance > 1500:
                    self.get_logger().info(str(distance))
                    person = "unknown"
                self.get_logger().info(person)
                responses = self.chat.send_message(f"{person} - \"{msg.chat_text}\"",
                                        generation_config=self.config,stream=True, safety_settings=self.safety_settings)
                
                while responses is not None:
                    api_part = None
                    data_part = None
                    for response in responses:
                        self.get_logger().info("CMD = %s" %(response.candidates[0].content.parts[0].function_call.name))
                        if response.candidates[0].content.parts[0].function_call.name == "get_weather":
                            coords =response.candidates[0].content.parts[0].function_call.args["coords"].split(",")
                            api_part = self.robot_runner.get_weather(coords)
                        elif response.candidates[0].content.parts[0].function_call.name == "get_time":
                            time_zone =response.candidates[0].content.parts[0].function_call.args["time_zone"]
                            api_part = self.robot_runner.get_current_time(time_zone)
                        elif response.candidates[0].content.parts[0].function_call.name == "store_voice":
                            person =response.candidates[0].content.parts[0].function_call.args["name"]
                            api_part = self.robot_runner.add_voice(person,result.embedding)
                        elif response.candidates[0].content.parts[0].function_call.name == "take_picture":
                            api_part = Part.from_function_response(
                                name="get_time",
                                response={
                                    "content": {"message":"image will be sent in next message"},
                                },
                            )
                            data_part = self.robot_runner.get_image()
                        elif response.candidates[0].content.parts[0].function_call.name == "remember_image_objects":
                            api_part = self.robot_runner.remember_image_objects(response.candidates[0].content.parts[0].function_call.args["picture_context"])
                        else:
                            self.q.put(response.candidates[0].content.parts[0].text)
                            self.get_logger().info(response.candidates[0].content.parts[0].text)
                            
                    if api_part != None:
                        responses = self.chat.send_message(
                                api_part,generation_config=self.config,stream=True, safety_settings=self.safety_settings
                        )
                        if data_part is not None:
                            responses = self.chat.send_message(
                                data_part,stream=True,generation_config=self.config,safety_settings=self.safety_settings
                            )
                    else:
                        responses = None
                            
                        
    def tworker(self):
        resp_text = ""
        nlp = spacy.load("en_core_web_sm")
        block_size=20480
        stream = sd.RawOutputStream(
            samplerate=self.sample_rate_hz, blocksize=block_size,
            device=sd.default.device, channels=1, dtype='int16',
            callback=stream_sound_callback)
        with stream:
            while True:
                try:
                    txt = self.q.get(block=True, timeout=.5)
                    if txt is not None:
                        resp_text += txt
                    if len(resp_text) > 50:
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents][:-1]
                        resp_text = [sent.text for sent in doc.sents][-1]
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences)
                            

                            req = { 
                                "language_code"  : "en-US",
                                "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
                                "sample_rate_hz" : self.sample_rate_hz,                          # Generate 44.1KHz audio
                                "voice_name"     : "English-US.Female-1",                    # The name of the voice to generate
                                "text": talk_text.replace("*", " ")
                            }
                            try:
                                resp = self.tts_service.synthesize(**req)
                                dtr = split_array_into_chunks(resp.audio,block_size*2)
                                for dt in dtr:
                                    sound_q.put(dt)  # Pre-fill queue
                                self.get_logger().info(f"saying {talk_text}")
                            except Exception as error:
                                self.get_logger().error(error.message)
                            
                        
                except queue.Empty:
                    if len(resp_text) > 0:
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents]
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences)
                            req = { 
                                "language_code"  : "en-US",
                                "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
                                "sample_rate_hz" : self.sample_rate_hz,                          # Generate 44.1KHz audio
                                "voice_name"     : "English-US.Female-1",                    # The name of the voice to generate
                                "text": talk_text.replace("*", " ")
                            }
                            try:
                                resp = self.tts_service.synthesize(**req)
                                dtr = split_array_into_chunks(resp.audio,block_size*2)
                                for dt in dtr:
                                    sound_q.put(dt)  # Pre-fill queue
                                self.get_logger().info(f"saying {talk_text}")
                                resp_text = ""
                            except Exception as error:
                                self.get_logger().error(error.message)
                        
    

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.chat_text)
        if self.check(msg.chat_text,self.robot_names):
            threading.Thread(target=self.read_input, args=[msg]).start()

sound_q = queue.Queue()
                    
def stream_sound_callback(outdata, frames, time, status):
    if status.output_underflow:
        print('Output underflow: increase blocksize?')
        raise sd.CallbackAbort
    assert not status
    try:
        data = sound_q.get_nowait()
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        else:
            outdata[:] = data
    except queue.Empty as e:
        outdata[:] = b'\x00' * len(outdata)
    
def split_array_into_chunks(array, max_length):
  """Splits an array into chunks with a maximum length of array.

  Args:
    array: The array to split.
    max_length: The maximum length of each chunk.

  Returns:
    A list of chunks.
  """

  chunks = []
  for i in range(0, len(array), max_length):
    chunk = array[i:i + max_length]
    chunks.append(chunk)
  return chunks





    

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
