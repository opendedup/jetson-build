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
import sounddevice as sd

from chat_interfaces.msg import Chat,SaySomething
from embeddings.srv import GetEmb
from rclpy.executors import MultiThreadedExecutor

import queue
import threading
import time
import json
import spacy
import queue
import re
import string


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
        model = GenerativeModel("gemini-1.5-pro-preview-0215")
        self.robot_names = self.get_parameter("robot_names").value
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
        self.chat.send_message("""Your name is Shimmy, which is short for Shimmel or Schimmel.Your are humorous and have a dry wit. You love helping people and making them laugh. Today is Wednesday April 3rd 2024 at 20:02:11.
During our conversation keep the following in mind: 

* Do not use markdown format in your responses during our conversation.    * Keep the dialog conversational.

You have the following skills:
* snapphotos = take pictures of scenes
* time = get the current time and date
* peoplevoice = remember people by their voices
* peoplefaces = remember people by their faces
* livelyconvo = have lively and informative conversations about general topics
* explainskills = explain the above skills that you have but to not use the short name in your explanation.

Your responses should be conversational. At the beginning of each response include the following
### Skill Used: the short name for the skill used

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
        self.srv = self.create_service(SaySomething, 'say_something', self.say_something)
        self.cli = self.create_client(GetEmb, 'get_emb')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.q = queue.Queue()
        
        t = threading.Thread(target=self.tworker)
        t.start()
    
    def say_something(self, request, response):
        responses = self.chat.send_message(request.text,
            generation_config=self.config,stream=True, safety_settings=self.safety_settings)
        
        for response in responses:
            self.q.put(response.text)
            self.get_logger().info(response.text)
        return response
        
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
                        skill = extract_skill(resp_text)
                        self.get_logger().info(f"#############{skill}")
                        resp_text = remove_skill_info(resp_text)
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
                                self.get_logger().info(talk_text)
                            except Exception as error:
                                self.get_logger().error(error.message)
                            
                        
                except queue.Empty:
                    if len(resp_text) > 0:
                        skill = extract_skill(resp_text)
                        self.get_logger().info(f"#############{skill}")
                        resp_text = remove_skill_info(resp_text)
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
                                self.get_logger().info(talk_text)
                                resp_text = ""
                            except Exception as error:
                                self.get_logger().error(error.message)
                        
    
    

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.chat_text)
        if self.check(msg.chat_text,self.robot_names):
            emb_req = GetEmb.Request()
            emb_req.k = 1
            emb_req.embedding = msg.embedding
            future = self.cli.call_async(emb_req)
            threading.Thread(target=read_input, args=[future,self,msg]).start()

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


def extract_skill(response_text):
  """Extracts the skill used from a response string.

  Args:
    response_text: The text of the response.

  Returns:
    The skill used, or None if no skill was found.
  """
  match = re.search(r"### Skill Used: (\w+)", response_text)
  if match:
    return match.group(1)
  else:
    return None

def remove_skill_info(response_text):
  """Removes the skill used information and surrounding syntax from a response.

  Args:
    response_text: The text of the response.

  Returns:
    The response text with the skill information removed.
  """
  return re.sub(r"### Skill Used: \w+", "", response_text)

def read_input(future, obj,msg):
    while not future.done():
        time.sleep(0.05)
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
