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

from concurrent.futures import ThreadPoolExecutor
import traceback
import rclpy
from rclpy.node import Node
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    Part,
)
import vertexai.preview.generative_models as generative_models

from google.cloud import texttospeech

from pydub import AudioSegment
import pyaudio
import wave
import io
import scipy.io.wavfile as wav

import numpy as np

from chat_interfaces.msg import Chat
from rclpy.executors import MultiThreadedExecutor

import queue
import threading
import json
import spacy
import queue
import string
import time


from .skills.agents import robot_agents,AgentRunner,google_search_tool


class GoogleASRSubscriber(Node):

    def __init__(self):
        super().__init__('google_asr_subscriber')
        self.subscription = self.create_subscription(
            Chat,
            'asr',
            self.listener_callback,
            10)
        self.declare_parameter('sound_device',0)
        self.declare_parameter('robot_names',["Schimmel", "Schimmy","Shimmy"])
        self.subscription  # prevent unused variable warning
        self.tts_service = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(language_code="en-US",name="en-US-Studio-O")
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        model = GenerativeModel("gemini-1.5-pro-preview-0409",tools=[robot_agents,google_search_tool])
        self.robot_names = self.get_parameter("robot_names").value
        self.chat = model.start_chat()
        self.config = {
            "max_output_tokens": 1024,
            "temperature": 0.0,
        }
        

        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.chat.send_message("""Your name is Shimmy, which is short for Shimmel or Schimmel.
Some Facts about you can use in context when answering questions:
    * You live with a family of 6 people.
    * You live in a house in Portland Oregon. This can be helpful when answering questions about weather and time.
    * You like to be silly

""", 
                               generation_config=self.config, safety_settings=self.safety_settings)
        
        self.sample_rate_hz = 44100
        p = pyaudio.PyAudio()

        self.stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=self.sample_rate_hz,
                        output=True,
                    output_device_index=self.get_parameter("sound_device").value)
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        self.sound_q = queue.Queue()
        self.robot_runner = AgentRunner()
        t = threading.Thread(target=self.tworker)
        t.start()
        t = threading.Thread(target=self.sound_chunker)
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
    
    def get_speech(self,text):
        try:
            self.get_logger().info("processing_text %s" % (text))
            input_text = texttospeech.SynthesisInput(text=text)
            response = self.tts_service.synthesize_speech(
                input=input_text, voice=self.voice, audio_config=self.audio_config
            )
            npa = convert_mp3(response.audio_content)
            return npa
        except:
            self.get_logger().error("an error occured")
            print(traceback.format_exc()) 
    
    def read_input(self,msg):
        result = self.robot_runner.voice_emb_client.send_request(msg.embedding)
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
                            lat =response.candidates[0].content.parts[0].function_call.args["latitude"]
                            lon =response.candidates[0].content.parts[0].function_call.args["longitude"]
                            api_part = self.robot_runner.get_weather(lat,lon)
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
                            self.text_q.put(response.candidates[0].content.parts[0].text)
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
    def sound_chunker(self):
        block_size=1024
        while True:
            try:
                self.get_logger().info("waiting for audio block")
                audio_task = self.audio_q.get(block=True)
                self.get_logger().info("got audio block")
                wf = wave.open(audio_task.result())
                
                self.get_logger().info("1")
                data = wf.readframes(block_size)

                while data != b'':
                    self.stream.write(data)
                    data = wf.readframes(block_size)
                self.get_logger().info("2")
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                self.get_logger().error('Failed process sound')
                self.get_logger().error('eeks %s' % traceback.format_exc())
                               
                        
    def tworker(self):
        resp_text = ""
        nlp = spacy.load("en_core_web_sm")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                try:
                    txt = self.text_q.get(block=True, timeout=.5)
                    if txt is not None:
                        resp_text += txt
                    if len(resp_text) > 50:
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents][:-1]
                        resp_text = [sent.text for sent in doc.sents][-1]
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences)
                            try:
                                self.audio_q.put(executor.submit(self.get_speech, talk_text))
                                self.get_logger().info(f"saying {talk_text}")
                            except Exception as error:
                                self.get_logger().error(error.message)
                except queue.Empty:
                    if len(resp_text) > 0:
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents]
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences)
                            try:
                                self.audio_q.put(executor.submit(self.get_speech, talk_text))
                                self.get_logger().info(f"end saying {talk_text}")
                                resp_text = ""
                            except Exception as error:
                                self.get_logger().error(error.message)
                        
    

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.chat_text)
        if self.check(msg.chat_text,self.robot_names):
            threading.Thread(target=self.read_input, args=[msg]).start()
            



def convert_mp3(data, normalized=False):
    """MP3 to numpy array"""
    sound = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    sound = sound.set_frame_rate(44100)
    samples = np.array(sound.get_array_of_samples())
    if sound.channels == 2:
        samples = samples.reshape((-1, 2))
    fr = sound.frame_rate
    npa = samples
    buffer = io.BytesIO()
    wav.write(buffer, fr, npa)
    return buffer

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


    riva_subscriber = GoogleASRSubscriber()

    executor.add_node(riva_subscriber)

    executor.spin()

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
