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
    Content
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


class WalleyASRSubscriber(Node):

    def __init__(self):
        super().__init__('wally_asr_subscriber')
        self.subscription = self.create_subscription(
            Chat,
            'asr',
            self.listener_callback,
            10)
        self.declare_parameter('sound_device',0)
        self.declare_parameter('robot_names',["Walley", "Walter","Wally"])
        self.declare_parameter('prompt',"""Your name is Walley. Sometime people call you Wally or Walter. You work at Walmart and love to help people pick the perfect item based on their needs. 
Remember:
* only provide answers that are Walmart specific.
* keep your questions to a minimum
* provide specific product names for product related questions. 
""")
        self.subscription  # prevent unused variable warning
        self.tts_service = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(language_code="en-US",name="en-US-Studio-O")
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        model = GenerativeModel("gemini-1.5-pro-preview-0409",tools=[google_search_tool])
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
        self.chat.send_message(self.get_parameter("prompt").value, 
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
        self.get_logger().debug(sentence)
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
            self.get_logger().debug("processing_text %s" % (text))
            input_text = texttospeech.SynthesisInput(text=text.replace("*",""))
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
        self.get_logger().debug(result.embeddings[0].metadata)
        if len(result.embeddings) > 0:
            emb = result.embeddings[0]
            metadata = json.loads(emb.metadata)
            person  = metadata["name"]
            distance = emb.distance
            if distance < 1500 and person.startswith("robo"):
                self.get_logger().debug("Ignoring %s match is %d" %(person,distance))
            else:
                if distance > 1500:
                    self.get_logger().debug(str(distance))
                    person = "unknown"
                prompt = f"{person} - \"{msg.chat_text}\""
                self.get_logger().info(prompt)
                responses = self.chat.send_message(prompt,
                                        generation_config=self.config,stream=True, safety_settings=self.safety_settings
                                        ,tools=[google_search_tool])
                rtxt = ""
                try:
                    
                    for response in responses:
                        self.text_q.put(response.candidates[0].content.parts[0].text)
                        rtxt += response.candidates[0].content.parts[0].text
                    
                except:
                    pass
                self.get_logger().info("walley - %s" % rtxt)
                self.chat._history.append(Content.from_dict({"role":"user",
                                           "parts":[{
                                               "text":prompt
                                           }]}))
                self.chat._history.append(Content.from_dict({"role":"model",
                                           "parts":[{
                                               "text":rtxt
                                           }]}))
    def sound_chunker(self):
        block_size=1024
        while True:
            try:
                audio_task = self.audio_q.get(block=True)
                if audio_task.result() is not None:
                    wf = wave.open(audio_task.result())
                    
                    data = wf.readframes(block_size)

                    while data != b'':
                        self.stream.write(data)
                        data = wf.readframes(block_size)
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
                                self.get_logger().debug(f"walley - {talk_text}")
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
                                self.get_logger().debug(f"walley - {talk_text}")
                                resp_text = ""
                            except Exception as error:
                                self.get_logger().error(error.message)
                        
    

    def listener_callback(self, msg):
        self.get_logger().debug('I heard: "%s"' % msg.chat_text)
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


    walley_subscriber = WalleyASRSubscriber()

    executor.add_node(walley_subscriber)

    executor.spin()

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
