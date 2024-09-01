from concurrent.futures import ThreadPoolExecutor
import traceback
import rclpy
from rclpy.node import Node
import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    Part,
    Tool,
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

from .utils import deEmojify, find_audio_device_index, map_emb_distance,current_milli_time,quaternion_to_rpy,extract_text_from_dict


from .skills.agents import robot_agents,AgentRunner
from .services.stream import FIFOCache

import asyncio

from transformers import AutoModel, AutoTokenizer
import torch

from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler

class GoogleASRSubscriber(Node):

    def __init__(self):
        super().__init__('google_asr_subscriber')
        self.subscription = self.create_subscription(
            Chat,
            'asr',
            self.listener_callback,
            10)
        self.declare_parameter('sound_device',"miniDSP")
        self.declare_parameter('train_voice',False)
        self.declare_parameter('auto_response_threshold',0.35)
        self.declare_parameter('auto_response_timeout',20000)
        self.declare_parameter('volume_adjust',-10)
        self.declare_parameter('voice','en-US-Journey-O')
        self.declare_parameter('train_voice_name','en-US-Journey-O')
        self.declare_parameter('transcript_file','/opt/shimmy/transcript.log')
        self.declare_parameter('robot_names',["Jimmy","Schimmel", "Schimmy","Shimmy","Shimmel","Shumi","Shimi","Shami","Shiml"])
        self.declare_parameter('prompt',"""Your name is Shimmy, which is short for Schimmel.
Some Facts about you can use in context when answering questions:
    * You live with a family of 6 people.
    * You live in a house in Portland Oregon. This can be helpful when answering questions about weather and time.
    * You like to be silly but not super chatty
    * You are a small Girl Differential wheeled robot that weighs about 2 KG
    * Your favorites are as follows:
        * basketball team is the Trail Blazers
        * baseball is the Giants
        * rum rasin ice cream
        * rasin bran cerial
    * Here are some interesting facts about your body:
        * 320mm wide
        * 410mm long
        * 430mm tall
        * 2 front Wheels are 110mm in diameter
        * You have a 8x8 neopixel display on your front to create a light show or show emotions.
        * You are powered by a 3S lithium ion battery pack that has about 15 Amp hours of power.
    * Here are some details about your internals:
        * running on a Jetson Orin NX with 16GB of Ram
        * Your eyes are a zed 2 stereo camera
        * You have a speaker you talk through
        * Your hearing is a ReSpeaker Mic Array v2.0
        * You are Running ROS2 Humble on Ubuntu 22.04 Nvidia Jetpack 6.    
""")
        self.subscription  # prevent unused variable warning
        self.tts_service = texttospeech.TextToSpeechClient()
        self.voice_name = self.get_parameter("voice").value
        self.train_voice = self.get_parameter('train_voice').value
        self.auto_response_threshold = self.get_parameter('auto_response_threshold').value
        self.auto_response_timeout = self.get_parameter('auto_response_timeout').value
        self.train_voice_name =  self.get_parameter('train_voice_name').value
        self.volume_adjust=self.get_parameter('volume_adjust').value
        self.transcript_file = self.get_parameter('transcript_file').value
        self.voice = texttospeech.VoiceSelectionParams(language_code="en-US",name=self.voice_name)
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        model = GenerativeModel("gemini-flash-experimental",tools=[Tool(
        function_declarations=robot_agents)],system_instruction=[self.get_parameter("prompt").value])
        self.robot_names = self.get_parameter("robot_names").value
        
        self.chat = model.start_chat()
        self.last_response = current_milli_time()
        self.config = {
            "max_output_tokens": 1024,
            "temperature": 1,
            "top_p": 0.95,
        }
        

        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.sample_rate_hz = 44100
        sound_device = find_audio_device_index(self.get_parameter("sound_device").value)
        self.p = pyaudio.PyAudio()
        
        self.stream = self.p.open(format=self.p.get_format_from_width(2),
                        channels=1,
                        rate=self.sample_rate_hz,
                        output=True,
                    output_device_index=sound_device)
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        self.sound_q = queue.Queue()
        self.robot_runner = AgentRunner(image_system_instructions=self.get_parameter("prompt").value)
        self.emb_cache = FIFOCache(6)
        self.emb_lock = threading.Lock()
        self.transcript_logger = logging.getLogger("transcript")
        handler = RotatingFileHandler(self.transcript_file, maxBytes=10000000, backupCount=2)
        formatter = logging.Formatter('At %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.transcript_logger.addHandler(handler)
        t = threading.Thread(target=self.tworker)
        t.start()
        t = threading.Thread(target=self.sound_chunker)
        t.start()
        ctx_model_name = "Alibaba-NLP/gte-large-en-v1.5"
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model_name)
        self.ctx_model = AutoModel.from_pretrained(ctx_model_name, trust_remote_code=True).to("cuda")
        self.agent_emb = {}
        for agent in robot_agents:
            dct = agent.to_dict()
            self.agent_emb[dct["name"]] = self.get_ctx_embeddings([extract_text_from_dict(dct)])[0]   
        
        
 
    def append_to_file(self,text,person):
        if person.startswith(self.voice_name):
            person = "Shimmy the robot"
        text = f"{person} said \"{text}\""
        self.transcript_logger.info(text)
        

    def check(self,sentence, words):
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

    async def get_shimmy_talk_emb(self,text):
        emb = self.get_ctx_embeddings([text])
        with self.emb_lock:
            self.emb_cache.set(emb[0])
            
            
        
    
    def get_speech(self,text):
        try:
            self.get_logger().info("processing_text %s" % (text))
            
            text = deEmojify(text)
            asyncio.run(self.get_shimmy_talk_emb(text))
            input_text = texttospeech.SynthesisInput(text=text)
            response = self.tts_service.synthesize_speech(
                input=input_text, voice=self.voice, audio_config=self.audio_config
            )
            npa = convert_mp3(response.audio_content,volume_adjust=self.volume_adjust)
            return npa
        except:
            self.get_logger().error("an error occured")
            print(traceback.format_exc())
            
    def get_ctx_embeddings(self,input_texts):
        
        batch_dict = self.ctx_tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to("cuda")
        
        with torch.no_grad():
            outputs =self.ctx_model(**batch_dict)
        emb =  outputs.last_hidden_state[:, 0]
        return emb.cpu().detach().numpy().tolist()
    
    def read_input(self,msg,person,distance):
        #while not future.done():
        #    time.sleep(0.05)

        self.get_logger().info("Person %s match is %d" %(person,distance))
        
        if distance < 1500 and person.startswith(self.voice_name):
            self.get_logger().info("Ignoring %s match is %d" %(person,distance))
        else:
            if distance > 1500:
                self.get_logger().info(str(distance))
                person = "unknown"
            self.get_logger().info(person)
            asyncio.run(self.get_shimmy_talk_emb(msg.chat_text))
            responses = self.chat.send_message(f"{person} - \"{msg.chat_text}\"",
                                    generation_config=self.config,stream=True, safety_settings=self.safety_settings)
            self.parse_responses(responses,msg.sid_embedding,person)
            
            
            
    def parse_responses(self,responses,emb,person):
        api_parts =[]
        while responses is not None:
            
            for response in responses:
                api_part = None
                self.get_logger().info("CMD = %s" %(response.candidates[0]))
                # if response.candidates[0].finish_reason.name != "FINISH_REASON_UNSPECIFIED":
                #     self.get_logger().info("done= %s" % response.candidates[0].finish_reason)
                #     break
                for part in response.candidates[0].content.parts:
                    self.get_logger().info("Part = %s" %(part))
                    fcmd = part.function_call.name
                    if fcmd is not None and len(fcmd) > 0 :
                        self.emb_cache.set(self.agent_emb[fcmd])
                    if fcmd == "use_web_browser":
                        context = part.function_call.args["search_txt"]
                        api_part = self.robot_runner.use_web(context)
                        self.get_logger().info("web API_PART = %s" % (api_part))
                    elif fcmd == "change_led_color":
                        red =part.function_call.args["red"]
                        green =part.function_call.args["green"]
                        blue =part.function_call.args["blue"]
                        api_part = self.robot_runner.change_led_color(red,green,blue)
                    elif fcmd == "change_brightness":
                        brightness =part.function_call.args["brightness"]
                        api_part = self.robot_runner.change_brightness(brightness)
                    elif fcmd == "get_power":
                        api_part = self.robot_runner.get_power()
                    elif fcmd == "move_around":
                        command =part.function_call.args["move_instructions"]
                        api_part = self.robot_runner.move_shimmy(command)
                    elif fcmd == "change_led_pattern":
                        pattern =part.function_call.args["pattern"]
                        api_part = self.robot_runner.change_led_pattern(pattern)
                    elif fcmd == "get_time":
                        time_zone =part.function_call.args["time_zone"]
                        api_part = self.robot_runner.get_current_time(time_zone)
                    elif fcmd == "remember_voice":
                        person =part.function_call.args["name"]
                        if person not in self.robot_names:
                            api_part = self.robot_runner.add_voice(person,emb)
                        else:
                            api_part = Part.from_function_response(
                            name="remember_voice",
                            response={
                            "content": {"user_name":person},
                            }
                            )
                            
                    elif fcmd == "change_voice_volume":
                        volume_percent = .10
                        if "volume_percent" in part.function_call.args:
                            volume_percent =part.function_call.args["volume_percent"]
                        increase_volume =part.function_call.args["increase_volume"]
                        nvol = self.robot_runner.change_voice_volume(volume_percent,increase_volume,self.volume_adjust)
                        dvol = self.volume_adjust - nvol
                        msg = f"volume was ajusted by {dvol} dB"
                        if nvol == self.volume_adjust  and nvol > 0:
                            msg = f"volume is already at its maximum"
                        if nvol == self.volume_adjust  and nvol < 0:
                            msg = f"volume is already at its minimum"
                        self.volume_adjust = nvol
                        api_part = Part.from_function_response(
                            name=msg,
                            response={
                                "content": "success",
                            },
                        )
                        self.get_logger().info("Volume = %d" %(self.volume_adjust))
                    elif fcmd == "use_robot_eyes":
                        req =part.function_call.args["user_request"]
                        prompt = f"{person} - {req}"
                        if "additional_context" in part.function_call.args:
                            additional_context =part.function_call.args["additional_context"]
                            prompt +=f"\n*** Additional Context ***\n{additional_context}"
                        api_part = self.robot_runner.get_image(prompt)
                        self.get_logger().info("API_PART = %s" % (api_part))
                    elif fcmd == "stop_moving":
                        api_part = self.robot_runner.cancel_move()
                        self.get_logger().info("API_PART = %s" % (api_part))
                    elif fcmd == "find_object_with_eyes":
                        req = part.function_call.args["object"]
                        additional_context=""
                        if "additional_context" in part.function_call.args:
                            additional_context =part.function_call.args["additional_context"]
                        api_part = self.robot_runner.find_object(req,additional_context=additional_context)
                        self.get_logger().info("API_PART = %s" % (api_part))
                    elif fcmd == "turn_inplace":
                        req = part.function_call.args["turn_instructions"]
                        api_part = self.robot_runner.turn_shimmy(req)
                        self.get_logger().info("API_PART = %s" % (api_part))
                    elif fcmd == "move_to_object_with_wheels":
                        additional_context=""
                        if "additional_context" in part.function_call.args:
                            additional_context =part.function_call.args["additional_context"]
                        movement_commands=""
                        if "movement_commands" in part.function_call.args:
                            movement_commands =part.function_call.args["movement_commands"]
                        req = part.function_call.args["object"]
                        api_part = self.robot_runner.move_to_object(req,additional_context=additional_context,move_command=movement_commands)
                        self.get_logger().info("API_PART = %s" % (api_part))
                    elif fcmd == "remember_image_objects":
                        api_part = self.robot_runner.remember_image_objects(part.function_call.args["picture_context"])
                    elif len(part.text) > 0:
                            self.text_q.put(part.text)
                            self.get_logger().info(part.text)
                    if api_part != None:
                        api_parts.append(api_part)
                        responses = None
                    else:
                        responses = None
        if len(api_parts) > 0:
            n_responses = self.chat.send_message(
                        api_parts,generation_config=self.config,stream=True, safety_settings=self.safety_settings
            )
            self.parse_responses(n_responses,emb,person)
            
            
    
    def sound_chunker(self):
        block_size=1024
        while True:
            try:
                self.get_logger().info("waiting for audio block")
                audio_task = self.audio_q.get(block=True)
                self.get_logger().info("got audio block")
                wf = wave.open(audio_task.result())
                
                data = wf.readframes(block_size)

                while data != b'':
                    self.stream.write(data)
                    data = wf.readframes(block_size)
                self.last_response = current_milli_time()
            except queue.Empty:
                time.sleep(0.05)
            except Exception as e:
                self.get_logger().error('Failed process sound')
                self.get_logger().debug('%s' % traceback.format_exc())
                               
                        
    def tworker(self):
        resp_text = ""
        nlp = spacy.load("en_core_web_sm")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                try:
                    txt = self.text_q.get(block=True, timeout=.3)
                    if txt is not None:
                        resp_text += txt
                    if len(resp_text) > 20:
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents][:-1]
                        resp_text = [sent.text for sent in doc.sents][-1]
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences)
                            if len(talk_text) > 0:
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
        if len(msg.sid_embedding) > 0 and self.train_voice == True:
                self.robot_runner.add_voice(self.train_voice_name,msg.sid_embedding,training=True)
                self.get_logger().info('Training  %s' % self.train_voice_name)
        emb = self.get_ctx_embeddings([msg.chat_text])[0]
        cont_convo = False
        with self.emb_lock:
            ct = current_milli_time()
            if len(self.emb_cache.cache) > 0 and (ct - self.last_response) < self.auto_response_timeout:
                
                mp = map_emb_distance(emb=emb,cache_emb=self.emb_cache.cache)
                if(min(mp)<self.auto_response_threshold) and min(mp)>0.03:
                    cont_convo = True
        person = "unknown"
        distance = 1600
        if len(msg.sid_embedding) > 0:
            result = self.robot_runner.voice_emb_client.send_request(msg.sid_embedding)
            if len(result.embeddings) > 0:
                emb = result.embeddings[0]
                self.get_logger().info(result.embeddings[0].metadata)
                metadata = json.loads(emb.metadata)
                person  = metadata["name"]
                distance = emb.distance
            if distance > 1500:
                person = "unknown"
        fpp = str(person)
        
        if distance < 1500 and person.startswith(self.voice_name):
            fpp = "Shimmy the Robot"
        self.append_to_file(msg.chat_text,fpp)
        if cont_convo is True or self.check(msg.chat_text,self.robot_names):
            
            threading.Thread(target=self.read_input, args=[msg,person,distance]).start()
            



def convert_mp3(data, normalized=False,volume_adjust=0):
    """MP3 to numpy array"""
    sound = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    sound = sound.set_frame_rate(44100)
    if volume_adjust != 0:
        sound = sound + volume_adjust
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
    riva_subscriber.streams.stop_stream()    # "Stop Audio Recording
    riva_subscriber.streams.close()          # "Close Audio Recording
    riva_subscriber.p.close()
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
