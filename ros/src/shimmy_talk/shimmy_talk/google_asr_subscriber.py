from concurrent.futures import ThreadPoolExecutor
import traceback
import rclpy
from rclpy.node import Node


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
import spacy
import queue
import string
import time

from .utils import deEmojify, find_audio_device_index, map_emb_distance,current_milli_time,quaternion_to_rpy,extract_text_from_dict


from .skills.agents import robot_agents,AgentRunner
from .services.stream import FIFOCache


from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler
from std_msgs.msg import String
from google import genai
from google.genai import types


class GoogleASRSubscriber(Node):

    def __init__(self,namespace="/shimmy_bot"):
        super().__init__('google_asr_subscriber')
        self.subscription = self.create_subscription(
            Chat,
            f'{namespace}/asr',
            self.listener_callback,
            10)
        self.subscription = self.create_subscription(
            String,
            f'{namespace}/system_message',
            self.system_message_callback,
            10)
        self.declare_parameter('sound_device',"miniDSP")
        self.declare_parameter('train_voice',False)
        self.declare_parameter('auto_response_threshold',0.35)
        self.declare_parameter('auto_response_timeout',20000)
        self.declare_parameter('volume_adjust',-10)
        self.declare_parameter('voice','en-US-Journey-F')
        self.declare_parameter('train_voice_name','en-US-Journey-F')
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
        self.declare_parameter('stop_on_vad', True)  # Add parameter to enable/disable this feature
        self.declare_parameter('vad_threshold', 0.6) # Default value if not set in config
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
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        )
        
        # --- Refactored Client Initialization for Vertex AI ---
        # Configure client to use Vertex AI backend
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

        self.system_prompt = self.get_parameter("prompt").value
        # Use the initialized client to get the model
        self.model = "gemini-2.5-flash-preview-04-17"
        # Note: System instruction and tools are typically passed during generation/chat start
        self.robot_names = self.get_parameter("robot_names").value
        self.config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=1.0, # Use float
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
            tools=robot_agents,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
        )

        # Start chat with system instruction and tools
        self.chat = self.client.chats.create(
            model=self.model,
            config=self.config
        )
        self.last_response = current_milli_time()
        
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
        t.daemon = True
        t.start()
        t = threading.Thread(target=self.sound_chunker)
        t.daemon = True
        t.start()
        self.stop_speaking_event = threading.Event()  # Event to stop audio playback
        self.stop_text_generation_event = threading.Event()  # Event to stop text generation
        self.is_speaking = False # New flag to track speech state
        self.speaking_lock = threading.Lock() # Lock for accessing is_speaking
        
        # --- VAD Subscriber --- 
        if self.get_parameter('stop_on_vad').value:
            self.vad_subscription = self.create_subscription(
                Chat,
                f'{namespace}/vad_detection',
                self.vad_callback,
                10) # Use default QoS (reliable)
            self.get_logger().info("VAD-based stop-talking feature enabled (Using default QoS)")
        
        self.function_handlers = {
            'use_web_browser': self.handle_use_web_browser,
            'change_led_color': self.handle_change_led_color,
            'change_brightness': self.handle_change_brightness,
            'get_power': self.handle_get_power,
            'move_around': self.handle_move_shimmy,
            'change_led_pattern': self.handle_change_led_pattern,
            'get_time': self.handle_get_time,
            'change_voice_volume': self.handle_change_voice_volume,
            'use_robot_eyes': self.handle_use_robot_eyes,
            'stop_moving': self.handle_stop_moving,
            'find_object_with_eyes': self.handle_find_object,
            'move_to_object_with_wheels': self.handle_move_to_object,
            'remember_image_objects': self.handle_remember_image_objects,
        }
        
        self.get_logger().info("GoogleASRSubscriber initialized.")
        
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


    def get_speech(self,text):
        try:
            self.get_logger().info("processing_text %s" % (text))
            
            text = deEmojify(text)
            text = text.replace("*", " ") 
            input_text = texttospeech.SynthesisInput(text=text)
            response = self.tts_service.synthesize_speech(
                input=input_text, voice=self.voice, audio_config=self.audio_config
            )
            npa = convert_wav(response.audio_content,volume_adjust=self.volume_adjust)
            return npa
        except:
            self.get_logger().error("an error occured")
            print(traceback.format_exc())
            
    
    def read_input(self,msg,person):
        self.get_logger().debug("Person %s match" %(person))
        
        if person in self.robot_names:
            self.get_logger().debug("Ignoring %s match" %(person))
        else:
            # Use generation_config and safety_settings keywords
            msg_text = f"{person} - \\\"{msg.chat_text}\\\""
            responses = self.chat.send_message_stream(msg_text)
            self.parse_responses(responses, person, msg_text)
            
    def handle_use_web_browser(self, part, person):
        context = part.function_call.args["search_txt"]
        return self.robot_runner.use_web(context)
    
    def handle_change_led_color(self, part, person):
        red = part.function_call.args["red"]
        green = part.function_call.args["green"]
        blue = part.function_call.args["blue"]
        return self.robot_runner.change_led_color(red, green, blue)
    
    def handle_change_brightness(self, part, person):
        brightness = part.function_call.args["brightness"]
        return self.robot_runner.change_brightness(brightness)
    
    def handle_get_power(self, part, person):
        return self.robot_runner.get_power()

    def handle_move_shimmy(self, part, person):
        command = part.function_call.args["move_instructions"]
        return self.robot_runner.move_shimmy(command)

    def handle_change_led_pattern(self, part, person):
        pattern = part.function_call.args["pattern"]
        return self.robot_runner.change_led_pattern(pattern)
    
    def handle_get_time(self, part, person):
        time_zone = part.function_call.args["time_zone"]
        return self.robot_runner.get_current_time(time_zone)

    def handle_change_voice_volume(self, part, person):
        volume_percent = .10
        if "volume_percent" in part.function_call.args:
            volume_percent = part.function_call.args["volume_percent"]
        increase_volume = part.function_call.args["increase_volume"]
        nvol = self.robot_runner.change_voice_volume(volume_percent, increase_volume, self.volume_adjust)
        dvol = self.volume_adjust - nvol
        msg = f"volume was adjusted by {dvol} dB"
        if nvol == self.volume_adjust and nvol > 0:
            msg = f"volume is already at its maximum"
        if nvol == self.volume_adjust and nvol < 0:
            msg = f"volume is already at its minimum"
        self.volume_adjust = nvol
        return types.Part.from_function_response(
            name=msg,
            response={
                "content": "success",
            },
        )
    
    def handle_use_robot_eyes(self, part, person):
        req = part.function_call.args["user_request"]
        prompt = f"{person} - {req}"
        if "additional_context" in part.function_call.args:
            additional_context = part.function_call.args["additional_context"]
            prompt += f"\n*** Additional Context ***\n{additional_context}"
        return self.robot_runner.get_image(prompt)
    
    def handle_stop_moving(self, part, person):
        return self.robot_runner.cancel_move()

    def handle_find_object(self, part, person):
        req = part.function_call.args["object"]
        additional_context = part.function_call.args.get("additional_context", "")
        return self.robot_runner.find_object(req, additional_context=additional_context)

    def handle_move_to_object(self, part, person):
        req = part.function_call.args["object"]
        additional_context = part.function_call.args.get("additional_context", "")
        movement_commands = part.function_call.args.get("movement_commands", "")
        return self.robot_runner.move_to_object(req, additional_context=additional_context, move_command=movement_commands)
    
    def handle_remember_image_objects(self, part, person):
        return self.robot_runner.remember_image_objects(part.function_call.args["picture_context"])
            
    def parse_responses(self, responses, person, msg):
        api_parts = []
        response_function_call_content = None # Keep track if we processed a function call

        for response in responses:
            if self.stop_text_generation_event.is_set():
                self.get_logger().info("Text generation stopped by event.")
                api_parts = [] # Clear any pending API parts
                break # Exit outer 'for response in responses' loop

            self.get_logger().info("Processing Candidate: %s" % (response))

            for part in response.candidates[0].content.parts:
                if self.stop_text_generation_event.is_set():
                    self.get_logger().info("Text generation stopped by event.")
                    api_parts = [] # Clear any pending API parts
                    break # Exit inner 'for part in parts' loop

                self.get_logger().info("Part: %s" % (part))

                # Check for and handle function calls
                if hasattr(part, 'function_call') and part.function_call:
                    fcmd = part.function_call.name
                    fargs = {key: val for key, val in part.function_call.args.items()}
                    self.get_logger().info("Function Call: %s, Args: %s" % (fcmd, fargs))

                    handler = self.function_handlers.get(fcmd)
                    if handler:
                        response_function_call_content = response.candidates[0].content # Store the content that contained the function call
                        api_part = handler(part, person)  # Pass necessary arguments to the handler
                        if api_part: # Ensure handler returned something
                             api_parts.append(api_part)
                    else:
                        self.get_logger().warning(f"No handler found for function call: {fcmd}")

                # Check for and handle text parts
                elif hasattr(part, 'text') and len(part.text) > 0:
                    self.text_q.put(part.text)
                    self.get_logger().debug(f"Text part added to queue: {part.text[:50]}...")

            # Check stop event again after processing all parts of a candidate
            if self.stop_text_generation_event.is_set():
                api_parts = [] # Clear any pending API parts
                break # Exit outer 'for response in responses' loop


        # If function calls were processed and returned results, send them back to the model
        if len(api_parts) > 0:
            self.get_logger().info(f"Sending {len(api_parts)} function response parts back to model.")
            print(api_parts) # Keep this for debugging?
            # If we didn't get interrupted, send the function responses back
            if not self.stop_text_generation_event.is_set():
                # It might be better to send the original content that contained the function call
                # along with the function response parts, depending on the expected Gemini behavior.
                # For now, just sending the api_parts.
                n_responses = self.chat.send_message_stream(
                    api_parts
                )
                # Recursively parse the new responses
                self.parse_responses(n_responses, person, msg)
            else:
                 self.get_logger().info("Skipping sending function responses due to stop event.")


        # Clear the stop event after processing is finished for this round
        if self.stop_text_generation_event.is_set():
            self.get_logger().info("Resetting text generation stop event.")
            self.stop_text_generation_event.clear()


    def sound_chunker(self):
        block_size = 1024
        self.get_logger().info("Sound chunker thread started.")
        while True:
            try:
                audio_task_future = self.audio_q.get(block=True, timeout=1.0)
                
                # is_speaking should already be True if audio got here via tworker
                # No need to set it True here
                
                try:
                     audio_buffer = audio_task_future.result()
                     if audio_buffer is None: 
                          self.get_logger().warning("Audio task resulted in None, skipping playback.")
                          # Check if queues are empty before setting is_speaking to False
                          with self.speaking_lock:
                               if self.audio_q.empty() and self.text_q.empty():
                                    self.is_speaking = False
                          continue
                     wf = wave.open(audio_buffer)
                except Exception as e:
                    self.get_logger().error(f"Error getting audio task result or opening wave: {e}")
                    self.get_logger().error('%s' % traceback.format_exc())
                    with self.speaking_lock:
                         self.is_speaking = False # Reset on error
                    continue # Skip to next audio task

                # Play audio                
                data = wf.readframes(block_size)
                while data != b'' and not self.stop_speaking_event.is_set():
                    self.stream.write(data)
                    data = wf.readframes(block_size)
                wf.close() # Ensure wave file is closed

                # --- Handling Stop Event --- 
                if self.stop_speaking_event.is_set(): 
                    self.get_logger().info("Stop speaking event detected in sound_chunker.")
                    # Clear the rest of the audio queue to prevent playing stale audio
                    while not self.audio_q.empty():
                        try:
                            stale_task = self.audio_q.get_nowait()
                            self.get_logger().debug(f"Discarded stale audio task: {stale_task}")
                        except queue.Empty:
                            break
                    self.stop_speaking_event.clear()  # Reset the event 
                    self.get_logger().info("Stop speaking event reset.")
                    # Explicitly set speaking to false after stop
                    with self.speaking_lock:
                        self.is_speaking = False 
                        self.get_logger().info("Set is_speaking to False due to stop event.")
                
                # --- Update State After Playback --- 
                self.last_response = current_milli_time()
                # Set is_speaking to False only if BOTH queues are now empty
                with self.speaking_lock:
                    if self.audio_q.empty() and self.text_q.empty(): # Check both queues
                        self.is_speaking = False
                        self.get_logger().debug("Set is_speaking to False (audio & text queues empty after playback).")
                    # else: # Keep is_speaking = True if either queue has items
                    #    self.get_logger().debug("Queues not empty, is_speaking remains True.")
                        
            except queue.Empty: # Timeout occurred
                 with self.speaking_lock:
                      # Set false only if both queues truly empty after timeout
                      if self.audio_q.empty() and self.text_q.empty() and self.is_speaking:
                           self.is_speaking = False
                           self.get_logger().debug("Set is_speaking to False (audio & text queues empty on timeout).")
                 continue
            except Exception as e:
                self.get_logger().error('Failed process sound in sound_chunker')
                self.get_logger().error('%s' % traceback.format_exc())
                with self.speaking_lock:
                    self.is_speaking = False # Reset on unexpected error
                time.sleep(0.1) # Prevent busy-loop on error
                               
                        
    def tworker(self):
        resp_text = ""
        nlp = spacy.load("en_core_web_sm")
        self.get_logger().info("Text worker thread started.")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                try:
                    txt = self.text_q.get(block=True, timeout=1.0)
                    if txt is not None:
                        # Set speaking flag as soon as text is received
                        with self.speaking_lock:
                            if not self.is_speaking:
                                 self.is_speaking = True
                                 self.get_logger().debug("Set is_speaking = True (received text in tworker)")
                        resp_text += txt
                    
                    # Check if text generation was stopped
                    if self.stop_text_generation_event.is_set():
                        self.get_logger().info("Text generation stopped by event in tworker.")
                        resp_text = "" # Clear accumulated text
                        # Clear the text queue as well
                        while not self.text_q.empty():
                            try:
                                self.text_q.get_nowait()
                            except queue.Empty:
                                break
                        self.stop_text_generation_event.clear() # Reset event
                        self.get_logger().info("Stop text generation event reset.")
                        # Also reset speaking flag on stop
                        with self.speaking_lock:
                             self.is_speaking = False
                             self.get_logger().info("Set is_speaking to False due to text generation stop event.")
                        continue # Skip synthesis for this iteration
                        
                    # Process text into sentences when enough has accumulated
                    if len(resp_text) > 20: # Threshold might need adjustment
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents][:-1]
                        remaining_text = [sent.text for sent in doc.sents][-1]
                        
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences).strip()
                            if len(talk_text) > 0:
                                try:
                                    # Submit speech synthesis task
                                    audio_future = executor.submit(self.get_speech, talk_text)
                                    self.audio_q.put(audio_future)
                                    self.get_logger().info(f"Submitted TTS for: '{talk_text[:50]}...'")
                                    # Ensure speaking is true when submitting TTS
                                    with self.speaking_lock:
                                         if not self.is_speaking:
                                              self.is_speaking = True
                                              self.get_logger().debug("Set is_speaking = True (submitted TTS)")
                                except Exception as e:
                                    self.get_logger().error(f'TTS submission failed: {e}')
                                    self.get_logger().error('%s' % traceback.format_exc())
                            resp_text = remaining_text # Keep the last partial sentence
                        else:
                             # Not enough full sentences yet, keep accumulating
                             pass 
                                
                except queue.Empty: # Timeout occurred
                    # Process any remaining text when the queue is empty after timeout
                    if len(resp_text) > 0 and not self.stop_text_generation_event.is_set():
                        doc = nlp(resp_text)
                        sentences = [sent.text for sent in doc.sents]
                        if len(sentences) > 0:
                            talk_text = " ".join(sentences).strip()
                            if len(talk_text) > 0:
                                try:
                                    audio_future = executor.submit(self.get_speech, talk_text)
                                    self.audio_q.put(audio_future)
                                    self.get_logger().info(f"Submitted final TTS for: '{talk_text[:50]}...'")
                                    # Ensure speaking is true when submitting final TTS
                                    with self.speaking_lock:
                                         if not self.is_speaking:
                                              self.is_speaking = True
                                              self.get_logger().debug("Set is_speaking = True (submitted final TTS)")
                                    resp_text = "" # Clear text after final processing
                                except Exception as e:
                                    self.get_logger().error(f'Final TTS submission failed: {e}')
                                    self.get_logger().error('%s' % traceback.format_exc())
                    # If text queue was empty, just loop again
                    continue
                except Exception as e:
                    self.get_logger().error(f'Error in tworker loop: {e}')
                    self.get_logger().error('%s' % traceback.format_exc())
                    resp_text="" # Clear text on error
                    time.sleep(0.1) # Prevent busy-loop
                        
    def system_message_callback(self, msg):
        try:
            # Prepare contents and system instruction for generate_content
            contents = [types.Part.from_text(text=msg.data)]
            # Create a specific system instruction for this context
            system_instruction_text = self.system_prompt + "\\n" + "Your job is to reword incoming statements and put it in your own words as if you were saying them.\\n"

            # Use a copy of the main config and update the system instruction if needed,
            # or create a specific config for this call. Here, we modify a copy.
            # Alternatively, if the config's system instruction is sufficient, just use self.config.
            # For this case, let's assume we NEED the specific system_instruction_text.
            # It's generally cleaner to pass the config only.
            # Let's adjust the config for this specific call.
            specific_config = types.GenerateContentConfig(
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                response_modalities=self.config.response_modalities,
                safety_settings=self.config.safety_settings, # Use safety settings from the main config
                # tools=self.config.tools # Decide if tools are needed here
                system_instruction=[types.Part.from_text(text=system_instruction_text)], # Use the specific instruction
            )


            # Call generate_content using the client's model reference
            responses = self.client.models.generate_content_stream(
                model=self.model, # Use the public model attribute directly
                contents=contents,
                config=specific_config, # Pass the adjusted config
                # Do not pass safety_settings or system_instruction separately
            )
            for response in responses:
                self.get_logger().info(response.text)
                self.text_q.put(response.text)
        except Exception:
            self.get_logger().error('%s' % traceback.format_exc())

    def listener_callback(self, msg):
        person = msg.person
        # Log the transcript (use 'Shimmy the Robot' if the person is the robot itself)
        speaker_name = "Shimmy the Robot" if person in self.robot_names else person
        self.append_to_file(msg.chat_text, speaker_name)
        # Ignore messages from the robot itself
        if person in self.robot_names:
            self.get_logger().debug(f"Ignoring ASR message from {person} (Shimmy)") # Changed to debug
            return  # Exit early 
        # Respond if the message is part of an adjacency pair or addressed to the robot
        if msg.adjacency_pairs or self.check(msg.chat_text, self.robot_names):
            if msg.stop_talking:
                self.get_logger().info("Received stop_talking=True in ASR message, stopping speech.")
                self.stop_speaking_event.set() 
                self.stop_text_generation_event.set() 
                # Consider removing this sleep or making it shorter
                # time.sleep(1) # Brief delay - This might prevent immediate stopping
            else:
                self.get_logger().info(f"Processing ASR message from {person}: '{msg.chat_text}'")
                threading.Thread(target=self.read_input, args=[msg, person]).start()
            
    def vad_callback(self, msg):
        """Callback for VAD detection - will stop shimmy from talking when user is talking"""
        self.get_logger().debug(f"Received VAD message: person='{msg.person}', prob={msg.voice_prob:.2f}")
        
        vad_stop_threshold = self.get_parameter('vad_threshold').value 
        should_stop = False
        current_is_speaking = False # Read flag within lock
        with self.speaking_lock:
             current_is_speaking = self.is_speaking # Read the current state
             # Stop if VAD is above threshold AND Shimmy is currently marked as speaking
             if msg.voice_prob > vad_stop_threshold and current_is_speaking:
                 should_stop = True
       
        # Log outside the lock
        if msg.voice_prob <= vad_stop_threshold:
             self.get_logger().debug(f"VAD prob {msg.voice_prob:.2f} <= threshold {vad_stop_threshold}")
        elif not current_is_speaking:
             self.get_logger().debug(f"VAD prob {msg.voice_prob:.2f} > threshold {vad_stop_threshold}, but Shimmy is not speaking (is_speaking=False)")
                       
        if should_stop:
            # Check event outside lock to avoid holding lock while setting events
            if not self.stop_speaking_event.is_set():
                self.get_logger().info(f"Stopping speech: VAD prob {msg.voice_prob:.2f} > {vad_stop_threshold} and is_speaking=True.")
                self.stop_speaking_event.set()       # Stop audio playback
                self.stop_text_generation_event.set()  # Stop adding new text/audio to queues


def convert_wav(data, normalized=False,volume_adjust=0):
    """Wav to numpy array"""
    sound = AudioSegment.from_file(io.BytesIO(data), format="wav")
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
