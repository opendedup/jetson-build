import rclpy
from rclpy.node import Node
from chat_interfaces.msg import FlacVADAudio, Chat
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

import io
from .asr_utils import Microphone,VAD,StartEndMonitor
from pydub import AudioSegment




import logging
from logging.handlers import RotatingFileHandler


class AGENTASR(Process):

    def __init__(self, input_queue,output_queue, use_channel: int = 0, ready_flag = None):
        super().__init__()
        self.input_queue = input_queue
        self.use_channel = use_channel
        self.ready_flag = ready_flag
        self.output_queue = output_queue
        
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

        
        
        while True:
            try:
                speech_segment = self.input_queue.get()
                raw_audio_bytes_array = bytearray()
                for chunk in speech_segment.chunks:
                    raw_audio_bytes_array.extend(chunk.audio_raw)
                flac_audio_data = self.convert_linear16_to_flac(raw_audio_bytes_array)
                try:
                    self.output_queue.put(flac_audio_data)
                except Exception as e:
                    print(f"Error extracting embeddings: {e}")
                    embeddings = None 

                
                
            except:
                print('Failed turning sound into text')
                print('%s' % traceback.format_exc())



class GCPAGENTASRPublisher(Node):
    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('gcp_agent_asr_publisher')
        self.publisher_ = self.create_publisher(FlacVADAudio, f'{namespace}/vad_audio', 10)
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

        self.logger.info("GEMINIPublisher initialized.")
        


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
            asr = AGENTASR(speech_segments, output_queue, ready_flag=asr_ready)

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
                audio = output_queue.get()
                self.logger.debug(f"Received audio from ASR queue")
                
                # Get current mic direction (handle potential errors)
                try:
                    direction = float(adjust_angle(self.r_mic.direction)) if self.r_mic else -1.0
                except Exception as mic_err:
                     self.logger.warn(f"Could not get mic direction in main loop: {mic_err}")
                     direction = -1.0 # Indicate error/unknown direction
                # self.logger.debug(f"Current Mic Direction: {direction}")

                
                msg = FlacVADAudio()
                msg.audio_data = audio.getvalue() # Read bytes from BytesIO object
                msg.direction = direction
                    
                self.publisher_.publish(msg)
                self.logger.info(f'Published audio MSG')
                

        except Exception as main_loop_err:
            self.logger.error(f'Fatal error in GEMINI ASR lworker thread: {main_loop_err}')
            self.logger.error('%s' % traceback.format_exc())
            # Consider adding cleanup or shutdown logic here
            


def main(args=None):
    rclpy.init(args=args)
    asr_publisher = GCPAGENTASRPublisher()
    

    rclpy.spin(asr_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    asr_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
