import rclpy
from rclpy.node import Node
from chat_interfaces.msg import Chat
import riva.client
import riva.client.audio_io


import string

import queue
import threading

from grpc._channel import _MultiThreadedRendezvous

import riva.client.proto.riva_asr_pb2 as rasr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
from riva.client.auth import Auth
from typing import Dict, Generator, Iterable, List, Optional, TextIO, Union
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification,Wav2Vec2FeatureExtractor
import scipy.io.wavfile as wav
import io
import numpy as np

class RivaASRPublisher(Node):

    def __init__(self):
        super().__init__('riva_asr_publisher')
        self.declare_parameter('device_number', 25)
        self.declare_parameter('delay', 1.0)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('robot_names',["Schimmel", "Schimmy","Shimmy"])
        self.declare_parameter('riva_uri',"localhost:50051")
        
        
        self.boosted_lm_words = self.get_parameter("robot_names").value
        self.device_number = self.get_parameter("device_number").value
        self.delay = self.get_parameter("delay").value
        self.sample_rate = self.get_parameter("sample_rate").value
    
        self.publisher_ = self.create_publisher(Chat, 'asr', 10)
        riva.client.audio_io.list_input_devices()
        # Riva ASR client configuration
        auth = riva.client.Auth(
            uri=self.get_parameter("riva_uri").value,  # Replace with your Riva server address
        )
        self.asr_service = EnASRService(auth)
        self.config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
            sample_rate_hertz=self.sample_rate,
            audio_channel_count=1,
        ),
        interim_results=False,
        )
        
        boosted_lm_score = 20.0
        riva.client.add_word_boosting_to_config(self.config, self.boosted_lm_words, boosted_lm_score)
        self.q = queue.Queue()
        t = threading.Thread(target=self.lworker)
        t.start()
        self.start_streaming()


    def lworker(self):
        resp_text = ""
        emb = []
        while True:
            try:
                ro = self.q.get(block=True, timeout=self.delay)
                if ro is not None:
                    if len(resp_text) == 0 and self.check(ro["chat"],self.boosted_lm_words):
                        resp_text = ro["chat"]
                        emb = ro["embedding"]
                    elif len(resp_text) > 0:
                        emb = ro["embedding"]
                        resp_text +=" " + ro["chat"]
            except queue.Empty:
                if len(resp_text) > 0:
                    msg = Chat()
                    msg.chat_text = resp_text
                    msg.embedding = emb
                    self.publisher_.publish(msg)
                    self.get_logger().info('Publishing: "%s"' % msg.chat_text)
                    resp_text = ""
                    emb = []
    
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
            
    def start_streaming(self):
        with riva.client.audio_io.MicrophoneStream(
            self.sample_rate,
            1600,
            self.device_number,
        ) as stream:
            for response, embedding in self.asr_service.streaming_response_generator(
                                    audio_chunks=stream,
                                    streaming_config=self.config,
            ):
                
                for result in response.results:
                    self.q.put({"chat": result.alternatives[0].transcript,"embedding":embedding})


class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []
        

    def set(self, value):
        if len(self.cache) == self.capacity:
            self.cache.pop(0)
        self.cache.append(value)

def streaming_request_generator(audio_chunks: Iterable[bytes], streaming_config: rasr.StreamingRecognitionConfig,cache: None) -> Generator[rasr.StreamingRecognizeRequest, None, None]:
        yield rasr.StreamingRecognizeRequest(streaming_config=streaming_config)
        for chunk in audio_chunks:
            resp = rasr.StreamingRecognizeRequest(audio_content=chunk)
            cache.set(chunk)
            yield resp

class EnASRService:
    """Provides streaming and offline recognition services. Calls gRPC stubs with authentication metadata."""
    def __init__(self, auth: Auth) -> None:
        """
        Initializes an instance of the class.

        Args:
            auth (:obj:`riva.client.auth.Auth`): an instance of :class:`riva.client.auth.Auth` which is used for
                authentication metadata generation.
        """
        model_name = "superb/wav2vec2-large-superb-sid"
        self.auth = auth
        self.stub = rasr_srv.RivaSpeechRecognitionStub(self.auth.channel)
        self.fifo = FIFOCache(100)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to("cuda")
    
    def get_embeddings(self):
        emb = self.fifo.cache.copy()
        raw_audio_bytes_array = bytearray()
        for b in emb:
            raw_audio_bytes_array.extend(b)
        audio_samples = np.frombuffer(raw_audio_bytes_array, dtype=np.int16)
        buffer = io.BytesIO()
        wav.write(buffer, 16000, audio_samples)
       
        input_audio, sample_rate = librosa.load(buffer,  sr=16000)
        i= self.feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate, padding=True).to("cuda")
        with torch.no_grad():
            o= self.model(i.input_values, output_hidden_states=True)
        emb = o.logits.cpu().detach().numpy().tolist()[0]
        return emb
        
    def streaming_response_generator(
        self, audio_chunks: Iterable[bytes], streaming_config: rasr.StreamingRecognitionConfig
    ):  
        generator = streaming_request_generator(audio_chunks, streaming_config,self.fifo)
        for response in self.stub.StreamingRecognize(generator, metadata=self.auth.get_auth_metadata()):
            emb = []
            for result in response.results:
                if result.is_final == True:
                   emb = self.get_embeddings()

            yield response,emb


def main(args=None):
    rclpy.init(args=args)
    riva_asr_publisher = RivaASRPublisher()
    

    rclpy.spin(riva_asr_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    riva_asr_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
