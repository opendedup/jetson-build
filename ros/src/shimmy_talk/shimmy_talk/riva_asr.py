import rclpy
from rclpy.node import Node
from chat_interfaces.msg import Chat
import riva.client
import riva.client.audio_io

from .utils import find_audio_device_index
from .usb_4_mic_array import Tuning
import usb.core
import usb.util


import queue
import threading


from .services.stream import EnASRService

class RivaASRPublisher(Node):

    def __init__(self):
        super().__init__('riva_asr_publisher')
        self.declare_parameter('sound_device',"respeaker")
        self.declare_parameter('delay', 0.30)
        self.declare_parameter('channels', 0.30)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('robot_names',["Schimmel", "Shimmy"])
        self.declare_parameter('riva_uri',"localhost:50051")
        
        self.boosted_lm_words = self.get_parameter("robot_names").value
        self.device_number = find_audio_device_index(self.get_parameter("sound_device").value)
        self.get_logger().info("Listening Device number is %d" % (self.device_number))
        self.delay = self.get_parameter("delay").value
        self.sample_rate = self.get_parameter("sample_rate").value
        self.publisher_ = self.create_publisher(Chat, 'asr', 10)
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        self.r_mic = Tuning(dev)
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
        
        boosted_lm_score = 30.0
        riva.client.add_word_boosting_to_config(self.config, self.boosted_lm_words, boosted_lm_score)
        self.q = queue.Queue()
        t = threading.Thread(target=self.lworker)
        t.start()
        self.start_streaming()
        


    def lworker(self):
        resp_text = ""
        sid_emb = []
        ctx_emb = []
        while True:
            try:
                ro = self.q.get(block=True, timeout=self.delay)
                if ro is not None:
                    if len(resp_text) == 0:
                        resp_text = ro["chat"]
                        sid_emb = ro["sid_embedding"]
                        ctx_emb = ro["ctx_embedding"]
                    elif len(resp_text) > 0:
                        sid_emb = ro["sid_embedding"]
                        ctx_emb = ro["ctx_embedding"]
                        resp_text +=" " + ro["chat"]
            except queue.Empty:
                if len(resp_text) > 0:
                    msg = Chat()
                    msg.chat_text = resp_text
                    msg.sid_embedding = sid_emb
                    msg.ctx_embedding = ctx_emb
                    self.publisher_.publish(msg)
                    self.get_logger().debug('Publishing: "%s"' % msg.chat_text)
                    resp_text = ""
                    emb = []
    
    
            
    def start_streaming(self):
        with riva.client.audio_io.MicrophoneStream(
            self.sample_rate,
            1600,
            self.device_number,
        ) as stream:
            for response, sid_embedding,ctx_embedding in self.asr_service.streaming_response_generator(
                                    audio_chunks=stream,
                                    streaming_config=self.config,
            ):
                
                for result in response.results:
                    self.get_logger().info("Direction %d" % (self.r_mic.direction))
                    self.q.put({"chat": result.alternatives[0].transcript,"sid_embedding":sid_embedding,"ctx_embedding":ctx_embedding})


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
