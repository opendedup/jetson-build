import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import riva.client
import riva.client.audio_io


import string

import queue
import threading


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
    
        self.publisher_ = self.create_publisher(String, 'asr', 10)
        # Riva ASR client configuration
        auth = riva.client.Auth(
            uri=self.get_parameter("riva_uri").value,  # Replace with your Riva server address
        )
        self.asr_service = riva.client.ASRService(auth)
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
        while True:
            try:
                txt = self.q.get(block=True, timeout=self.delay)
                self.get_logger().info(txt)
                if txt is not None:
                    self.get_logger().info("1")
                    if len(resp_text) == 0 and self.check(txt,self.boosted_lm_words):
                        self.get_logger().info("2")
                        resp_text = txt
                    elif len(resp_text) > 0:
                        self.get_logger().info("3")
                        resp_text +=" " + txt
            except queue.Empty:
                if len(resp_text) > 0:
                    self.get_logger().info("4")
                    msg = String()
                    msg.data = resp_text
                    self.publisher_.publish(msg)
                    self.get_logger().info('Publishing: "%s"' % msg.data)
                    resp_text = ""
    
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
        self.get_logger().info(sentence)

        # Check if all of the words in the array are in the sentence.
        for word in words:
            self.get_logger().info(word)
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
            for response in self.asr_service.streaming_response_generator(
                                    audio_chunks=stream,
                                    streaming_config=self.config,
            ):
                
                for result in response.results:
                    self.q.put(result.alternatives[0].transcript)


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
