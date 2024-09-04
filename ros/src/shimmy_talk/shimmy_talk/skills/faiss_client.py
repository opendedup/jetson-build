
import rclpy
from rclpy.node import Node
from embeddings.srv import GetEmb
from embeddings.msg import Emb
import json

class FaissClientAsync(Node):

    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('faiss_client_async')
        self.cli = self.create_client(GetEmb, f"{namespace}/get_emb")
        self.emb_publisher = self.create_publisher(Emb, f'{namespace}/embeddings', 10)
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'faiss service {namespace} not available, waiting again...')
        self.emb_publisher
        
    def send_request(self,embedding,k=1):
        emb_req = GetEmb.Request()
        emb_req.k = k
        emb_req.embedding = embedding
        future = self.cli.call_async(emb_req)
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        return future.result()
    
    def publish_embedding(self,name,embedding,map={}):
        msg = Emb()
        map["name"] = name
        msg.metadata = json.dumps(map)
        msg.embedding = embedding
        self.emb_publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.metadata)