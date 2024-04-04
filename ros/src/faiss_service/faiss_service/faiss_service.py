from embeddings.srv import GetEmb
from embeddings.msg import Emb
from embeddings.msg import EmbResult

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn


import faiss                   # make faiss available
from sqlitedict import SqliteDict
import os
import jsonlines
import json
import numpy as np
import threading

class EmbeddingService(Node):

    def __init__(self):
        super().__init__('embedding_service')
        self.declare_parameter('channel', 'embeddings')
        self.declare_parameter('dimensions', 768)
        self.declare_parameter('fixed_embeddings','/opt/ros2/embeddings')
        self.declare_parameter('sqlpath','/opt/ros2/embeddings/db')
        self.dimensions = self.get_parameter("dimensions").value
        self.get_logger().info('Emb Dimensions = %d'% (self.dimensions))
        nbits = self.dimensions*4
        self.channel = self.get_parameter("channel").value
        
        
        self.lock = threading.Lock()
        self.increment = 0
        self.index = faiss.IndexLSH(self.dimensions,nbits)  # build the index
        dbpath = os.path.join(self.get_parameter("sqlpath").value, f"{self.channel}.sqlite")
        jsonpath = os.path.join(self.get_parameter("fixed_embeddings").value,f"{self.channel}.jsonl")
        if not os.path.exists(self.get_parameter("fixed_embeddings").value):
            os.makedirs(self.get_parameter("fixed_embeddings").value)
        if not os.path.exists(self.get_parameter("sqlpath").value):
            os.makedirs(self.get_parameter("sqlpath").value)
        self.db = SqliteDict(dbpath, autocommit=True,journal_mode="OFF")
        self.db.clear()
        if os.path.exists(jsonpath):
            self.get_logger().info('Loading json from %s'% (jsonpath))
            with jsonlines.open(jsonpath) as reader:
                for obj in reader:
                    ar = np.array(obj["value"]).astype(np.float32)
                    faiss.normalize_L2(ar)
                    self.index.add(ar)
                    del obj["value"]
                    self.db[f'key{self.increment}'] = json.dumps(obj)
                    self.increment +=1
            self.get_logger().info('Loaded %d items' % (self.increment))
        json_output = open(jsonpath, "a")
        self.json_writer = jsonlines.Writer(json_output)
        
        self.srv = self.create_service(GetEmb, 'get_emb', self.get_emb)
        self.subscription = self.create_subscription(
            Emb,
            self.channel,
            self.addemb_callback,
            10)
        self.subscription
                
                
        

    def addemb_callback(self, msg: Emb):
        self.get_logger().info('I got a message')
        kb = json.loads(msg.metadata)
        kb["key"] = msg.key
        kb["value"] = [msg.embedding]
        
        with self.lock():
            self.json_writer.write(kb)
            self.index.add()
            ar = np.array(kb["value"]).astype(np.float32)
            faiss.normalize_L2(ar)
            self.index.add(ar)
            del kb["value"]
            self.db[f'key{self.increment}'] = json.dumps(kb)
            self.increment +=1
    
    def get_emb(self, request, response):
        ar = np.array([request.embedding]).astype(np.float32)
        faiss.normalize_L2(ar)
        _d,_ids = self.index.search(ar,request.k)
        ids = _ids.tolist()[0]
        distances = _d.tolist()[0]
        response.embeddings = []
        for index, item in enumerate(ids):
            emb = EmbResult()
            emb.distance = float(distances[index])
            emb.metadata = self.db[f"key{item}"]
            response.embeddings.append(emb)
            self.get_logger().info('Loaded %s items' % (emb.metadata))
        self.get_logger().info('done')
        return response

    
        
    


def main():
    rclpy.init()

    embedding_service = EmbeddingService()

    rclpy.spin(embedding_service)


    rclpy.shutdown()


if __name__ == '__main__':
    main()