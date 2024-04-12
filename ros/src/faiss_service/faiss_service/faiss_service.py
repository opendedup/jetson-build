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
        self.declare_parameter('dimensions', 768)
        self.declare_parameter('embeddings_path','/opt/ros2/embeddings')
        self.dimensions = self.get_parameter("dimensions").value
        self.get_logger().info('Emb Dimensions = %d'% (self.dimensions))
        nbits = self.dimensions*4
        self.lock = threading.Lock()
        self.increment = 0
        self.index = faiss.IndexLSH(self.dimensions,nbits)  # build the index
        dbpath = os.path.join(self.get_parameter("embeddings_path").value,"db", f"embeddings.sqlite")
        jsonpath = os.path.join(self.get_parameter("embeddings_path").value,f"embeddings.jsonl")
        if not os.path.exists(self.get_parameter("embeddings_path").value):
            os.makedirs(self.get_parameter("embeddings_path").value)
        if not os.path.exists(os.path.join(self.get_parameter("embeddings_path").value,"db")):
            os.makedirs(os.path.join(self.get_parameter("embeddings_path").value,"db"))
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
            'embeddings',
            self.addemb_callback,
            10)
        self.subscription
                
                
        

    def addemb_callback(self, msg: Emb):
        kb = json.loads(msg.metadata)
        mm = np.array([msg.embedding]).astype(np.float32).tolist()
        kb["value"] = mm
        
        with self.lock:
            self.json_writer.write(kb)
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
        response.embedding = request.embedding
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