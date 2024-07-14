import re
import pyaudio
import asyncio
from typing import List, Optional

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from scipy.spatial.distance import cosine
import multiprocessing as mp
import time



def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def get_text_emb( 
    texts: List[str] = ["banana muffins?", "banana bread? banana muffins?"],
    task: str = "CLUSTERING",
    model_name: str = "text-embedding-004",
    dimensionality: Optional[int] = 256,)-> List[List[float]]:
        """Embeds texts with a pre-trained, foundational model."""
        model = TextEmbeddingModel.from_pretrained(model_name)
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
        embeddings = model.get_embeddings(inputs, **kwargs)
        return [embedding.values for embedding in embeddings]


def current_milli_time():
    return round(time.time() * 1000)

def find_audio_device_index(device_name):

    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount")
    device_index = None
    for i in range(num_devices):

        device_info = p.get_device_info_by_host_api_device_index(0, i)
        print(device_info.get("name"))
        if device_name.lower() in device_info.get("name").lower():

            device_index = i
            break

    return device_index

def get_emb_distance(tpl):
    distance = cosine(tpl[0], tpl[1])
    return distance

pool = mp.Pool(processes = mp.cpu_count())
def map_emb_distance(emb,cache_emb):
    embs = []
    for c_emb in cache_emb:
        embs.append((c_emb,emb))
    dists = pool.map(get_emb_distance, embs)
    return dists

    

