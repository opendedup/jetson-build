import re
import pyaudio
import asyncio
from typing import List, Optional

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from scipy.spatial.distance import cosine
import multiprocessing as mp
import numpy as np
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

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q

def quaternion_to_rpy(quaternion):
  """
  Converts a quaternion to roll, pitch, and yaw angles.

  Args:
    quaternion: A numpy array representing the quaternion (w, x, y, z).

  Returns:
    A numpy array containing the roll, pitch, and yaw angles in radians.
  """

  # Extract quaternion components
  w, x, y, z = quaternion

  # Calculate roll, pitch, and yaw angles
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
  pitch = np.arcsin(2.0 * (w * y - z * x))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

  return np.array([roll, pitch, yaw])

def adjust_angle(raw_angle):
  """Adjusts the raw angle from the microphone to be 0 degrees in front.

  Args:
    raw_angle: The raw angle output from the microphone (where front is 270 degrees).

  Returns:
    The adjusted angle, where 0 degrees is in front.
  """
  adjusted_angle = (raw_angle - 270) % 360
  return adjusted_angle

def extract_text_from_dict(data, sentence_separator=" "):
    """Extracts all text from a dictionary and its nested dictionaries into a single sentence.

    Args:
    data: The dictionary to extract text from.
    sentence_separator: The separator to use between extracted text elements.

    Returns:
    A string containing all the extracted text as a sentence.
    """

    text_parts = []
    for key, value in data.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, dict):
            text_parts.append(extract_text_from_dict(value, sentence_separator))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text_parts.append(extract_text_from_dict(item, sentence_separator))

    # Remove potential unwanted characters like newline and tab
    text = re.sub(r"[\n\t]", " ", sentence_separator.join(text_parts))
    # Ensure sentence ends with a period
    if not text.endswith("."):
        text += "."
    return text

    

