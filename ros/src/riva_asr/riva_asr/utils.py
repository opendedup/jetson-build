import re
import pyaudio

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


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