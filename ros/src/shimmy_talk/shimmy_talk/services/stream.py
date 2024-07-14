import riva.client.proto.riva_asr_pb2 as rasr
import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
from riva.client.auth import Auth
from typing import Dict, Generator, Iterable, List
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification,Wav2Vec2FeatureExtractor
import scipy.io.wavfile as wav
import io
import numpy as np
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer

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
        sid_model_name = "superb/wav2vec2-large-superb-sid"
        ctx_model_name = "Alibaba-NLP/gte-large-en-v1.5"
        self.auth = auth
        self.stub = rasr_srv.RivaSpeechRecognitionStub(self.auth.channel)
        self.fifo = FIFOCache(100)
        self.sid_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(sid_model_name)
        self.sid_model = Wav2Vec2ForSequenceClassification.from_pretrained(sid_model_name).to("cuda")
        # self.ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model_name)
        # self.ctx_model = AutoModel.from_pretrained(ctx_model_name, trust_remote_code=True).to("cuda")
    
    def get_sid_embeddings(self):
        emb = self.fifo.cache.copy()
        raw_audio_bytes_array = bytearray()
        for b in emb:
            raw_audio_bytes_array.extend(b)
        audio_samples = np.frombuffer(raw_audio_bytes_array, dtype=np.int16)
        buffer = io.BytesIO()
        wav.write(buffer, 16000, audio_samples)
       
        input_audio, sample_rate = librosa.load(buffer,  sr=16000)
        i= self.sid_feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate, padding=True).to("cuda")
        with torch.no_grad():
            o= self.sid_model(i.input_values, output_hidden_states=True)
        emb = o.logits.cpu().detach().numpy().tolist()[0]
        return emb
    
    # def get_ctx_embeddings(self,input_text):
    #     input_texts = [input_text]
    #     print(input_text)
    #     batch_dict = self.ctx_tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        
    #     with torch.no_grad():
    #         outputs =self.ctx_model(**batch_dict)
    #     emb =  outputs.last_hidden_state[:, 0]
    #     return emb.detach().numpy().tolist()[0]
        
    def streaming_response_generator(
        self, audio_chunks: Iterable[bytes], streaming_config: rasr.StreamingRecognitionConfig
    ):  
        generator = streaming_request_generator(audio_chunks, streaming_config,self.fifo)
        for response in self.stub.StreamingRecognize(generator, metadata=self.auth.get_auth_metadata()):
            sid_emb = []
            ctx_emb = []
            txt = ""
            
            for result in response.results:
                # txt += result.alternatives[0].transcript
                if result.is_final == True:
                    sid_emb = self.get_sid_embeddings()
                    # ctx_emb = self.get_ctx_embeddings(txt)
                    

            yield response,sid_emb,ctx_emb