# Message for raw audio data chunk after VAD

std_msgs/Header header         # Standard ROS header (timestamp, frame_id)
int16[] audio_data             # Raw audio samples (e.g., 16-bit PCM)
float32 sample_rate            # Sample rate of the audio (e.g., 16000.0 Hz) 