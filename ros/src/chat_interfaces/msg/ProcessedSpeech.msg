# Message containing processed speech information

std_msgs/Header header      # Standard ROS header

string transcription        # Text transcription of the speech
string user_id              # Identified speaker (if diarization is done)
float32 direction_of_arrival # Angle in degrees/radians (optional, needs specific hardware)
string intent               # Detected intent
string tone                 # Detected tone/emotion
string intended_audience    # Estimated intended audience
bool adjacency_pairs      # Whether this is part of an adjacency pair with Shimmy
bool is_final             # Indicator if this is a final result for an utterance 
string audio_timestamp    # Timestamp from the audio (e.g., "0:18") 