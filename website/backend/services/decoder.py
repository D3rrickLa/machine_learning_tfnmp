import io
import subprocess 

class Decoder():
    def __init__(self) -> None:
        self.decoder_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f", "rawvideo",                # Input format is raw video
            "-pix_fmt", "bgr24",             # Pixel format
            "-s", "640x480",                 # Resolution
            "-r", "30",                      # Frame rate
            "-i", "-",                       # Read input from stdin
            "-an",
            "-b:v", "1M",                    # Video bitrate
            "-vf", "scale=640:480",          # Scale filter
            "-movflags", "frag_keyframe+empty_moov", # Fix typo
            "-f", "rawvideo",                     # Output format
            "-"
        ]
    
    def decode(self, encoded_data) -> list:
        ffmpeg_decoding_process = subprocess.Popen(
            self.decoder_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        frame_list = []
        frame_size = 640*480*3

        ffmpeg_decoding_process.stdin.write(encoded_data)
        ffmpeg_decoding_process.stdin.close()
        buff_stdout = io.BufferedReader(ffmpeg_decoding_process.stdout, buffer_size=10**8)

        while True:
            frame = buff_stdout.read(frame_size) 
            if len(frame) < frame_size:
                break
            frame_list.append(frame)
        ffmpeg_decoding_process.wait()
        return frame_list
