import subprocess 

class Decoder():
    def __init__(self) -> None:
        self.decoder = [
            "ffmpeg",
            "-threads", "0",
            "-nostats",
            "-loglevel", "-8",
            "-v", "quiet",
            "-probesize", "8192",
            "-hide_banner",
            "-i", "pipe:",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-sn",
            "pipe:"
        ]
    
    def decode(self, encoded_data) -> bytes:
        ffmpeg_decoding_process = subprocess.Popen(
            self.decoder,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        ffmpeg_decoding_process.stdin.write(encoded_data)
        stdout, _ = ffmpeg_decoding_process.communicate() 
        ffmpeg_decoding_process.stdin.close() 
        return stdout