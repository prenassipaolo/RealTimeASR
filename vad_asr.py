import logging
import collections
import queue
import os
from datetime import datetime
from wav2vec2_inference import Wave2Vec2Inference
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import torch
import torchaudio
from rx.subject import BehaviorSubject
import time
from sys import exit

logging.basicConfig(level=20)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 20 

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):

        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.frame_duration_ms = 1000 * self.block_size // self.sample_rate
        self.pa = pyaudio.PyAudio()

        # function to put the data into the streaming queue
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        
        def proxy_callback(in_data, frame_count, time_info, status): 
            # when audio file is used, read it
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # Change parameters if device is not None or the audio source is a file
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device
        # open audio file
        elif file is not None:
            self.chunk = self.block_size_input
            self.wf = wave.open(file, 'rb')

        # initialize stream
        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        the chosen asr model
        Args:
            data (binary): Input audio stream
        Return:
            (bytes): Resampled data in bytes form
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tobytes()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)
    
    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        # close the stream
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
    
    def write_wav(self, filename, data):
        #logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, vad_model, input_rate, prob_threshold=0.5, device=None):# TO REMOVE ): # TO REMOVE aggressiveness=3
        super().__init__(device=device, input_rate=input_rate)
        # TO REMOVE
        # self.vad = webrtcvad.Vad(aggressiveness)
        self.vad_model = vad_model
        self.prob_threshold = prob_threshold

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            # if you want to impose resampling before the process to increase the program speed, uncomment the following line
            # raise Exception("Resampling required")
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < self.block_size_input*2: # TO REMOVE 640
                return

            # TO REMOVE
            # is_speech = self.vad.is_speech(frame, self.sample_rate)
            newsound = np.frombuffer(frame, np.int16)
            audio_float32 = Int2FloatSimple(newsound)
            speech_prob = self.vad_model(audio_float32, self.sample_rate).item()
            is_speech = speech_prob >= self.prob_threshold

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


def main(ARGS):

    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
    print("ASR model in use: ", model_name)
    print('Initializing model...')
    #logging.info("ASR model: %s", model_name)
    wave_buffer = BehaviorSubject(np.array([]))
    wave2vec_asr = Wave2Vec2Inference(model_name)
    wave_buffer.subscribe(
        on_next=lambda x: asr_output_formatter(wave2vec_asr, x))

    # TO REMOVE
    """
    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.webRTC_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate)
    """

    # load silero VAD
    torchaudio.set_audio_backend("soundfile")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model=ARGS.silero_model_name,
                                  force_reload=ARGS.reload,
                                  onnx=True)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    vad_audio = VADAudio(vad_model=model,
                        input_rate=ARGS.rate) #, input_rate = 16000)
    
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to Wav2Vec 2.0 using VAD
    print("audio length\tinference time\ttext")    
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner='line')
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner:
                spinner.start()
            wav_data.extend(frame)
        else:
            if spinner:
                spinner.stop()
            #print("webRTC has detected a possible speech")
            newsound = np.frombuffer(wav_data, np.int16)
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav,
                                                datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")),
                                                wav_data)
                wav_data = bytearray()
            audio_float32 = Int2FloatSimple(newsound)
            time_stamps = get_speech_timestamps(audio_float32, model, sampling_rate=ARGS.rate)

            if(len(time_stamps) > 0):
                #print("silero VAD has detected a possible speech")              
                wave_buffer.on_next(audio_float32.numpy())
            else:
                print("VAD detected noise")
            wav_data = bytearray()



def asr_output_formatter(asr, audio):
    start = time.perf_counter()
    text = asr.buffer_to_text(audio)
    inference_time = time.perf_counter()-start
    sample_length = len(audio) / DEFAULT_SAMPLE_RATE
    print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{text}")    

def Int2FloatSimple(sound):
    return torch.from_numpy(np.frombuffer(sound, dtype=np.int16).astype('float32') / 32767)

def Int2Float(sound):
    """converts the format and normalizes the data"""
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(
        description="Stream from microphone to webRTC and silero VAD")

    parser.add_argument('-v', '--webRTC_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of webRTC: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")

    parser.add_argument('-name', '--silero_model_name', type=str, default="silero_vad",
                        help="select the name of the model. You can select between 'silero_vad',''silero_vad_micro','silero_vad_micro_8k','silero_vad_mini','silero_vad_mini_8k'")
    
    parser.add_argument('--reload', action='store_true',
                        help="download the last version of the silero vad")
    
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    
    parser.add_argument('-w', '--savewav', default=None, help="Save .wav files of utterences to given directory")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)

"""
TO DO

- undestand ring_buffer and frames management
- add previous frame to avoid first word truncation -> VADIterator: https://github.com/snakers4/silero-vad/blob/master/hubconf.py
- import the model only in inference and not in main too

"""