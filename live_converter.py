
import pyaudio
import wave
import logging
import argparse
import os
import pickle
from math import ceil

import numpy as np
import soundfile as sf
import torch

from collections import OrderedDict
from autovc.model_bl import D_VECTOR
from autovc.model_vc import Generator
from config import Config
import utility
from data_converter import Converter
from vocoders import MelGan
from data_converter_melgan import MelganConverter
import threading, queue, collections
import librosa
import yaml
import hdfdict
import sklearn 
import time
from NumpyQueue import NumpyQueue, ThreadNumpyQueue
log = logging.getLogger(__name__ )
 
# FORMAT = pyaudio.paInt16
FORMAT=pyaudio.paFloat32
CHANNELS = 1
# RATE = 24000
# CHUNK = 2048
CHUNK= 4096
# RECORD_SECONDS = 10 #300 * 128 / 24000 #The amount of seconds to record --> hop size (300) * crop_len (128) = amount of samples used per prediction
WAVE_OUTPUT_FILENAME = "./sample_recording.wav"





class LiveConverter():
    def __init__(self, melgan_config, melgan_stats, device, melgan_converter, generator, speaker_encoder, target_embedding, vocoder, source_embedding, processing_buffer_size=24000 * 10):
        #==================Melgan properties=========================
        #General properties
        self.melgan_config = melgan_config
        self.sampling_rate = self.melgan_config["sampling_rate"]
        self.hop_size = self.melgan_config["hop_size"]
        self.fft_size = self.melgan_config["fft_size"]
        self.win_length = self.melgan_config["win_length"]
        self.window = self.melgan_config["window"]
        self.num_mels = self.melgan_config["num_mels"]
        self.fmin = self.melgan_config["fmin"]
        self.fmax = self.melgan_config["fmax"]

        #Trim silence
        self.trim_silence = self.melgan_config["trim_silence"]
        self.trim_top_db=self.melgan_config["trim_threshold_in_db"]
        self.trim_frame_length=self.melgan_config["trim_frame_size"]
        self.trim_hop_length=self.melgan_config["trim_hop_size"]
    

        # restore Melgan conversion scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.mean_ = melgan_stats["mean"]
        self.scaler.scale_ = melgan_stats["scale"]
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]

        #==================Other========================
        self.device= device
        self.melgan_convert = melgan_converter
        self.generator = generator
        self.speaker_encoder = speaker_encoder
        self.target_embedding = target_embedding
        self.vocoder = vocoder

        #============================================================
        self.processing_buffer_size = processing_buffer_size #Buffer size (in frames) Should always be larger than both window size/chunksize to counter delays etc.
        self.preprocessed_samples = queue.deque()
        # self.chunk_queue = queue.deque([0.0] * 2048, 2048)
        # self.chunk_queue = queue.deque([0.0] * processing_buffer_size, processing_buffer_size)
        # self.chunk_queue = NumpyQueue(processing_buffer_size, roll_when_full=False)
        log.info("Initializing Big-Chunkus")
        self.chunk_queue = ThreadNumpyQueue(size=processing_buffer_size, roll_when_full=False, dtype=np.float32)
        self.processed_spect_queue = ThreadNumpyQueue(size=  (4096, 80) , roll_when_full=False, dtype=np.float32)
        # self.processed_spect_queue.append([[0] * 80] * 256) #Insert 256 empty frames
        # self.processed_spect_queue = ThreadNumpyQueue(size=  (int(processing_buffer_size/self.hop_size), 80) , roll_when_full=False, dtype="Float32")

        self.processed_wav_queue = ThreadNumpyQueue(size=processing_buffer_size, roll_when_full=False, dtype=np.float32)
        # self.source_embed = torch.tensor(np.zeros(256, dtype=np.float32)).to(device)
        self.source_embedding = source_embedding
        

        self._unprocessed_frames = 0
        log.info("Done initializing LiveConverter")
        # self.unprocessed_frames_counter = threading.Lock()



    def original_live_playback(self):
        log.info("Now starting live playback of original sounds")

        while True:
            while len(self.chunk_queue) >= CHUNK: #if at least one frame
                # log.info(f"Cur chunkerqueue: {len(self.chunk_queue)}")
                sample = self.chunk_queue.pop(CHUNK)
                self.output_stream.write(sample)
                # self.output_stream.write(sample)
                # self.output_stream.write(sample)
        # log.info("Stopped playing live playback")

    def get_loopback_frame(self, in_data, frame_count, time_info, status):
        data = np.zeros(CHUNK)
        if len(self.chunk_queue) >= CHUNK:
            data = self.chunk_queue.pop(CHUNK)
        return (data, pyaudio.paContinue)

    def process_recording(self, in_data, frame_count, time_info, status_flags):
        # print(f"{type(in_data)}")
        if type(in_data) == bytes:
            # converted = np.frombuffer(in_data, 'Float32')
            converted = np.frombuffer(in_data, dtype="float32")
            self.chunk_queue.append(converted)
        else:
            log.info(f"Ecountered non-byte buffer of type {type(in_data)}")
        # log.info(f"Unprocessed frames: {len(self.chunk_queue)}, len converted = {len(converted)}, { np.all(converted == popped)}, {converted}, {popped}\n")
        # self.output_stream.write(popped)
        return(in_data, pyaudio.paContinue)

    
    def dynamic_vocoder(self):
        SPECT_CROP_LEN = 512
        SPECT_HOPS_PER_LEN = 2 #TODO: implement overlap between consecutive frames #ATTENTION: SHOULD BE DIVISIBLE BY SPECT CROP LEN
        # SPECT_BUFFER = 2

        while True:
            while(len(self.processed_spect_queue)) > SPECT_CROP_LEN: # //SPECT_HOPS_PER_LEN:
                log.info(f"Chunk queue size: {len(self.chunk_queue)} - Processed spect queue size: {len(self.processed_spect_queue)}, processed wave queue size: {len(self.processed_wav_queue)}")

                # log.info(f"Processed spect queue size: {len(self.processed_spect_queue)}")
                # data = self.processed_spect_queue.peek(SPECT_CROP_LEN):
                # spect = self.processed_spect_queue.peek(SPECT_CROP_LEN)
                spect = self.processed_spect_queue.pop(SPECT_CROP_LEN)
                # log.info(f"Popping {SPECT_CROP_LEN//SPECT_HOPS_PER_LEN} items")
                # self.processed_spect_queue.pop(SPECT_CROP_LEN//SPECT_HOPS_PER_LEN) 
                
                #Don't pop everything as to use overlap in next iteration
                # self.processed_spect_queue.pop(SPECT_CROP_LEN//SPECT_HOPS_PER_LEN)# - SPECT_BUFFER) 


                spect_torch = torch.from_numpy(spect[ :, :]).to(device)
                wav_data = self.vocoder.synthesize(spect_torch)
                # log.info(f"Index [{len(wav_data)//SPECT_HOPS_PER_LEN} :] of wav with size {len(wav_data)}")
                self.processed_wav_queue.append(wav_data)  
                # self.processed_wav_queue.append(wav_data[len(wav_data)//SPECT_HOPS_PER_LEN:]) 
                log.info(f"Appending {len(wav_data)} to processed_wav_queue")
                # self.processed_wav_queue.append(wav_data[len(wav_data)//SPECT_HOPS_PER_LEN - 1:])  
                

    def get_processed_frame(self, in_data, frame_count, time_info, status):
        # log.info(f"Now trying to retrieve a processed wav frame from queue of size: {len(self.processed_wav_queue)}")
        data = np.zeros(CHUNK)
            # log.info(f"len{len(self.processed_wav_queue)}")
        while True:
            if len(self.processed_wav_queue) >= CHUNK:
                data = self.processed_wav_queue.pop(CHUNK)
                log.info(f"Received a frame of size: {len(data)}")
                log.info(f"Chunk queue size: {len(self.chunk_queue)} - Processed spect queue size: {len(self.processed_spect_queue)}, processed wave queue size: {len(self.processed_wav_queue)}")
                # log.info(f"Returning wav data of length {len(data)} - Chunk queue size: {len(self.chunk_queue)} - Spect buffer size: {len(self.processed_spect_queue)} - Wav que buffer size: {len(self.processed_wav_queue)}")
                return(data, pyaudio.paContinue)
        return (data, pyaudio.paContinue)
    

    def data_processor(self):
        """Continuously generates converted spectrograms from input sounds and puts them in the converted wav queue
        """
        # CHUNK_COUNT = 40 #How many chunks to process simultaneously 
        CHUNK_COUNT = 20 # <--------- This works ok
        #========To draw spectrograms dynamically:
        # import matplotlib.pyplot as plt
        # # fig, ax = plt.subplots(1)
        # fig = plt.figure()

        #========================The conversion chain========================
        while True:
            while(len(self.chunk_queue) >= CHUNK_COUNT * self.fft_size):#self.win_length): #Process in chynks of size fft_size
                # log.info(f"Chunker queue size: {len(self.chunk_queue)}")
                # wav = self.chunk_queue.peek(CHUNK_COUNT * self.fft_size) #TODO: fft_size? or window_size here? TODO: probably window size
                
                #Optie 1
                # wav = self.chunk_queue.peek(CHUNK_COUNT * self.fft_size).copy()
                # self.chunk_queue.pop(CHUNK_COUNT * self.fft_size - self.fft_size//2)

                #Optie 1
                wav = self.chunk_queue.peek(CHUNK_COUNT * self.fft_size).copy()
                self.chunk_queue.pop(CHUNK_COUNT * self.fft_size - (self.fft_size//self.hop_size - 1) * self.hop_size)
                
                # wav = wav[:-(self.fft_size//self.hop_size - 1) * self.hop_size]
                wav = wav[: -self.hop_size]
                
                
                
                # self.chunk_queue.pop( CHUNK_COUNT * self.win_length ) #+ max(-self.win_length//2 + self.hop_size - self.win_length, self.hop_size) ) #Hop over
                
                # self.chunk_queue.pop(self.hop_size * 0.5)
                # source_spect = converter._wav_to_melgan_spec(np.array(wav).flatten(), sampling_rate)                  #            1.convert recorded audio to spect
                #===========================Trim silence=====================
                # wav, _ = librosa.effects.trim(
                #                                 wav, 
                #                                 top_db = self.trim_top_db,
                #                                 frame_length= self.trim_frame_length,
                #                                 hop_length = self.trim_hop_length
                #                             )
                
                #===========================To melgan spect===============================
                
                # get amplitude spectrogram
                x_stft = librosa.stft(wav, n_fft=self.fft_size, hop_length=self.hop_size, #hopping is done by chunk_queu.pop(self.hop_size) TODO: hop_length window size or fft_size?
                                    win_length=self.win_length, window=self.window, center= False)#, pad_mode="reflect")

                spc = np.abs(x_stft).T  # (#frames, #bins)


                # get mel basis
                fmin = 0 if self.fmin is None else self.fmin
                fmax = self.sampling_rate / 2 if self.fmax is None else self.fmax
                
                mel_basis = librosa.filters.mel(self.sampling_rate, self.fft_size, self.num_mels, fmin, fmax)
                source_spect = np.log10(np.maximum(1e-10, np.dot(spc, mel_basis.T))) #Create actual source spect
                source_spect = self.scaler.transform(source_spect)
                
                self.processed_spect_queue.append(source_spect)
                continue
                
                #===========================Through model==============================
                source_spect = torch.from_numpy(source_spect[np.newaxis, :, :]).to(self.device)                    # (to torch)

                #==========================TODO: moving average? Or not? =========================
                # self.source_embed =torch.divide(torch.add(self.source_embed, self.speaker_encoder(source_spect)), 2)                   #     2. Source spect --> source embedding

                # target_spect = simple_spect_inference(source_spect, source_embed, target_embedding, generator)  #           3. Source + target --> target spectrogram
                #===========================Pad + put through network=======================================
                len_pad = ceil(source_spect.size()[1]/32) * 32 - source_spect.size()[1]
                padded_source = torch.nn.functional.pad(input=source_spect, pad=(0, 0, 0, len_pad, 0, 0), mode='constant', value=0) #pad to base32 #TODO: why +1?? 
                with torch.no_grad():
                    _, x_identic_psnt, _ = self.generator(padded_source, self.source_embedding, self.target_embedding)
                target_spect = x_identic_psnt[0]
                target_spect = torch.nn.functional.pad(input=target_spect, pad=(0, 0, 0, -len_pad, 0, 0), mode='constant', value=0) #pad to base32 ?  
                
                # target_spect_np = target_audio[0].cpu().numpy() #load to numpy

                # target_audio = vocoder.synthesize(target_spect[0])
                self.processed_spect_queue.append(target_spect[0])


                # ============== To draw spectrograms =================:
                # plt.clf()
                # plt.imshow(self.processed_spect_queue.peek(len(self.processed_spect_queue)).T)
                # fig.canvas.draw()
                # plt.pause(0.0001)


    def start_main_loop(self):
        log.info("Starting main loop...")        
        #================= Recording stream ============================
        self.audio = pyaudio.PyAudio() 
        self.recording_stream = self.audio.open(format=FORMAT, 
                                                channels=CHANNELS,
                                                rate=self.sampling_rate, 
                                                input=True,
                                                # frames_per_buffer=CHUNK, 
                                                stream_callback = self.process_recording
                                                )


        #=================outputting stream =============================
        self.audio_output = pyaudio.PyAudio()
        self.output_stream = self.audio_output.open(
                                                format=FORMAT,
                                                #format=pyaudio.paFloat32,
                                                channels=CHANNELS,
                                                rate=self.sampling_rate,
                                                frames_per_buffer=CHUNK, 
                                                output=True,
                                                stream_callback=self.get_processed_frame
                                                # stream_callback=self.get_loopback_frame
                                                # output_device_index=2
                                            )

        # log.info("Now starting continuous conversion process")
        # while True:
        #     time.sleep(1)
        log.info("Started audio streams...")      
        spect_processing_thread = threading.Thread(target=self.data_processor)
        spect_processing_thread.daemon=True
        spect_processing_thread.start() #Continuously convert wavs to spect and add them to spect queu
        log.info("Started threads... Now creating dynamic vocoder")
        # vocoder_thread = threading.Thread(target=self.dynamic_vocoder)
        self.dynamic_vocoder()


        
        


    
    def wav_block_to_spect(self, block):
        x_stft = librosa.stft(block, n_fft = self.fft_size, 
                                hop_length = self.fft_size, #TODO: hop size is defined in the stream, here, it should be either fft_size or window size
                                win_length = self.win_length,
                                window = self.window,
                                pad_mode = "reflect",
                                center = True #TODO: does this introduce artifacts? 
                                ) 
        spec = np.abs(x_stft).T #Transpose and absolute
        fmin = 0 if self.fmin is None else self.fmin
        fmax = self.sampling_rate / 2 if fmax is None else self.fmax
        mel_basis = librosa.filters.mel(self.sampling_rate, self.fft_size, self.num_mels, self.fmin, self.fmax)
        return np.log10(np.maximum(1e-10, np.dot(spec, mel_basis.T)))


    # def read_recording_stream(self):
    #     #Read a single block of appropriate size from the recording stream if available
        
    
    # def continuous_converter(self):
    #     log.info("Starting continuous conversion")
        
    #     # librosa.stream()
    #     while self.recording: #While recording
    #         for block in self.fft_conversion_stream:
    #             log.info(f"Converting - cur spect queue length = {len(self.spectrogram_queue)}")
    #             self.spectrogram_queue.append(wav_block_to_spect(self, block)) #Append 
    #             pass
    

    
    def preprocess_samples(self):
        pass

        """
        print("Converting using wav to melgan!")
        if self.melgan_config["trim_silence"]:
            wav, _ = librosa.effects.trim(wav,
                                            top_db=self.melgan_config["trim_threshold_in_db"],
                                            frame_length=self.melgan_config["trim_frame_size"],
                                            hop_length=self.melgan_config["trim_hop_size"])

        if introduce_noise:
            log.error(f"Introduce_noise is set tot {introduce_noise}, however, this is not implemented. Exiting...")
            exit(0)

        if sample_rate != self.melgan_config["sampling_rate"]: #Resampling
            # log.inf("Resampling ")
            wav = librosa.resample(wav, sample_rate, self.melgan_config["sampling_rate"])
            print(f"Wav file with sr {sample_rate} != {self.melgan_config['sampling_rate']}, Now resampling to {self.melgan_config['sampling_rate']}")

        mel = self.logmelfilterbank( #Create mel spectrogram using the melGAN settings
                        wav,  
                        sampling_rate=self.melgan_config["sampling_rate"],
                        hop_size=self.melgan_config["hop_size"],
                        fft_size=self.melgan_config["fft_size"],
                        win_length=self.melgan_config["win_length"],
                        window=self.melgan_config["window"],
                        num_mels=self.melgan_config["num_mels"],
                        fmin=self.melgan_config["fmin"],
                        fmax=self.melgan_config["fmax"])
        
        # make sure the audio length and feature length are matched
        wav = np.pad(wav, (0, self.melgan_config["fft_size"]), mode="reflect")
        wav = wav[:len(mel) * self.melgan_config["hop_size"]]
        assert len(mel) * self.melgan_config["hop_size"] == len(wav)

        #================================================Normalization=========================================================
        # restore scaler
        scaler = StandardScaler()
        if self.melgan_config["format"] == "hdf5":
            scaler.mean_ = read_hdf5(self.melgan_stats_path, "mean")
            scaler.scale_ = read_hdf5(self.melgan_stats_path, "scale")
        # elif config["format"] == "npy":
            # scaler.mean_ = np.load(args.stats)[0]
            # scaler.scale_ = np.load(args.stats)[1]
        else:
            raise ValueError("support only hdf5 (and normally npy - but not now) format.... cannot load in scaler mean/scale, exiting")
            exit(0)
        # from version 0.23.0, this information is needed
        scaler.n_features_in_ = scaler.mean_.shape[0]
        mel = scaler.transform(mel)
        return mel
        """
    


def test_convert(args, device, melgan_converter, melgan_config, generator, speaker_encoder, vocoder, sampling_rate=24000):

    TEST_LENGTH = 192000
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=sampling_rate, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
    
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    
    chunk_queue = ThreadNumpyQueue(size=TEST_LENGTH, roll_when_full=False)
    
    for i in range(3):
        while len(chunk_queue) < TEST_LENGTH:
            data = stream.read(CHUNK)
            chunk_queue.append(np.frombuffer(data, dtype=np.float32))
        chunk_queue.pop(TEST_LENGTH)

    while len(chunk_queue) < TEST_LENGTH:
        data = stream.read(CHUNK)
        chunk_queue.append(np.frombuffer(data, dtype=np.float32))

    #General properties
    sampling_rate = melgan_config["sampling_rate"]
    hop_size = melgan_config["hop_size"]
    fft_size = melgan_config["fft_size"]
    win_length = melgan_config["win_length"]
    window = melgan_config["window"]
    num_mels = melgan_config["num_mels"]
    fmin = melgan_config["fmin"]
    fmax = melgan_config["fmax"]


    x_stft = librosa.stft(chunk_queue.pop(TEST_LENGTH), n_fft=fft_size, hop_length=hop_size, #hopping is done by chunk_queu.pop(self.hop_size) TODO: hop_length window size or fft_size?
                    win_length=win_length, window=window, center= True, pad_mode="reflect")

    spc = np.abs(x_stft).T  # (#frames, #bins)


    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    source_spect = np.log10(np.maximum(1e-10, np.dot(spc, mel_basis.T))) #Create actual source spect

    #======================Scaling==========================
    
    # restore Melgan conversion scaler
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.mean_ = melgan_stats["mean"]
    scaler.scale_ = melgan_stats["scale"]
    scaler.n_features_in_ = scaler.mean_.shape[0]

    source_spect = scaler.transform(source_spect)
    
    processed_spect_queue = ThreadNumpyQueue(size=(2048, 80), roll_when_full=False)
    processed_spect_queue.append(source_spect)

    # total_data = []
    # while len(total_data) < 48000:
    #     data = stream.read(2000)
    #     new_dat = list(np.frombuffer(data, "Float32"))
    #     total_data.extend(new_dat)
    SPECT_CROP_LEN = 512
    SPECT_HOPSIZE = 1 #TODO: implement overlap between consecutive frames

    processed_wav_queue = ThreadNumpyQueue(size=TEST_LENGTH, roll_when_full=False)
    while len(processed_spect_queue) > SPECT_CROP_LEN:
        
        log.info(f"Processed spect queue size: {len(processed_spect_queue)}")
        # data = processed_spect_queue.peek(SPECT_CROP_LEN):
        spect = processed_spect_queue.pop(min(SPECT_CROP_LEN, len(processed_spect_queue))) #get data
        # spect_torch = torch.from_numpy(spect[ :, :]).to(device)
        wav_data = vocoder.synthesize(spect)
        processed_wav_queue.append(wav_data)  
        log.info(f"Done processing 1 --> Processed spect queue size: {len(processed_spect_queue)}, processed wave queue size: {len(processed_wav_queue)}")
        
    print("finished recording")
    


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    #========================The conversion chain========================
    wav = processed_wav_queue.pop(len(processed_wav_queue))     
    # wav = total_data
              

    while True:
        utility.play_wav_from_npy(wav)

    
    




def live_convert(args, device, melgan_converter, generator, speaker_encoder, vocoder, sampling_rate=24000):
    test_convert(args, device, melgan_converter, generator, speaker_encoder, vocoder, sampling_rate)

    #=======================Load in target speaker encoding (npy array)=========================
    target_embedding = np.load(args.target_embedding_path) #load in target embedding
    target_embedding = torch.from_numpy(target_embedding[np.newaxis, :]).to(device)

    #==========================Record a sample=========================
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=sampling_rate, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
    
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    data = stream.read(58000)
    print("finished recording")
        

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    #========================The conversion chain========================
    wav = np.frombuffer(data, 'Float32')                                                            #(to float32 wav array)
    # while True:  
    #     utility.play_wav_from_npy(wav, RATE)
    #                                                # (to float32 wav)
    source_spect = converter._wav_to_melgan_spec(np.array(wav).flatten(), sampling_rate)                  #            1.convert recorded audio to spect
    source_spect = torch.from_numpy(source_spect[np.newaxis, :, :]).to(device)                    # (to torch)


    source_embed = speaker_encoder(source_spect)                                                    #           2. Source spect --> source embedding

    # target_spect = simple_spect_inference(source_spect, source_embed, target_embedding, generator)  #           3. Source + target --> target spectrogram

    len_pad = ceil(source_spect.size()[1]/32) * 32 - source_spect.size()[1]
    padded_source = torch.nn.functional.pad(input=source_spect, pad=(0, 0, 0, len_pad, 0, 0), mode='constant', value=0) #pad to base32 #TODO: why +1?? 
    with torch.no_grad():
        _, x_identic_psnt, _ = G(padded_source, source_embed, target_embedding)
    target_spect = x_identic_psnt[0]
    target_spect = torch.nn.functional.pad(input=target_spect, pad=(0, 0, 0, -len_pad, 0, 0), mode='constant', value=0) #pad to base32 #TODO: why +1?? 
    

    target_audio = vocoder.synthesize(target_spect[0])

    # target_audio_np = target_audio[0].cpu().numpy() #load to numpy
    while True:
        utility.play_wav_from_npy(target_audio)

    
    # waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # waveFile.setnchannels(CHANNELS)
    # waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    # waveFile.setframerate(RATE)
    # waveFile.writeframes(b''.join(frames))
    # waveFile.close()




if __name__ == "__main__":
    #======================Logging==========================
    logging.basicConfig(level=logging.INFO)
    
    #=======================Device==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #=======================Arguments=======================
    parser = argparse.ArgumentParser(description='Transfer voices using pretrained AutoVC')
    # parser.add_argument("--model_path", type=str, default="./checkpoints/20210504_melgan_lencrop514_autovc_1229998.ckpt",
    #                     help="Path to trained AutoVC model")
    parser.add_argument("--model_path", type=str, default="./checkpoints/20210504_melgan_lencrop514_autovc_1229998.ckpt",
                        help="Path to trained AutoVC model")
    # parser.add_argument("--vocoder", type=str, default="griffin", choices=["griffin", "wavenet", "melgan"],
    #                     help="What vocoder to use")
    parser.add_argument("--target_embedding_path", type=str, default="./spectrograms/p226/p226_emb.npy",
                        help="What embedding to use, path to target embedding .npy file with shape: [256], source embedding is calculated dynamically")
    parser.add_argument("--source_embedding_path", type=str, default="./spectrograms/Wouter/Wouter_emb.npy",
                        help="What embedding to use, path to target embedding .npy file with shape: [256], source embedding is calculated dynamically")
    # parser.add_argument("--spectrogram_type", type=str, choices=["standard", "melgan"], default="standard",
    #                         help="What converter to use, use 'melgan' to convert wavs to 24khz fft'd spectrograms used in the parallel melgan implementation")
    args = parser.parse_args()

    #=======================Data Converter (spectrogram etc.)========================
    # if args.spectrogram_type == "standard":
    #     converter = Converter(device)
    # elif args.spectrogram_type == "melgan":
    converter = MelganConverter(device, Config.dir_paths["melgan_config_path"], Config.dir_paths["melgan_stats_path"])


    #=======================Load in spectrogram converter============================
    G = Generator(**Config.autovc_arch).eval().to(device)
    g_checkpoint = torch.load(args.model_path, map_location=device) 
    G.load_state_dict(g_checkpoint['model'])

    #=======================Load in speaker speaker encoder========================
    network_dir = Config.dir_paths["networks"]
    speaker_encoder_name = Config.pretrained_names["speaker_encoder"]
    speaker_encoder = D_VECTOR(**Config.wavenet_arch).eval().to(device)
    c_checkpoint = torch.load(os.path.join(network_dir, speaker_encoder_name), map_location=device)     

    
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    speaker_encoder.load_state_dict(new_state_dict)
    
    #=======================Load in vocoder==================================

    vocoder = MelGan(device)
    try:
        with open(Config.dir_paths["melgan_config_path"]) as f:
            melgan_config = yaml.load(f, Loader=yaml.Loader)
        
        melgan_stats = hdfdict.load(Config.dir_paths["melgan_stats_path"])
    except Exception as err:
        log.error(f"Unable to load in melgan config or stats ({err})")
        exit(0)


    target_embedding = np.load(args.target_embedding_path) #load in target embedding
    target_embedding = torch.from_numpy(target_embedding[np.newaxis, :]).to(device)


    source_embedding = np.load(args.source_embedding_path)
    source_embedding = torch.from_numpy(source_embedding[np.newaxis, :]).to(device)

    #========================Main loop start===========================
    log.info("Started live_convert.py")
    # live_convert(args, device, converter, melgan_config, G, speaker_encoder, vocoder)
    voice_converter = LiveConverter(melgan_config, melgan_stats, device, converter, G, speaker_encoder, target_embedding, vocoder, source_embedding)
    try:
        voice_converter.start_main_loop()
    except Exception as err:
        log.error({err})
        exit(0)