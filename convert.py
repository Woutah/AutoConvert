"""
Convert audio files using a pretrained model
"""
import argparse
import logging
import os
import pickle
from math import ceil
import vocoders

import numpy as np
import torch

from autovc.model_vc import Generator
from config import Config
from data_converter import Converter
import soundfile as sf

logging.basicConfig(level=logging.INFO) 
log = logging.getLogger(__name__)
    


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def inference(output_dir, device, model_path, input_dir=None, input_data=None, savename = "results"): 
    # Define AutoVC model
    G = Generator(32,256,512,32).eval().to(device)
    g_checkpoint = torch.load(model_path, map_location=device) 
    G.load_state_dict(g_checkpoint['model'])
    
    print("Using model {}".format(model_path))
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)    
    
    # Load input data
    if input_data is not None:
        metadata = input_data
    elif input_dir is not None:
        metadata = pickle.load(open(os.path.join(input_dir, Config.metadata_name), "rb"))
        log.debug(input_dir)
    
    print("Starting inference...")
    
    spect_vc = []
    for src_speaker in metadata["source"].values(): 
        emb_org = torch.from_numpy(src_speaker["emb"][np.newaxis, :]).to(device) #Get source em
        
        for speaker_j in metadata["target"].keys():
            for utterance_i in src_speaker["utterances"].keys(): 
                uttr_total = np.empty(shape=(0,80)) #used to concat sub-spectrograms (2-sec parts)
                
                for sub_utterance in src_speaker["utterances"][utterance_i]: #iterate through sub-utterances (due to 2-sec limit spects. are split up if > 2 sec)
                    # utterance = src_speaker["utterances"][utterance_i] 
                    #x_org = sbmt_i[2]
                    x_org, len_pad = pad_seq(sub_utterance)
                    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
                

                    trg_speaker = metadata["target"][speaker_j]
                    emb_trg = torch.from_numpy(trg_speaker["emb"][np.newaxis, :]).to(device)
                    
                    with torch.no_grad():
                        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
                        
                    if len_pad == 0:
                        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
                    else:
                        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

                    uttr_total = np.concatenate((uttr_total, uttr_trg), axis=0) #append the split-spectrograms to create new one
                    
                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(2)
                    # ax[0].imshow(np.swapaxes(uttr_trg, 0, 1))
                    # ax[1].imshow(np.swapaxes(uttr_total, 0, 1))
                    # plt.show()
                spect_vc.append( ('{}x{}'.format(utterance_i, speaker_j), uttr_total))


    with open(os.path.join(output_dir, savename + '.pkl'), 'wb') as handle:
        pickle.dump(spect_vc, handle) 
    
    print(f"Pickled the inferred spectrograms at:{handle}")
    
    return spect_vc


def output_to_wav(output_data, vocoder):
        """Convert mel spectograms to audio files

        Args:
            output_data (list): List of mel spectograms to convert
        """
        # model = build_model()
        # # model = build_model_melgan().to(self._device)
        # checkpoint = torch.load(os.path.join(Config.dir_paths["networks"], Config.pretrained_names["vocoder"]), map_location=self._device)
        # model.load_state_dict(checkpoint["state_dict"])
        
        print("Starting vocoder...")
        for spect in output_data:
            name = spect[0]
            print(name)
            # TODO: enable this for wavenet conversion
            #------------------------------------------------
            # c = spect[1]
            # print(c.shape)
            # waveform = wavegen(model, self._device, c=c)
            #------------------------------------------------
            
            # TODO: enable this for melgan conversion
            #------------------------------------------------
            # c = cv2.resize(c, None, fx=1.0, fy=24000.0/16000.0, interpolation=cv2.INTER_AREA)
            # print(c.shape)
            # c = self.preprocess_melgan(c)
            # waveform = melgan(model, self._device, c)
            #------------------------------------------------
            
            # TODO: enable this for librosa conversion
            #------------------------------------------------
            # c = spect[1]
            # lin_out = librosa.feature.inverse.mel_to_stft(c, sr=Config.audio_sr, n_fft=Config.n_fft, fmin=Config.fmin, fmax=Config.fmax) 
            # waveform = librosa.griffinlim(lin_out, win_length=Config.win_length, hop_length=Config.hop_length)
            # name += "_librosa"
            #--------------------------------------------------
            c = spect[1]
            
            waveform = vocoder.synthesize(c)
            
            
            sf.write(os.path.join(Config.dir_paths["output"], name + ".wav"), waveform, 16000)
  
  
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Transfer voices using pretrained AutoVC')
    parser.add_argument("--source", default=None,
                        help="Source speaker folder")
    parser.add_argument("--target", default=None,
                        help="Target speaker folder")
    parser.add_argument("--source_wav", nargs='+', default=None,
                        help="Source speaker utterance")
    parser.add_argument("--model_path", type=str, default=os.path.join(Config.dir_paths["networks"], Config.pretrained_names["autovc"]),
                        help="Path to trained AutoVC model")
    parser.add_argument("--vocoder", type=str, default="griffin", choices=["griffin", "wavenet", "melgan"],
                        help="What vocoder to use")
    args = parser.parse_args()

    source_speaker = args.source if args.source is not None else "p225"
    target_speaker = args.target if args.target is not None else "p226"
    source_list = args.source_wav if args.source_wav is not None else ["p225_024"]

    # directories
    input_dir = Config.dir_paths["input"]
    converted_data_dir = Config.dir_paths["metadata"]
    output_file_dir = Config.dir_paths["output"]
    metadata_name = Config.convert_metadata_name

    if not os.path.isdir(os.path.join(input_dir, source_speaker)):
        print("Didn't find a {} folder in the {} folder".format(source_speaker, input_dir))
        exit(1)
        
    if not os.path.isdir(os.path.join(input_dir, target_speaker)):
        print("Didn't find a {} folder in the {} folder".format(target_speaker, input_dir))
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        
    if args.vocoder == "griffin":
        from vocoders import GriffinLim
        vocoder = GriffinLim(device)
    elif args.vocoder == "wavenet":
        from vocoders import WaveNet
        vocoder_path = os.path.join(Config.dir_paths["networks"], Config.pretrained_names["wavenet"])
        vocoder = WaveNet(device, vocoder_path)
    elif args.vocoder == "melgan":
        from vocoders import MelGan
        vocoder = MelGan(device)

    converter = Converter(device)

    input_data = converter.wav_to_convert_input(input_dir, source_speaker, target_speaker, source_list, converted_data_dir, metadata_name, skip_existing=True)

    output_data = inference(output_file_dir, device, args.model_path, input_data=input_data, savename=f"spects_{source_speaker}x{target_speaker}_sources_{str(*source_list)}")

    output_to_wav(output_data, vocoder)
