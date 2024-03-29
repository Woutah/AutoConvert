"""
Convert audio files using a pretrained model
"""
import argparse
import logging
import os
import pickle
from math import ceil

import numpy as np
import soundfile as sf
import torch

import timer
from autovc.model_vc import Generator
from config import Config
from data_converter import Converter
from data_converter_melgan import MelganConverter

logging.basicConfig(level=logging.INFO) 
log = logging.getLogger(__name__)
	


def pad_seq(x, base=32):
	len_out = int(base * ceil(float(x.shape[0])/base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def inference(output_dir, device, model_path, input_dir=None, input_data=None, savename = "results"): 
	# Define AutoVC model
	G = Generator(**Config.autovc_arch).eval().to(device)
	g_checkpoint = torch.load(model_path, map_location=device) 
	G.load_state_dict(g_checkpoint['model'])
	
	print("Using model {}".format(model_path))
		
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)    
	
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
				uttr_total = np.empty(shape=(0, Config.n_mels)) #used to concat sub-spectrograms (2-sec parts)
				
				for sub_utterance in src_speaker["utterances"][utterance_i]: #iterate through sub-utterances (due to 2-sec limit spects. are split up if > 2 sec)
		 
					x_org, len_pad = pad_seq(sub_utterance)
					uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
					print(f"Org shape of speaker {speaker_j} - utterance {utterance_i}: first:{sub_utterance.shape}, after padding: {uttr_org.shape}")
				
					trg_speaker = metadata["target"][speaker_j]
					emb_trg = torch.from_numpy(trg_speaker["emb"][np.newaxis, :]).to(device)
					
					with torch.no_grad():
						_, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
						
					if len_pad == 0:
						uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
					else:
						uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

					uttr_total = np.concatenate((uttr_total, uttr_trg), axis=0) #append the split-spectrograms to create new one
					
				spect_vc.append( ('{}x{}'.format(utterance_i, speaker_j), uttr_total))


	with open(os.path.join(output_dir, savename + '.pkl'), 'wb') as handle:
		pickle.dump(spect_vc, handle) 
	
	print(f"Pickled the inferred spectrograms at:{handle}")
	
	return spect_vc


def output_to_wav(output_data, vocoder, output_dir, sample_rate):
		"""Convert mel spectograms to audio files

		Args:
			output_data (list): List of mel spectograms to convert
		"""
		print("Starting vocoder...")
		wall_total = 0
		cpu_total = 0
		for spect in output_data:
			# print(spect)
			name = spect[0]
			print(name)
			
			c = spect[1]
			wall_timer = timer.WallTimer()
			cpu_timer = timer.CpuTimer()

			wall_timer.start_timer()
			cpu_timer.start_timer()
			waveform = vocoder.synthesize(c)
			wall_timer.pause_timer()
			cpu_timer.pause_timer()
			print(f"Vocoder conversion time    CPU: {cpu_timer.get_time()}    Wall: {wall_timer.get_time()}")

			wall_total += wall_timer.get_time()
			cpu_total += cpu_timer.get_time()

			log.info(f"Writing inferred audio to: {output_dir}/{name}.wav")
			sf.write(os.path.join(output_dir, name + ".wav"), waveform, sample_rate)
		
		print(f"Average vocoder conversion time    CPU: {cpu_total/len(output_data)}    Wall: {wall_total/len(output_data)}")

  
  
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
	parser.add_argument("--vocoder", type=str, default="melgan", choices=["griffin", "wavenet", "melgan"],
						help="What vocoder to use")
	parser.add_argument("--force_preprocess", action="store_true",
						help="Whether to force preprocessing or not")
	parser.add_argument('--len_crop', type=int, default=0, help='dataloader output sequence length, split on this amount')
	parser.add_argument("--spectrogram_type", type=str, choices=["standard", "melgan"], default="standard",
							help="What converter to use, use 'melgan' to convert wavs to 24khz fft'd spectrograms used in the parallel melgan implementation")
	args = parser.parse_args()

	target_speaker = args.target if args.target is not None else "p225"
	source_speaker = args.source if args.source is not None else "p226"
	source_list = args.source_wav if args.source_wav is not None else ["p226_003"]

	# directories
	input_dir = Config.dir_paths["input"]
	converted_data_dir = Config.dir_paths["metadata"]
	autovc_name = os.path.split(args.model_path)[-1][:-5]
	output_file_dir = os.path.join(Config.dir_paths["output"], autovc_name)
	metadata_name = Config.convert_metadata_name

	if not os.path.isdir(os.path.join(input_dir, source_speaker)):
		print("Didn't find a {} folder in the {} folder".format(source_speaker, input_dir))
		exit(1)
		
	if not os.path.isdir(os.path.join(input_dir, target_speaker)):
		print("Didn't find a {} folder in the {} folder".format(target_speaker, input_dir))
		exit(1)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if device.type == "cuda":
		print(torch.cuda.get_device_name(0))
		
	spectrogram_type = "standard"
		
	if args.vocoder == "griffin":
		from vocoders import GriffinLim
		vocoder = GriffinLim(device)
		output_file_dir = os.path.join(output_file_dir, "griffin")
	elif args.vocoder == "wavenet":
		from vocoders import WaveNet
		vocoder_path = os.path.join(Config.dir_paths["networks"], Config.pretrained_names["wavenet"])
		vocoder = WaveNet(device, vocoder_path)
		output_file_dir = os.path.join(output_file_dir, "wavenet")
	elif args.vocoder == "melgan":
		from vocoders import MelGan
		spectrogram_type = "melgan"
		vocoder = MelGan(device)
		output_file_dir = os.path.join(output_file_dir, "melgan")

	sr = 16000
	if spectrogram_type == "standard":
		converter = Converter(device)
	elif spectrogram_type == "melgan":
		sr = 24000
		converter = MelganConverter(device, Config.dir_paths["melgan_config_path"], Config.dir_paths["melgan_stats_path"])

	skip = not args.force_preprocess
	input_data = converter.wav_to_convert_input(input_dir, source_speaker, target_speaker, source_list, converted_data_dir, metadata_name, skip_existing=skip, len_crop=args.len_crop)
	wall_timer = timer.WallTimer()
	cpu_timer = timer.CpuTimer()
	
	wall_timer.start_timer()
	cpu_timer.start_timer()

	output_data = inference(output_file_dir, device, args.model_path, input_data=input_data, savename=f"spects_{source_speaker}x{target_speaker}_lencrop{args.len_crop}_sources_{str([ str(i)+'_' for i in source_list])}")
	output_to_wav(output_data, vocoder, output_file_dir, sr)

	wall_timer.pause_timer()
	cpu_timer.pause_timer()
	print(f"Total conversion (spectrogram+vocoder) time    CPU: {cpu_timer.get_time()}    Wall: {wall_timer.get_time()}")
