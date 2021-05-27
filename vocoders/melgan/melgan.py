import os

import torch
from parallel_wavegan.utils import download_pretrained_model, load_model
from vocoders.base_vocoder import BaseVocoder


class MelGan(BaseVocoder):
    def __init__(self, device, model_name="vctk_multi_band_melgan.v2"):
        super().__init__(device)
        
        self._model = self._build_model(model_name)
        
    def _build_model(self, model_name):
        
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        download_pretrained_model(model_name, "vocoders/melgan/models")
    
        pytorch_melgan = load_model(os.path.join(model_path, model_name, "checkpoint-1000000steps.pkl"))
        pytorch_melgan.remove_weight_norm()
      
        return pytorch_melgan
    
    def _melgan(self, c):
        model = self._model.to(self._device)
        model.eval()

        # B x C x T
        if not torch.is_tensor(c):
            c = torch.FloatTensor(c)# .unsqueeze(0)[0]
        c = c.to(self._device)

        print(c.shape)
        with torch.no_grad():
            y_hat = model.inference(c)

        y_hat = y_hat.view(-1).cpu().data.numpy()

        return y_hat
        
    def synthesize(self, mel):
        waveform = self._melgan(mel)
        return waveform
