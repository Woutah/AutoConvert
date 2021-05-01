import torch
from autovc.hparams import hparams
from tqdm import tqdm
from vocoders.base_vocoder import BaseVocoder
from wavenet_vocoder import builder


class WaveNet(BaseVocoder):
    def __init__(self, device, model_path):
        super().__init__(device)
        
        self._model = self._build_model(model_path)
        
        
    def _build_model(self, model_path):
        model = getattr(builder, hparams.builder)(
            out_channels=hparams.out_channels,
            layers=hparams.layers,
            stacks=hparams.stacks,
            residual_channels=hparams.residual_channels,
            gate_channels=hparams.gate_channels,
            skip_out_channels=hparams.skip_out_channels,
            cin_channels=hparams.cin_channels,
            gin_channels=hparams.gin_channels,
            weight_normalization=hparams.weight_normalization,
            n_speakers=hparams.n_speakers,
            dropout=hparams.dropout,
            kernel_size=hparams.kernel_size,
            upsample_conditional_features=hparams.upsample_conditional_features,
            upsample_scales=hparams.upsample_scales,
            freq_axis_kernel_size=hparams.freq_axis_kernel_size,
            scalar_input=True,
            legacy=hparams.legacy,
        )
        
        checkpoint = torch.load(model_path, map_location=self._device)
        model.load_state_dict(checkpoint["state_dict"])
        
        return model
    
    
    def _wavegen(self, c):
        """Generate audio using WaveNet

        Args:
            c (np.array): Mel spectogram. Assumed to be normalized using the AutoVC method

        Returns:
            np.array: Audio time series
        """
        model = self._model.to(self._device)
        model.eval()
        model.make_generation_fast_()

        Tc = c.shape[0]
        upsample_factor = hparams.hop_size
        # Overwrite length according to feature size
        length = Tc * upsample_factor

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

        initial_input = torch.zeros(1, 1, 1).fill_(0.0)

        # Transform data to GPU
        initial_input = initial_input.to(self._device)
        c = c.to(self._device)

        with torch.no_grad():
            y_hat = model.incremental_forward(
                initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=hparams.log_scale_min)

        y_hat = y_hat.view(-1).cpu().data.numpy()

        return y_hat
        
        
    def synthesize(self, mel):
        waveform = self._wavegen(mel)
        return waveform
