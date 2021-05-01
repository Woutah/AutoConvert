

class Config:
    """Simple configuration class with constants/paths/etc.  used accross multiple files 
    """
    
    dir_paths = {
        "networks" : "./networks",
        "input" : "./input",
        "output" : "./output",
        "metadata" : "./metadata",
        "spectrograms" : "./spectrograms",
        "melgan_config_path": "./vocoders/melgan/config.yml",
        "melgan_stats_path": "./vocoders/melgan/stats.h5"
    }
    
    pretrained_names = {
        "speaker_encoder" : "3000000-BL.ckpt",
        "autovc" : "autovc.ckpt",
        "wavenet" : "checkpoint_step001000000_ema.pth"
        # "vocoder" : "lj_checkpoint_step000320000_ema.pth"
        # "vocoder" : "cmu_arctic_checkpoint_step000740000_ema.pth"
    }
    
    convert_metadata_name = "metadata.pkl"
    train_metadata_name = "train.pkl"
    
    audio_sr = 16000
    n_fft = 1024
    win_length = 1024
    hop_length = 256
    n_mels = 80
    fmin = 90
    fmax = 7600
    ref_level_db = 16
    min_level_db = -100
    
    