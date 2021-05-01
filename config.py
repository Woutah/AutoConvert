

class Config:
    """Simple configuration class with constants/paths/etc.  used accross multiple files 
    """
    
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
    dim_emb = 256
    
    
    dir_paths = {
        "networks" : "./networks",
        "input" : "./input",
        "output" : "./output",
        "metadata" : "./metadata",
        "spectrograms" : "./spectrograms",
    }
    
    pretrained_names = {
        "speaker_encoder" : "3000000-BL.ckpt",
        "autovc" : "autovc.ckpt",
        "wavenet" : "checkpoint_step001000000_ema.pth"
        # "vocoder" : "lj_checkpoint_step000320000_ema.pth"
        # "vocoder" : "cmu_arctic_checkpoint_step000740000_ema.pth"
    }
    
    autovc_arch = {
        "dim_neck" : 32,
        "dim_emb" : dim_emb,
        "dim_pre" : 512,
        "freq" : 32
    }
    
    wavenet_arch = {
        "dim_input" : n_mels,
        "dim_cell" : 768,
        "dim_emb" : dim_emb
    }
    
    emb_len_crop = 128
    emb_num_uttr = 10
    
    len_crop = 128
    
    
    