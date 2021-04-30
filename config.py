

class Config:
    """Simple configuration class with constants/paths/etc.  used accross multiple files 
    """
    
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
        "vocoder" : "checkpoint_step001000000_ema.pth"
        # "vocoder" : "lj_checkpoint_step000320000_ema.pth"
        # "vocoder" : "cmu_arctic_checkpoint_step000740000_ema.pth"
    }
    
    convert_metadata_name = "metadata.pkl"
    train_metadata_name = "train.pkl"
    
    audio_sr = 16000
    
    
    