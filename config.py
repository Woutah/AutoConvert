

class Config:
    """Simple configuration class with constants/paths/etc.  used accross multiple files 
    """
    
    dir_paths = {
        "networks" : "./networks",
        "input" : "./input",
        "output" : "./output",
        "metadata" : "./metadata",
        "spectograms" : "./spectograms"
    }
    
    pretrained_names = {
        "speaker_encoder" : "3000000-BL.ckpt",
        "autovc" : "autovc.ckpt",
        "vocoder" : "checkpoint_step001000000_ema.pth"
    }
    
    metadata_name = "metadata.pkl"
    
    
    