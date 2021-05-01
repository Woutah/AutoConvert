

class BaseVocoder:
    def __init__(self, device):
        self._device = device
        
    def synthesize(self, mel):
        raise NotImplementedError("Please subclass!")