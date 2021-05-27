import os, pathlib, logging
import numpy as np

log = logging.getLogger(__name__)


def get_full_path(subpath):
    """Gets full path using the current directory (of this script) + the subpath

    Args:
        subpath (str): subpath in current directory

    Returns:
        str: the full path
    """
    cur_dir = pathlib.Path(__file__).parent.absolute()
    log.info(f"Parent path: {pathlib.Path(__file__).parent.absolute()}")
    return os.path.join(cur_dir, subpath)

def create_path(path : str):
    """creates path if it does not yet exist 

    Args:
        path (str): the full path to be created if it does not exist
    """

    if not os.path.exists(path):
        os.makedirs(path)

    


def overwrite_to_file(filename, content):
    """Simple function that (over)writes passed content to file 

    Args:
        filename (str): name of the file including extension
        content (str): what to write to file
    """
    f = open(filename, "w")
    f.write(content)
    f.close()



import pyaudio

def play_wav_from_npy(wav : np.ndarray, sample_rate = 24000):
    
    p = pyaudio.PyAudio()
    stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    )
    stream.write(np.concatenate([wav, wav, wav, wav])) #Due to some problems it needs to be played 4 times? 
    stream.start_stream()
    stream.stop_stream()
    stream.close()
    p.terminate()