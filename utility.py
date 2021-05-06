import subprocess, os, platform, pathlib, logging
log = logging.getLogger(__name__)
import numpy as np

# sys.path.insert(0, os.getcwd())


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
                    #format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    # output_device_index=2
                    )
    # i = 0
    # chunksize = 1024 * 4
    # data = [1]
    # while len(data) != 0:
    #     while(stream.get_write_available() > 0) and len(data) > 0:
    #         slice = (chunksize * i, min(chunksize * (i+1), len(wavedata)))
    #         data = wavedata[slice[0]:slice[1]]
    #         print(f"writing {slice}, of len {len(data)}")
    #         stream.write(data)
    #         i+=1

    stream.write(np.concatenate([wav, wav, wav, wav])) #Due to some problems it needs to be played 4 times? 
    stream.start_stream()
    stream.stop_stream()
    stream.close()
    p.terminate()