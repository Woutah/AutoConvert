
import matplotlib.pyplot as plt
import argparse
import logging
import torch
log = logging.getLogger(__name__)

def translate_lossname(lossname):
    rename_dict = {
        "G/loss_id": "Recon loss",
        "G/loss_id_psnt": "Recon loss postnet",
        "G/loss_cd": "Content loss"
    }
    # rename_dict.setdefault(None)

    name = rename_dict[lossname]
    if name: #if translation succesful
        return name 
    
    return lossname

def plot_loss_history(loss_dict : dict) -> None:
    """Plots the loss histories in a lossdict 

    Args:
        loss_dict (dict): dicionary with lossnames and loss per epoch, of the form:
            {
                "lossname1" : [(epochnr, loss), (epochnr, loss),  etc.]
                "lossname2" : etc
            }
    """
    
    for lossname in loss_dict.keys():
        log.info(f"Plotting {lossname} : {zip(*loss_dict[lossname])}")
        x,y = zip(*loss_dict[lossname])
        plt.plot(x,y, label=translate_lossname(lossname))

    plt.legend()
    plt.show()

    pass


if __name__ == "__main__":
    log.info("Starting main function in plot_loss_history.py")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/autovc_60.ckpt")
    config = parser.parse_args()

    if config.checkpoint_path == None:
        log.error("No checkpoint specified... exiting")
        exit(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        train_checkpoint = torch.load(config.checkpoint_path, map_location=device)
    except:
        log.error(f"Could not load: {config.checkpoint_path}, exiting")
        exit(0)
    plot_loss_history(train_checkpoint["loss"])
    