# API
Seminar Audio Processing and Indexing 2021 final project

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

Install PyTorch using the command found [here](https://pytorch.org/get-started/locally/)

## Usage
To convert audio files, download the pretrained network weights using the instructions [here](networks/README.md). Next, place speaker audio files in the `input` folder using the following structure:

```
input
+-- speaker1
|   +-- audio1
|   |   ...
|    ...
```

Run the following command to convert a specific source audio file to sound like a target speaker

## Metadata format
Conversion data is converted to the intermediary `metadata.pkl` file used for converting. It consists of the following structure:

```
metadata.pkl
|
+-- source
|   +-- speaker1
|   |   +-- emb
|   |   +-- utterances
|   |       +-- utterance1
|   |       |   +-- part1
|   |       |   |   ...    
|   |       |   ...
|   |       
|   |   ...
|   
+-- target
    +-- speaker1
    |   +-- emb
    |   +-- utterances
    |       +-- utterance1
    |       |   +-- part1
    |       |   |   ...    
    |       |   ...
    |       
    |   ...
```