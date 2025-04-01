# AIvsHuman
AI vs Human song classifier. The idea is to use cpu rather than gpu.

## Hugging Face space

https://huggingface.co/spaces/dkappe/AISong

## install prerequisites

`pip install torch librosa matplotlib`

and run against a directory with mp3 and wav files:

`python classify2.py directory_path`

## CR2NN model

The larger model uses two convolutional layers, two GRUs and a FC layer. Also a focal loss function.

Running against 166 human songs yields

```
163/166 = 98.2% correct
```

Running against 1075 AI songs yields

```
1054/1075 = 98% correct
```

Running against 1163 spotify AI songs yields

```
878/1163 = 75.5% correct
```

The latest net on spotify AI songs yields

```
1020/1163 = 87.7% correct
```

## Usage

If you want to detect an artist that uses AI, put several albums worth of mp3's in a directory and run the above classifiers.
