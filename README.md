
# Pytorch Char-RNN

This project is a Pytorch implementation of a character-level Recurrent Neural Network (Char-RNN). Char-RNN is a type of model that takes in a sequence of characters and attempts to generate a sequence of characters based on the input sequence. This project includes the following:

- A data preprocessing script to prepare data for training and testing.
- A training script to train the Char-RNN model.
- A prediction script to generate character sequences from the trained model.

This project is designed to be used as a template for building and training Char-RNN models for any type of text-based data.

## Requirements

This project requires the following libraries:

- Python 3
- Pytorch
- Numpy

## Usage

1. Clone or download this repository.
2. Add `data.txt` with your dataset. You can have multiple characters per line.
3. Edit the `train.py` script to define your model architecture and training parameters.
4. Run `python train.py` to train your model. You will see generated text as you train the model.

## Credits

This project was inspired by [Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
