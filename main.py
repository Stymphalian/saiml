import os
import numpy as np

from typing import *
from pprint import pprint
import utils
import autograd2 as ag
from layers import *
from dataloader.shakespeare import ShakespeareDataLoader
from tokenizer import Tokenizer
      
    
def main():
    dl = ShakespeareDataLoader("data/shakespeare.txt")
    dl.load_data()
    print("alphabet: ", dl.alphabet)
    # pprint(dl.freq)
    # print(dl.num_characters())
    # print(dl.num_lines())
    
    tok = Tokenizer(dl.vocab)
    encoded = tok.encode(dl.x_train[0])
    decoded = tok.decode(encoded)
    print(encoded.shape)
    print(decoded)
    
    # print(dl.x[:100])

if __name__ == "__main__":
    main()
    