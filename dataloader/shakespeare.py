from collections import defaultdict

class ShakespeareDataLoader:
    EOF = "<EOF>"

    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        self.alphabet = set(
            list("0123456789") + 
            list("abcdefghijklmnopqrstuvwxyz") + 
            list("abcdefghijklmnopqrstuvwxyz".upper()) +
            list(".,!?'\";:-") +
            list(" \t\n")
        )
        self.freq = defaultdict(int)
        self._num_chars = None
        self.x_train = None
        self.x_test = None

    def process_line(self, line):
        self.lines.append(line)
        # for word in line.split():
        #     self.vocab.add(word)
        for char in line:
            self.alphabet.add(char)
            self.freq[char] += 1

    @property
    def vocab(self):
        return self.alphabet

    def num_characters(self):
        if self._num_chars is None:
            self._num_chars = sum([len(line) for line in self.lines])
        return self._num_chars
    def num_lines(self):
        return len(self.lines)

    def load_data(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                self.process_line(line)
        self.alphabet = sorted(list(self.alphabet))
        self.alphabet.append(__class__.EOF)

        num_lines = self.num_lines()
        train_end = num_lines // 10
        self.x_train = self.lines[:train_end]
        self.x_test = self.lines[train_end:]
        return self

    


        
