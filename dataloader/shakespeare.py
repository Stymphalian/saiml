from collections import defaultdict

class ShakespeareDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        # self.alphabet = set(
        #     list("0123456789") + 
        #     list("abcdefghijklmnopqrstuvwxyz") + 
        #     list("abcdefghijklmnopqrstuvwxyz".upper()) +
        #     list(".,!?'\";:-") +
        #     list(" \t\n")
        # )
        self.alphabet = set([])
        self.freq = defaultdict(int)
        self._num_chars = None
        self.text = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

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

    def load_data(self, train_split_ratio=0.9):
        with open(self.filepath, 'r') as f:
            for line in f:
                self.process_line(line)
        self.alphabet = sorted(list(self.alphabet))

        self.text = list("".join(self.lines))
        self.split = int(0.9*len(self.text))
        self.train = self.text[:self.split]
        self.valid = self.text[self.split:]
        
        return self

    


        
