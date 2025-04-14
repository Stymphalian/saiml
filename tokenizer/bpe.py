from collections import defaultdict
import pickle

# Reference: https://www.youtube.com/watch?v=zduSFxRajkE
class BytePairEncoder:
    def __init__(self):
        self.vocab = None
        self.pair_to_token = None

    # TODO: Don't use pickle, not secure
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            self.vocab = obj.vocab
            self.pair_to_token = obj.pair_to_token

    def fit(self, text, num_merges, start_index=256):
        text= self.to_ints(text)        

        pair_to_token = {}   # (a(int), b(int)) -> token(int)
        next_token = start_index
        for i in range(num_merges):
            counts = self.find_byte_pair_counts(text)
            pair = max(counts, key=counts.get)
            pair_to_token[pair] = next_token

            text = self.replace_pairs_with_token(text, pair, next_token)
            next_token += 1
        
        # print("Old text len:", old_text_len)
        # print("New text len:", len(text))
        # print("Compression Ratio: {:.2f}".format(old_text_len / len(text)))

        vocab = {token: [token] for token in range(256)}
        for (a, b), token in pair_to_token.items():
            vocab[token] = vocab[a] + vocab[b]

        self.vocab = vocab
        self.pair_to_token = pair_to_token
        return self.vocab, self.pair_to_token

    def encode(self, text, pair_to_token=None):
        text = self.to_ints(text)
        if pair_to_token is None:
            pair_to_token = self.pair_to_token

        token_pair_order = sorted(list(pair_to_token.values()))
        token_to_pair = {v: k for k, v in pair_to_token.items()}
        encoded_text = text
        for token in token_pair_order:
            pair = token_to_pair[token]
            encoded_text = self.replace_pairs_with_token(encoded_text, pair, token)
            # TODO: can make this more efficient by introducing an early break
        return encoded_text

    def decode(self, encoded_text, vocab=None):
        if vocab is None:
            vocab = self.vocab

        decoded = []
        for token in encoded_text:
            if token in vocab:
                decoded.append(bytes(vocab[token]))
            else:
                raise ValueError("Unknown token: {}".format(token))
        # decoded = [bytes(vocab[token]) for token in encoded_text]
        decoded = b"".join(decoded)
        decoded = decoded.decode('utf-8', errors="replace")
        return decoded

    def to_ints(self, text):
        if isinstance(text, str):
            return list(map(int, text.encode('utf-8')))
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], str):
            return list(map(int, "".join(text).encode('utf-8')))
        return text
    
    def replace_pairs_with_token(self, int_text, pair, replacement):
        new_text = []
        for i in range(len(int_text)):
            new_text.append(int_text[i])
            if len(new_text) >= 2 and (new_text[-2], new_text[-1]) == pair:
                new_text.pop(-1)
                new_text.pop(-1)
                new_text.append(replacement)

        return new_text

    def find_byte_pair_counts(self, text):
        int_text = self.to_ints(text)

        counts = defaultdict(int)
        for i in range(len(int_text) - 1):
            b1 = int_text[i]
            b2 = int_text[i+1]
            counts[(b1, b2)] += 1
        return counts
