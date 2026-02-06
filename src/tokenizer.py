class CharTokenizer:
    """
    Character-level tokenizer for ASR.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        # Special tokens
        self.pad_token = "<pad>"
        self.blank_token = "<blank>"

    def build_vocab(self, texts):
        """
        Build vocabulary from a list of transcriptions.
        """

        chars = set()
        for text in texts:
            for ch in text:
                chars.add(ch)

        # Sort for reproducibility
        chars = sorted(list(chars))

        vocab = [self.pad_token, self.blank_token] + chars

        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def encode(self, text):
        """
        Convert text to list of token IDs.
        """
        return [self.char2idx[ch] for ch in text]

    def decode(self, token_ids):
        """
        Convert token IDs back to string.
        """
        return "".join([self.idx2char[i] for i in token_ids])
