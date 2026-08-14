import logging

from retro_eval.tokenizer_coverage import ensure_full_char_coverage

LOGGER = logging.getLogger("test_tokenizer_coverage")

UNK_ID = 2


class StubTokenizer:
    """Minimal stand-in for a SentencePiece tokenizer.

    Mirrors the trap the real fix was written for: `vocab` holds metaspace-prefixed
    pieces, so a naive "is this bare char a vocab key?" test would call every character
    missing. Only `known_chars` actually encode to something other than the unk id.
    """

    unk_token_id = UNK_ID

    def __init__(self, known_chars, size=10):
        self.known_chars = set(known_chars)
        self.vocab = {f"▁{c}": i for i, c in enumerate(sorted(known_chars))}
        self.added = []
        self._size = size

    def __call__(self, text, add_special_tokens=True):
        ids = [10 + ord(c) if c in self.known_chars else UNK_ID for c in text]
        return {"input_ids": ids}

    def __len__(self):
        return self._size + len(self.added)

    def add_tokens(self, tokens):
        new = [t for t in tokens if t not in self.known_chars]
        self.known_chars.update(new)
        self.added.extend(new)
        return len(new)


class StubModel:
    def __init__(self):
        self.resized_to = None
        self.mean_resizing = None

    def resize_token_embeddings(self, size, mean_resizing=True):
        self.resized_to = size
        self.mean_resizing = mean_resizing


def test_adds_only_genuinely_missing_chars():
    tokenizer = StubTokenizer(known_chars="CO()=c1")
    model = StubModel()

    # `.` and `K` are absent -- the CompoundT5-on-ORD case.
    added = ensure_full_char_coverage(tokenizer, model, ["CCO.CK"], LOGGER)

    assert added == 2
    assert sorted(tokenizer.added) == [".", "K"]


def test_no_resize_when_vocab_already_covers_everything():
    tokenizer = StubTokenizer(known_chars="CO.")
    model = StubModel()

    added = ensure_full_char_coverage(tokenizer, model, ["CCO.OCC"], LOGGER)

    assert added == 0
    assert tokenizer.added == []
    assert model.resized_to is None


def test_resize_uses_mean_resizing_false():
    """Guards the Kaggle-specific fix: the transformers default init produced
    pathological large-norm rows that training never recovered from."""
    tokenizer = StubTokenizer(known_chars="CO")
    model = StubModel()

    ensure_full_char_coverage(tokenizer, model, ["CCO."], LOGGER)

    assert model.mean_resizing is False
    assert model.resized_to == len(tokenizer)


def test_deduplicates_chars_across_many_texts():
    tokenizer = StubTokenizer(known_chars="CO")
    model = StubModel()

    added = ensure_full_char_coverage(tokenizer, model, ["C.C", "O.O", "CO.OC"], LOGGER)

    assert added == 1
    assert tokenizer.added == ["."]
