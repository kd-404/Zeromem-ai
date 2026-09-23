"""Single source of truth for special tokens.

Both the tokenizer trainer and ZeroMemConfig must agree on these ids.
Order matters: this list IS the id assignment (index = token id) for the
first N ids of the vocabulary. Never reorder after a tokenizer has been
trained on real data — that silently breaks every checkpoint trained
against it.
"""

SPECIAL_TOKENS = [
    "<PAD>",         # id 0 — padding
    "<BOS>",         # id 1 — beginning of sequence
    "<EOS>",         # id 2 — end of sequence
    "<CHUNK_SEP>",   # id 3 — separates retrieved chunks in the context window
    "<KNOW>",        # id 4 — model asserts: the chunks answer this, here it is
    "<UNSURE>",      # id 5 — model asserts: partial/weak evidence, hedge
    "<REFUSE>",      # id 6 — model asserts: chunks do not answer this
    "<DONE>",        # id 7 — model asserts: no more retrieval needed (agentic stop signal)
]

PAD, BOS, EOS, CHUNK_SEP, KNOW, UNSURE, REFUSE, DONE = range(len(SPECIAL_TOKENS))
