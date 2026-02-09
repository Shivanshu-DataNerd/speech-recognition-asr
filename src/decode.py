def greedy_decode(log_probs, tokenizer):
    """
    log_probs: (time, batch, vocab)
    """
    predictions = log_probs.argmax(dim=-1).transpose(0, 1)

    results = []
    for seq in predictions:
        decoded = []
        prev = None
        for token in seq.tolist():
            if token != prev and token != tokenizer.char2idx[tokenizer.blank_token]:
                decoded.append(token)
            prev = token
        results.append(tokenizer.decode(decoded))

    return results
