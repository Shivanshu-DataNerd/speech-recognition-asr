import editdistance

def cer(pred, ref):
    return editdistance.eval(pred, ref) / max(len(ref), 1)

def wer(pred, ref):
    pred_words = pred.split()
    ref_words = ref.split()
    return editdistance.eval(pred_words, ref_words) / max(len(ref_words), 1)
