import torch

def nopeak_mask(size):
    np_mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
    return np_mask

#def create_masks(trg, trg_pad, src=None, src_pad=None):
    if src is not None:
        src_mask = (src != src_pad).unsqueeze(-2)
    else:
        src_mask = None

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size).to(trg.device)
        if trg.size(1) > 1: 
            np_mask = np_mask.repeat(trg.size(0), 1, 1)  
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return trg_mask, src_mask

def create_masks(sequences, pad_token):
    batch_size, seq_len = sequences.shape
    pad_mask = (sequences != pad_token).unsqueeze(1)

    look_ahead_mask = torch.triu(torch.ones((seq_len, seq_len), device=sequences.device), diagonal=1).bool()
    look_ahead_mask = look_ahead_mask.unsqueeze(0).repeat(batch_size, 1, 1)  

    combined_mask = pad_mask & (~look_ahead_mask)

    return combined_mask