import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path, get_weights_file_path
from model import build_transformer
from dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Autoregressively generate target tokens using greedy search.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute encoder output once
    encoder_output = model.encode(source, source_mask)

    # Initialize decoder input with [SOS] token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Decoder forward pass
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project output to target vocabulary
        prob = model.project(out[:, -1])

        # Pick the token with highest probability
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def load_translator(config=None, checkpoint_path=None, device=None):
    """
    Loads model, tokenizers, and weights ready for translation.
    """
    if config is None:
        config = get_config()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt']))

    if not tokenizer_src_path.exists() or not tokenizer_tgt_path.exists():
        raise FileNotFoundError(
            f"Tokenizer files not found ({tokenizer_src_path} or {tokenizer_tgt_path}). "
            "Please train the model or tokenizers first using `python train.py`."
        )

    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    if checkpoint_path is None:
        checkpoint_path = latest_weights_file_path(config)

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Model weights not found at '{checkpoint_path}'. "
            "Please train the model first or provide a valid checkpoint with --checkpoint."
        )

    print(f"Loading weights from: {checkpoint_path} (on {device})")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    return model, tokenizer_src, tokenizer_tgt, config, device


def translate(sentence: str, model=None, tokenizer_src=None, tokenizer_tgt=None, config=None, device=None):
    """
    Translate a source language sentence into the target language.
    """
    if model is None or tokenizer_src is None or tokenizer_tgt is None:
        model, tokenizer_src, tokenizer_tgt, config, device = load_translator(config=config, device=device)

    if config is None:
        config = get_config()
    if device is None:
        device = next(model.parameters()).device

    seq_len = config['seq_len']
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    # Encode source sentence
    source_tokens = tokenizer_src.encode(sentence).ids
    num_padding_tokens = seq_len - len(source_tokens) - 2

    if num_padding_tokens < 0:
        print(f"Warning: Sentence exceeds max seq_len ({seq_len}), truncating...")
        source_tokens = source_tokens[:seq_len - 2]
        num_padding_tokens = 0

    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(source_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * num_padding_tokens, dtype=torch.int64)
        ]
    ).unsqueeze(0).to(device)  # (1, seq_len)

    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)  # (1, 1, 1, seq_len)

    with torch.no_grad():
        out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, seq_len, device)

    return tokenizer_tgt.decode(out.detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description="Translate sentences using trained Transformer")
    parser.add_argument("sentence", type=str, nargs="?", default=None, help="English sentence to translate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--epoch", type=str, default=None, help="Epoch number to load from weights folder")
    args = parser.parse_args()

    config = get_config()
    checkpoint = args.checkpoint
    if checkpoint is None and args.epoch is not None:
        checkpoint = get_weights_file_path(config, args.epoch)

    try:
        model, tokenizer_src, tokenizer_tgt, config, device = load_translator(
            config=config, checkpoint_path=checkpoint
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.sentence:
        translation = translate(args.sentence, model, tokenizer_src, tokenizer_tgt, config, device)
        print(f"\n[Source ({config['lang_src']})]: {args.sentence}")
        print(f"[Target ({config['lang_tgt']})]: {translation}\n")
    else:
        print(f"\n--- Transformer Translation CLI ({config['lang_src']} -> {config['lang_tgt']}) ---")
        print("Type 'q' or 'exit' to quit.\n")
        while True:
            try:
                text = input(f"Enter {config['lang_src']} text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text or text.lower() in ('q', 'exit'):
                break
            translation = translate(text, model, tokenizer_src, tokenizer_tgt, config, device)
            print(f"Translation ({config['lang_tgt']}): {translation}\n")


if __name__ == "__main__":
    main()
