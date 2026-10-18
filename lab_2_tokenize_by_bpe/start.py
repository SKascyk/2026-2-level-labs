"""
BPE Tokenizer starter
"""

from lab_2_tokenize_by_bpe.main import (
    calculate_bleu,
    collect_frequencies,
    decode,
    encode,
    get_vocabulary,
    load_vocabulary,
    train
)

# pylint:disable=too-many-locals, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/secrets/secret_4.txt", "r", encoding="utf-8") as text_file:
        secret_text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8") as text_file:
        en_text_ref_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8") as text_file:
        en_translation_enc_raw = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/ru_raw.txt", "r", encoding="utf-8") as text_file:
        ru_text_ref_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/ru_encoded.txt", "r", encoding="utf-8") as text_file:
        ru_translation_enc_raw = text_file.read()

    word_frequencies = collect_frequencies(text, None, "</s>")
    new_word_frequencies = train(word_frequencies, 100)
    if new_word_frequencies is None:
        return None
    vocabulary = get_vocabulary(new_word_frequencies, "<unk>")
    if vocabulary is None:
        return None
    revealed_secret = decode(
        [int(num) for num in secret_text.split()],
        vocabulary,
        "</s>"
    )
    print(revealed_secret)

    vocab = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")
    encoded_ru_text = encode(ru_text_ref_translation, vocab, "\u2581", None, "<unk>")
    if encoded_ru_text is None:
        return None
    print(all(
        encoded_ru_text[i] == [int(num) for num in ru_translation_enc_raw.split()][i]
        for i in range(20)
    ))
    decoded_en_text = decode(
        [int(num) for num in en_translation_enc_raw.split()],
        vocab,
        None
    )
    if decoded_en_text is None:
        return None
    decoded_en_text = decoded_en_text.replace("\u2581", " ")
    result = calculate_bleu(decoded_en_text, en_text_ref_translation)
    print(result)
    assert result, "Translation not working"
    return None


if __name__ == "__main__":
    main()
