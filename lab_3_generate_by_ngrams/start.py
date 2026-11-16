"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
    WordProcessor,
)

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
    WordProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    print('Mark 4 task')

    text_processor = TextProcessor('_')
    encoded_text = text_processor.encode(text)
    if encoded_text is None:
        return

    decoded_text = text_processor.decode(encoded_text)
    if decoded_text is None:
        return

    print(
        f'Encoded text:\n{encoded_text[-100:]}\n\n'
        f'Decoded text:\n{decoded_text[-100:]}'
    )

    print('\nNGram LM check task')

    model = NGramLanguageModel(encoded_text, 7)
    print('Model built:', model.build())

    print('\nMark 6 task')

    greedy_generator = GreedyTextGenerator(model, text_processor)
    generated_text = greedy_generator.run('Vernon', 51)
    print('Generated text:')
    print(generated_text)

    print('\nMark 8 task')
    generator = BeamSearchTextGenerator(
        language_model=model,
        text_processor=text_processor,
        beam_width=3,
    )
    generated = generator.run('Vernon', 56)
    print('Generated text:')
    print(generated)

    print('\nMark 10 task 1')
    reader = NGramLanguageModelReader(
        "./lab_3_generate_by_ngrams/assets/en_own.json", '_'
    )
    models = []
    for ngram_size in (2, 3, 4):
        loaded_model = reader.load(ngram_size)
        if loaded_model is not None:
            models.append(loaded_model)

    backoff_generator = BackOffGenerator(
        tuple(models),
        reader.get_text_processor(),
    )
    backoff_generated = backoff_generator.run('Vernon', 100)
    print(backoff_generated)

    print('\nMark 10 task 2')
    prompt = 'Harry looked at'

    word_processor = WordProcessor('<eos>')
    encoded_text_2 = word_processor.encode(text)
    if encoded_text_2 is None:
        return

    word_models = {}
    for ngram_size in (2, 3, 4):
        word_model = NGramLanguageModel(encoded_text_2, ngram_size)
        word_model.build()
        word_models[ngram_size] = word_model

    print(
        '\nGreedy algorithm continue:\n'
        f'{GreedyTextGenerator(word_models[4], word_processor).run(prompt, 12)}'
    )

    beam_continue = BeamSearchTextGenerator(
        language_model=word_models[4],
        text_processor=word_processor,
        beam_width=3,
    )
    print(f'\nBeam Search algorithm continue:\n{beam_continue.run(prompt, 12)}')

    backoff_continue = BackOffGenerator(
        tuple(word_models.values()),
        word_processor,
    )
    print(f'\nBackOff algorithm continue:\n{backoff_continue.run(prompt, 12)}')

    print(
        '\nComparison to greedy generated text at the level of letters:\n'
        f'{greedy_generator.run(prompt, 60)}'
    )

    result = backoff_generated
    assert result

    text_processor = TextProcessor("_")
    encoded_text = text_processor.encode(text)
    if encoded_text is None:
        return None
    decoded_text = text_processor.decode(encoded_text)
    assert decoded_text
    print(decoded_text)

    language_model = NGramLanguageModel(encoded_text, 7)
    language_model.build()

    greedy_generator = GreedyTextGenerator(language_model, text_processor)
    generated_text = greedy_generator.run("Vernon", 51)
    assert generated_text
    print(generated_text)

    beam_generator = BeamSearchTextGenerator(language_model, text_processor, 3)
    generated_text = beam_generator.run("Vernon", 56)
    assert generated_text
    print(generated_text)

    reader = NGramLanguageModelReader("./lab_3_generate_by_ngrams/assets/en_own.json", "_")
    sym_models = []
    for n_gram_size in range(2, 5):
        sym_model = reader.load(n_gram_size)
        if sym_model is None:
            return None
        sym_models.append(sym_model)
    backoff = BackOffGenerator(tuple(sym_models), reader.get_text_processor())
    generated_text = backoff.run("Vernon", 100)
    result = generated_text
    assert result
    print(result)

    prompt = "Harry looked at"
    word_processor = WordProcessor("<eos>")
    word_encoded_text = word_processor.encode(text)
    word_models = {}
    for n_gram_size in range(2, 5):
        word_model = NGramLanguageModel(word_encoded_text, n_gram_size)
        word_model.build()
        word_models[n_gram_size] = word_model
    greedy_generator = GreedyTextGenerator(word_models[4], word_processor)
    beam_generator = BeamSearchTextGenerator(word_models[4], word_processor, 3)
    backoff = BackOffGenerator(tuple(word_models.values()), word_processor)
    for generator in (greedy_generator, beam_generator, backoff):
        print(generator.run(prompt, 12))
    print(GreedyTextGenerator(language_model, text_processor).run(prompt, 60))
    return None

if __name__ == "__main__":
    main()
