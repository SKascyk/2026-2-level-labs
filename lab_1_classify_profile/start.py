"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
import lab_1_classify_profile.main


def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    de_profile = lab_1_classify_profile.main.create_language_profile('de', de_text, stopwords)
    en_profile = lab_1_classify_profile.main.create_language_profile('en', en_text, stopwords)
    unk_profile = lab_1_classify_profile.main.create_language_profile('unknown', unknown_text, stopwords)
    result = lab_1_classify_profile.main.detect_language_by_top_n(unk_profile, de_profile, en_profile, 15)
    assert result, "Detection result is None"
    print(result)


if __name__ == "__main__":
    main()
