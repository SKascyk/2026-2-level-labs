"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
import lab_1_classify_profile.main as cp


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
    de_tokens = cp.tokenize(de_text)
    cleaned_de_tokens = (
        cp.remove_stop_words(de_tokens, stopwords)
        if de_tokens is not None
        else None
    )
    de_freqs = (
        cp.calculate_frequencies(cleaned_de_tokens)
        if cleaned_de_tokens is not None
        else None
    )
    if de_freqs is not None:
        print(f"{cp.get_top_n_words(de_freqs, 7)}")
    de_profile = cp.create_language_profile("de", de_text, stopwords)
    en_profile = cp.create_language_profile("en", en_text, stopwords)
    unk_profile = cp.create_language_profile("unknown", unknown_text, stopwords)
    if not(
        unk_profile is None
        or de_profile is None
        or en_profile is None
    ):
        print(
            f"{cp.detect_language_by_top_n(unk_profile, de_profile, en_profile, 15)}"
        )
        print(
            f"{cp.detect_language_by_mse(unk_profile, de_profile, en_profile)}"
        )
    profiles = [de_profile, en_profile, unk_profile]
    paths_to_profiles = []
    for _ in profiles:
        if _ is not None:
            cp.save_profile(_, "lab_1_classify_profile/assets/profiles")
            paths_to_profiles.append(f"lab_1_classify_profile/assets/profiles/{_[0]}.json")
    if paths_to_profiles is not None:
        print(f"{cp.collect_profiles(paths_to_profiles)}")
    paths_to_profiles.remove("lab_1_classify_profile/assets/profiles/unknown.json")
    paths_to_profiles.append("lab_1_classify_profile/assets/profiles/la.json")
    prof_collection = cp.collect_profiles(paths_to_profiles)
    result = None
    if not(unk_profile is None or prof_collection is None):
        result = cp.detect_language_advanced(unk_profile, prof_collection, 15)
        print(result)
    if not(unk_profile is None or result is None):
        cp.print_report(unk_profile, result, 15)


if __name__ == "__main__":
    main()
