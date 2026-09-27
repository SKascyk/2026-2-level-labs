"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
from lab_1_classify_profile.main import tokenize, remove_stop_words, calculate_frequencies, get_top_n_words, create_language_profile, detect_language_by_top_n, detect_language_by_mse, save_profile, collect_profiles, detect_language_advanced, print_report

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
    de_tokens = tokenize(de_text)
    cleaned_de_tokens = remove_stop_words(de_tokens, stopwords) if de_tokens is not None else None
    de_freqs = calculate_frequencies(cleaned_de_tokens) if cleaned_de_tokens is not None else None
    result = get_top_n_words(de_freqs, 7) if de_freqs is not None else None
    assert result, "Detection result is None"
    print(result)
    de_profile = create_language_profile('de', de_text, stopwords)
    en_profile = create_language_profile('en', en_text, stopwords)
    unk_profile = create_language_profile('unknown', unknown_text, stopwords)
    result = (
        detect_language_by_top_n(unk_profile, de_profile, en_profile, 15)
        if not(
            unk_profile is None
            or de_profile is None
            or en_profile is None
        )
        else None
    )
    assert result, "Detection result is None"
    print(result)
    result = (
        detect_language_by_mse(unk_profile, de_profile, en_profile)
        if not(
            unk_profile is None
            or de_profile is None
            or en_profile is None
        )
        else None
    )
    assert result, "Detection result is None"
    print(result)
    profiles = [de_profile, en_profile, unk_profile]
    path = "lab_1_classify_profile/assets/profiles"
    paths_to_profiles = []
    for profile in profiles:
        if profile is not None:
            save_profile(profile, path)
            paths_to_profiles.append(path + f"/{profile[0]}.json")
    result = collect_profiles(paths_to_profiles)
    assert result, "Detection result is None"
    # print(result)
    paths_to_known_profiles = [path + "/de.json", path + "/en.json", path + "/la.json"]
    known_profiles = collect_profiles(paths_to_known_profiles)
    if not(unk_profile is None or known_profiles is None):
        result = detect_language_advanced(unk_profile, known_profiles, 15)
        print(result)
    assert result, "Detection result is None"
    if not(unk_profile is None or known_profiles is None):
        print_report(unk_profile, known_profiles, 15)


if __name__ == "__main__":
    main()
