from typing import Literal
from nltk.corpus import words, wordnet as wn

RAW_WORD_LIST = {word.lower() for word in words.words()}
DEFINED_WORDS = set(wn.words())
CLEAN_WORD_LIST = RAW_WORD_LIST.intersection(DEFINED_WORDS)


def find_words_with_matching_ends(n: int, require_gap: bool = False) -> list[str]:
    """Find words where the first and last `n` characters are identical.

    Args:
        n: The number of characters to compare at the beginning and end.
        require_gap: If True, restricts results to words with at least one
            character between the matching ends (minimum length 2n + 1).
            If False, includes words of exactly length 2n.

    Returns:
        A alphabetically sorted list of matching words.
    """
    min_len = 2 * n + (1 if require_gap else 0)
    return sorted(
        word
        for word in CLEAN_WORD_LIST
        if len(word) >= min_len and word[:n] == word[-n:]
    )


def find_qwerty_words(
    n: int, group: Literal["left", "right", "top", "middle"]
) -> list[str]:
    """Find words of length `n` built exclusively from a specific QWERTY keyboard section.

    Args:
        n: The exact length of the desired words.
        group: The section of the keyboard to use ('left', 'right', 'top', or 'middle').

    Returns:
        An alphabetically sorted list of matching words.

    Raises:
        ValueError: If `group` is not one of the supported keyboard sections.
    """
    key_groups = {
        "left": set("abcdefgqrstvwxz"),
        "right": set("hijklmnopuy"),
        "top": set("qwertyuiop"),
        "middle": set("asdfghjkl"),
    }

    try:
        allowed_chars = key_groups[group.lower()]
    except KeyError:
        raise ValueError(
            "The 'group' argument must be one of: 'left', 'right', 'top', or 'middle'."
        ) from None

    return sorted(
        word
        for word in CLEAN_WORD_LIST
        if len(word) == n and set(word).issubset(allowed_chars)
    )


def find_shifted_word_pairs(n: int) -> list[tuple[str, str, str]]:
    """Find pairs of words where one is formed by shifting the other by `n` characters.

    Args:
        n: The shift amount.

    Returns:
        An alphabetically sorted list of tuples in the format:
        (word1, word2, shift_direction).

    Raises:
        ValueError: If `n` is not between 1 and 13.
    """
    if not (1 <= n <= 13):
        raise ValueError("The shift amount 'n' must be between 1 and 13.")

    pairs = set()
    results = []

    for word in CLEAN_WORD_LIST:
        if len(word) >= 3:
            shifted_word = "".join(
                chr(((ord(char) - 97 + n) % 26) + 97) for char in word
            )

            if shifted_word in CLEAN_WORD_LIST:
                w1, w2 = min(word, shifted_word), max(word, shifted_word)

                if (w1, w2) not in pairs:
                    pairs.add((w1, w2))

                    test_shift = "".join(
                        chr(((ord(char) - 97 + n) % 26) + 97) for char in w1
                    )
                    shift_direction = f"+{n}" if test_shift == w2 else f"-{n}"

                    results.append((w1, w2, shift_direction))

    return sorted(results)
