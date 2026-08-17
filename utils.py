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


def find_leetspeak_words(n: int) -> list[str]:
    """Find words of length `n` built exclusively of letters that resemble digits.

    The qualifying letters are: O, I, Z, E, A, S, G, L, and B, which are
    commonly used as stand-ins for 0, 1, 2, 3, 4, 5, 6, 7, and 8 respectively.

    Args:
        n: The exact length of the desired words.

    Returns:
        An alphabetically sorted list of matching words.
    """
    allowed_chars = set("oizeasglb")

    return sorted(
        word
        for word in CLEAN_WORD_LIST
        if len(word) == n and set(word).issubset(allowed_chars)
    )


def find_strobogrammatic_words(n: int) -> list[str]:
    """Find words of length `n` that look the same when rotated 180 degrees.

    This function checks for words built from letters that have 180-degree
    rotational symmetries (e.g., 'o', 's', 'x') or rotational pairs (e.g.,
    'm' and 'w', 'b' and 'q').

    Args:
        n: The exact length of the desired words.

    Returns:
        An alphabetically sorted list of matching words.
    """
    rotation_map = {
        "o": "o",
        "i": "i",
        "x": "x",
        "s": "s",
        "z": "z",
        "l": "l",
        "b": "q",
        "q": "b",
        "d": "p",
        "p": "d",
        "m": "w",
        "w": "m",
        "n": "u",
        "u": "n",
    }

    allowed_chars = set(rotation_map.keys())
    results = []

    for word in CLEAN_WORD_LIST:
        if len(word) == n and set(word).issubset(allowed_chars):
            rotated_word = "".join(rotation_map[char] for char in reversed(word))

            if rotated_word == word:
                results.append(word)

    return sorted(results)


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


def find_anadromes(n: int) -> list[str]:
    """Find words of length `n` that form a different valid word when reversed.

    An anadrome (or emordnilap) is a word that makes sense both forward and
    backward, but forms a different word in each direction (e.g., 'desserts'
    and 'stressed'). Palindromes are explicitly excluded.

    Args:
        n: The exact length of the desired words.

    Returns:
        An alphabetically sorted list of matching words.
    """
    return sorted(
        word
        for word in CLEAN_WORD_LIST
        if len(word) == n and word < word[::-1] and word[::-1] in CLEAN_WORD_LIST
    )


def find_isograms(n: int) -> list[str]:
    """Find words of length `n` where no letter is repeated (isograms).

    Args:
        n: The exact length of the desired words.

    Returns:
        An alphabetically sorted list of matching words.
    """
    return sorted(
        word for word in CLEAN_WORD_LIST if len(word) == n and len(set(word)) == n
    )


def find_wrong_article_words(n: int) -> list[str]:
    """Find words of length `n` that form a valid word with the wrong indefinite article.

    Matches words that start with 'a' followed by a valid word starting
    with a vowel (e.g., 'aisle' -> 'a' + 'isle'), or start with 'an' followed
    by a valid word starting with a consonant (e.g., 'antic' -> 'an' + 'tic').

    Args:
        n: The exact length of the desired words.

    Returns:
        An alphabetically sorted list of matching words.
    """
    vowels = set("aeiou")
    results = []

    for word in CLEAN_WORD_LIST:
        if len(word) != n:
            continue

        if word.startswith("a"):
            rem_a = word[1:]
            if rem_a and rem_a[0] in vowels and rem_a in CLEAN_WORD_LIST:
                results.append(word)
                continue

        if word.startswith("an"):
            rem_an = word[2:]
            if rem_an and rem_an[0] not in vowels and rem_an in CLEAN_WORD_LIST:
                results.append(word)

    return sorted(results)
