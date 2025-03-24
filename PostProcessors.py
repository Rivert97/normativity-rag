from typing import List

def remove_hyphens(text: str) -> str:
    """

    This fails for:
    * Natural dashes: well-known, self-replication, use-cases, non-semantic,
                      Post-processing, Window-wise, viewpoint-dependent
    * Trailing math operands: 2 - 4
    * Names: Lopez-Ferreras, VGG-19, CIFAR-100
    """
    # TODO: Esto lo puede hacer una expresion regular
    lines = [line.rstrip() for line in text.split("\n")]

    # Find dashes
    line_numbers = []
    for line_no, line in enumerate(lines[:-1]):
        if line.endswith("-"):
            line_numbers.append(line_no)

    # Replace
    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)


def dehyphenate(lines: List[str], line_no: int) -> List[str]:
    next_line = lines[line_no + 1]
    word_suffix = next_line.split(" ")[0]

    if lines[line_no].endswith(" -"):
        lines[line_no] = lines[line_no][:-2] + word_suffix
    else:
        lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = lines[line_no + 1][len(word_suffix) + 1:]
    return lines

def replace_ligatures(text: str) -> str:
    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        # "Ꜳ": "AA",
        # "Æ": "AE",
        "ꜳ": "aa",
    }
    for search, replace in ligatures.items():
        text = text.replace(search, replace)

    return text

def merge(text: str, body: str) -> str:
    body_characters = body.replace("\n", "").replace("\t", "").replace(" ", "").replace("-", "")

    merged = ""
    current_index = 0
    for c in text:
        if c in ("\n", "\t", " ", "-"):
            merged += c
        elif c == body_characters[current_index]:
            merged += c
            current_index += 1
        #else:
        #    print(c, end="")
    
    return merged
