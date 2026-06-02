from os import path
from rapidfuzz import fuzz
import re
import unicodedata

# COMMENT: tested splitting text into sections with a few files, 
# most of the time finds the sections (at least on the tested files) 

# TODO: I still need to split long sections into smaller chunks
# and index them properly


# testing with one file
texts_directory = 'extracted_texts'
example_file_name = 'b001.txt'


# =========================================
# Preparing section titles to compare lines to
# =========================================

section_1_headings = ["какво представлява"]

section_2_headings = ["какво трябва да знаете преди да",
                      "преди да приемете"]

section_3_headings = ["как да прилагате",
                     "как се прилага",
                     "как да приемате",
                     "как да използвате",
                     "как ще ви бъде прилаган"
                    ]

section_4_headings = ["възможни нежелани реакции"]
section_5_headings = ["как да съхранявате",
                      "съхранение на"]

# text in section 6 might not be needed
section_6_headings = ["съдържание на опаковката и допълнителна информация"]

# ids for headings
ALL_HEADINGS = {
    "1": section_1_headings,
    "2": section_2_headings,
    "3": section_3_headings,
    "4": section_4_headings,
    "5": section_5_headings,
    "6": section_6_headings,
}

# get all headers in a single list
FLAT_HEADINGS = []
# reverse lookup
HEADING_TO_SECTION = {}

for section_id, headings in ALL_HEADINGS.items():
    for h in headings:
        FLAT_HEADINGS.append(h)
        HEADING_TO_SECTION[h] = section_id


# used for comparing beginning of lines to section titles
SECTION_HEADINGS_LENGTHS = {
    "1": len(section_1_headings[0]) + 10,
    "2": len(section_2_headings[0]) + 10,
    "3": len(section_3_headings[4]) + 10, # the longest title
    "4": len(section_4_headings[0]) + 10,
    "5": len(section_5_headings[0]) + 10,
    "6": len(section_6_headings[0]) + 10
}

def normalize(text):
    """
    Returns normalized text by trimming unnecessary spaces, 
    converting to lower case and removing punctuation.
    """
    # replace latin letters with similar cyrillic letters (capital)
    text = text.replace('H', 'Н')
    text = text.replace('B', 'В')

    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # replace latin letters with similar cyrillic letters
    text = text.replace('k', 'к')
    text = text.replace('a', 'а')
    text = text.replace('p', 'р')
    text = text.replace('c', 'с')
    text = text.replace('e', 'е')
    text = text.replace('x', 'х')

    return text

def match_heading(line, threshold=90):
    '''Returns the section heading that the line resembles'''
    line = normalize(line)

    best_score = 0
    best = None

    # check every section title, get the best match
    for h in FLAT_HEADINGS:
        section_id = HEADING_TO_SECTION[h]
        line_length = SECTION_HEADINGS_LENGTHS[section_id]
        line_cut = line[:line_length]
        score = fuzz.token_set_ratio(line_cut, h)

        # if a line starts with the number means it's probably what we need
        if line.startswith(str(section_id)): 
            score += 35

        if score > best_score:
            best_score = score
            best = h

    # if we are certain a title matches, return the id 
    if best_score >= threshold:
        return best

    # otherwise return nothing
    return None

def split_into_sections(lines):
    '''Splits the lines of texts by sections'''
    current_section = None
    buffer = []
    results = {str(section): "" for section in range(1, 7)}

    # track whether we've seen each section already
    section_seen = {str(section): 0 for section in range(1, 7)}  # e.g. ["1","2","3","4","5","6"]

    for line in lines:
        match = match_heading(line)

        if match:
            section_id = HEADING_TO_SECTION[match]

            # mark occurrence
            section_seen[section_id] += 1

            # ignore first occurrence
            if section_seen[section_id] == 1:
                continue

            # valid start (2nd occurrence)
            if current_section is not None:
                results[current_section] += "\n".join(buffer)

            current_section = section_id
            buffer = []
            continue

        # normal text
        if current_section is not None:
            buffer.append(line)

    # flush last section
    if current_section is not None:
        results[current_section] += "\n".join(buffer)

    return results


# =========================================
# execute main method
# =========================================
if __name__ == "__main__": 

    # trying with one file
    
    file_name = path.join(texts_directory, example_file_name)
    with open(file_name, 'r', encoding="utf-8") as fd:
        lines = fd.readlines()
        lines = [line[:-1] for line in lines] # remove the new line character

    sections_texts = split_into_sections(lines)

    # print the start of sections (for debugging)
    for sec_id, sec_text in sections_texts.items():
        print(f"section {sec_id}, length is {len(sec_text)}") 
        print(sec_text[:100])

    str1 = "5, КАК ДА СЪХРАНЯВАТЕ БУПРЕНОФИН АКТАВИС"
    print(normalize(str1))
    print(match_heading(str1))

    # for (i, line) in enumerate(lines, start=1):
    #     print(f"{line}", end="")