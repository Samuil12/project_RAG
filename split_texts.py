from os import path, listdir
from rapidfuzz import fuzz
import re
import unicodedata
import json

# =============================================
# CONFIGURATION
# =============================================

# input directory, where we get the texts from
input_directory = 'extracted_texts'

# output file
output_dir = 'extracted_chunks'
output_file_name = 'chunks.jsonl'
output_path = path.join(output_dir, output_file_name)

# global variables that save text chunks
chunks = []
seen = set() # we don't want duplicate chunks

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
# reverse lookup for title ids
HEADING_TO_SECTION = {}

for section_id, headings in ALL_HEADINGS.items():
    for h in headings:
        FLAT_HEADINGS.append(h)
        HEADING_TO_SECTION[h] = section_id


# used for comparing beginning of lines to section titles
# cuts the line to
extra_chars = 10
SECTION_HEADINGS_LENGTHS = {
    "1": max(len(heading) for heading in section_1_headings) + extra_chars,
    "2": max(len(heading) for heading in section_2_headings) + extra_chars,
    "3": max(len(heading) for heading in section_3_headings) + extra_chars, # the longest title
    "4": max(len(heading) for heading in section_4_headings) + extra_chars,
    "5": max(len(heading) for heading in section_5_headings) + extra_chars,
    "6": max(len(heading) for heading in section_6_headings) + extra_chars
}

# ========================================
# METHODS FOR EXTRACTING CHUNKS
# ========================================

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
    '''Returns the section heading that the line resembles, uses a similarity score'''
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

def extract_medicine_name(line, heading):
    '''
    Extracts the name of the medicine 
    from the line that is the title of section 5
    '''

    # remove first words of line:
    line = line.split(sep=" ")

    if line[0] == "5." or line[0] == "5": # starts with "5. ", remove one more word
        line = line[1:]

    line = line[2:] # remove first 2 words

    # cases of different titles - 
    # "как да съхранявате" vs "съхранение на"
    if heading == "как да съхранявате":
        line = line[1:] # in this case remove 1 more word from the start

    medicine_name = " ".join(line)
    return medicine_name

def split_into_sections(lines):
    '''Splits the lines of texts by sections'''
    current_section = None
    buffer = []
    results = {str(section): "" for section in range(1, 7)}
    medicine_name = ""

    # track whether we've seen each section already
    section_seen = {str(section): 0 for section in range(1, 7)}  # ["1","2","3","4","5","6"]

    for line in lines:
        match = match_heading(line) # match is the title that the line resembles

        if match:
            section_id = HEADING_TO_SECTION[match]

            # mark occurrence
            section_seen[section_id] += 1

            # first occurence, heading 5 - we can extract the medicine name
            if section_id == "5" and section_seen[section_id] == 1:
                medicine_name = extract_medicine_name(line, match)

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

    return results, medicine_name


class Chunk:
    def __init__(self, medicine_name, section_id, text, chunk_id=1):
        '''Initialize a chunk object that stores metadata and text for the chunk'''
        medicine_name = normalize(medicine_name)

        self.unique_id = f"{medicine_name}::s{section_id}::c{chunk_id}"

        self.medicine_name = medicine_name
        
        # this is a string, not an int, if you need it as an int, 
        # uncomment the next line
        self.section_id = section_id 
        # self.section_id = int(section_id)

        self.text = text
        self.chars = len(text)

        # the chunk of the current section for the current medicine
        self.chunk_id = chunk_id

    # str and repr for debugging
    def __str__(self):
        res = f"medicine: {self.medicine_name}\n"
        res += f"section: {self.section_id}\n"
        res += f"chunk: {self.chunk_id}\n"
        res += f"chunk length: {self.chars}\n"
        res += f"text: {self.text[:100]}\n" + "...\n"
        res += f"{self.text[-100:]}\n"
        return res

    def __repr__(self):
        return str(self)
       

def split_into_chunks(sections_texts, medicine_name, chunk_size=1500, overlap=100):
    '''Splits text into chunks and saves them as a list of chunk objects'''
    
    # get every section
    for sec_id, sec_text in sections_texts.items():
        
        # no text caught in this section, move on to the next section
        chars_left = len(sec_text)
        if chars_left == 0:
            continue

        # split into chunks if the text is too long
        chunk_id = 1
        while chars_left > chunk_size:
            chunk_text = sec_text[:chunk_size]
            new_chunk = Chunk(medicine_name, sec_id, chunk_text, chunk_id)  
            if new_chunk.unique_id not in seen:
                chunks.append(new_chunk)
                seen.add(new_chunk.unique_id)    

            sec_text = sec_text[chunk_size-overlap:]
            chars_left -= chunk_size-overlap

            chunk_id += 1

        # flush the last chunk - the remaining text in sec_text
        new_chunk = Chunk(medicine_name, sec_id, sec_text, chunk_id)
        if new_chunk.unique_id not in seen:
            chunks.append(new_chunk)
            seen.add(new_chunk.unique_id)

    return chunks


def extract_file(file_path):
    '''Extracts text chunks from a single file'''
    
    with open(file_path, 'r', encoding="utf-8") as fd:
        lines = fd.readlines()
        lines = [line[:-1] for line in lines]

    sections_texts, medicine_name = split_into_sections(lines)

    split_into_chunks(sections_texts, medicine_name)


def extract_all_files(limit: None|int = 5):
    '''Extract text chunks from all chunks in the chosen directory'''
    
    count = 0
    for file in listdir(input_directory): # os.listdir
        
        # stop after "limit" amount of files extracted
        if limit and count >= limit:
            break

        if file.endswith(".txt"):
            file_path = path.join(input_directory, file)
            extract_file(file_path)
            count += 1

            print(f"Finished with file {file}")


def save_chunks_to_jsonl(overwrite=True):
    '''Saves chunk objects in json lines to the output file'''

    open_mode = "w" if overwrite else "a"

    with open(output_path, open_mode, encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "medicine_name": chunk.medicine_name,
                "section_id": chunk.section_id,
                "chunk_id": chunk.chunk_id,
                "chars": chunk.chars,
                "text": chunk.text
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================================
# EXECUTE MAIN METHOD
# =========================================
if __name__ == "__main__": 

    # set limit to None to extract all files, default is limit = 5
    # extract_all_files(limit=None)
    extract_all_files()  

    # set parameter overwrite=False to append chunks to end of file
    # save_chunks_to_jsonl(overwrite=False)
    save_chunks_to_jsonl() 

    print("\nDone!")
    
    # print chunks for debugging
    # for chunk in chunks:
    #   print(chunk)