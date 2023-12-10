from .imports import *

def remove_trailing_punctuation(word):
    # remove trailing punctuation and spaces
    # don't remove the dash '-', as this might interfere with the function to repair broken words!
    # Question: should we also remove punct at the beginning of the token...? Not doing that now.
    return re.sub(r'[\.,\?!"\')(:;`]+\s*$', '', word)

# # small test
# word = "...example.,...! "
# clean_word = remove_trailing_punctuation(word)
# print(clean_word)






# process a list of word pairs, where each pair consists of an 'incorrect' word with a historic long 's' (ſ) and its 'correct' modern equivalent
# the script then replaces the historic long 's' (ſ) words with 'f', generates new word pairs
# ONLY if the newly generated f-word does NOT exist in the English language, we retain the word!! For this, we use language stats provided by wordfreq
# the resulting pairs are then written to the outfile, while pairs that exists -- with high frequency in English -- are written to a separate disregard_file
# i think this is clever, so i named the function accordingly :-)

def generate_clever_f_s_hack(source_file, output_file, disregard_file, skip_words=None, frequency_threshold=1e-6):
    if skip_words is None:
        skip_words = {'ſlip'}  # add specific words to skip here -- dunno if this is still useful, the file will capture most of these words

    unique_pairs = set()  # set to keep track of unique (incorrect f-word, correct s-word) pairs

    with open(source_file, 'r') as infile, open(output_file, 'w') as outfile, open(disregard_file, 'w') as disregard:
        # skip the title line of the infile
        next(infile)

        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            incorrect, correct = parts[:2]
            # e.g.:
            # incorrect correct
            # moſt 	    most
            # muſt 	    must
            # ſo 	      so
            # ſome 	    some
            # ſee       see   etc.

            # strip leading/trailing spaces
            incorrect = incorrect.strip()
            correct = correct.strip()

            # remove trailing punctuation
            incorrect = remove_trailing_punctuation(incorrect)
            correct = remove_trailing_punctuation(correct)

            # replace 'ſ' with 'f' in the incorrect word
            f_incorrect = incorrect.replace('ſ', 'f')
            # e.g.:
            # incorrect correct
            # moft 	    most
            # muft 	    must
            # fo 	      so
            # fome 	    some
            # fee       see   etc.

            # skip if the incorrect word is in skip_words or already in pairs
            if f_incorrect in skip_words or (f_incorrect, correct) in unique_pairs:
                continue

            # check the frequency of the word
            word_frequency = wordfreq.word_frequency(f_incorrect.lower(), 'en')

            # skip if the word exists and its frequency is above the threshold
            if word_frequency > frequency_threshold:
                disregard.write(f"{f_incorrect}\t{correct}\n")
                #print(f'Word that exist with the f-spelling and we don\'t want to include: {f_incorrect}')
                # e.g.
                # Words that exist with the f-spelling and we don't want to include: fame
                # Words that exist with the f-spelling and we don't want to include: found    etc.
                continue

            # check if the generated word exists in English
            if word_frequency <= frequency_threshold:
                outfile.write(f"{f_incorrect}\t{correct}\n")
                unique_pairs.add((f_incorrect, correct))
                # e.g.
                # moft 	    most
                # muft 	    must
                # fo 	      so
                # fome 	    some    etc.

# apply
# generate_clever_f_s_hack(
#     source_file=os.path.join(PATH_OCR_RULESETS, "all_long_s_corrections_log.txt"),
#     output_file=os.path.join(PATH_OCR_RULESETS, "clever_f_ſ_hack.txt"),
#     disregard_file=os.path.join(PATH_OCR_RULESETS, "disregard_fſs_replacements.txt")
# )





@cache
def load_correction_rules(file_path = os.path.join(PATH_OCR_RULESETS, 'CorrectionRules.txt')):
    correction_rules = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                incorrect, correct = parts[:2]
                correction_rules[incorrect] = correct
    return correction_rules


def correct_ocr_errors(text, correction_rules):
    corrections = 0
    for incorrect, correct in correction_rules.items():
        if incorrect in text:
            text = text.replace(incorrect, correct)
            corrections += 1
    return text, corrections

def rejoin_linebreaks(text, specific_linebreak_corrections):
    """
    function to addresses the issue of words that are split between two lines due to a line break, typically indicated by a hyphen
    the function rejoins such words
    """
    corrections = 0
    parts = text.split('-\n')
    corrected_text = parts[0]
    for part in parts[1:]:
        corrected_text_words = corrected_text.split()
        part_words = part.split()

        if corrected_text_words and part_words:  # check if both lists are not empty
            last_word_before_break = corrected_text_words[-1]
            first_word_after_break = part_words[0]

            # form the broken word and the corrected word
            broken_word = last_word_before_break + '-\n' + first_word_after_break
            corrected_word = last_word_before_break + first_word_after_break

            # log the correction (gets later written to the txt file)
            # specific_linebreak_corrections[broken_word + " \t " + corrected_word] += 1
            specific_linebreak_corrections.append((broken_word,corrected_word))

            corrected_text += part
            corrections += 1
        else:
            # if either part is empty or doesn't contain words, simply append a hyphen
            corrected_text += '-' + part

    return corrected_text, corrections

def replace_historic_long_s(text, long_s_corrections):
    """
    function to replaces the historic long 's' (ſ) with the regular 's'

    :text: text to be processed
    :long_s_corrections: dictionary to log specific corrections and their counts
    :return: tuple of processed text with long 's' replaced, and the number of corrections made
    """
    corrected_text = text.replace('ſ', 's')
    corrections = 0
    if corrected_text != text:
        words_with_long_s = set(text.split()) - set(corrected_text.split())
        for word in words_with_long_s:
            corrected_word = word.replace('ſ', 's')
            long_s_corrections.append((word,corrected_word))
            corrections += 1
    return corrected_text, corrections

@cache
def load_f_s_hack_corrections(file_path = os.path.join(PATH_OCR_RULESETS, "clever_f_ſ_hack.txt")):
    """
    little helper script to load the f-->s words (from generate_clever_f_s_hack) into a dict, for convenient lookup
    """
    correction_rules = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                incorrect, correct = parts[:2]
                correction_rules[incorrect] = correct
    return correction_rules

def identify_headers(pages, remove_headers=True, similarity_threshold=80):
    """
    function to identifies and optionally removes running headers
    inspired by Ted Underwood's GREAT headerfinder script: https://github.com/tedunderwood/DataMunging/blob/master/runningheaders/HeaderFinder.py
    some changes made:
      - flexibility to remove headers or just identify them (just by setting the boolean value)
      - we don't explicitly handle roman numerals, the line comparison logic (combining str.isalpha and a threshold for fuzzy matching) should take care of it

    :pages: list of dicts, each representing a page with 'page_text'
    :remove_headers: bool, if set to True --> removes identified headers, otherwise just identifies them and wirtes them to the log
    :similarity_threshold: int, threshold for fuzzy matching to consider lines as similar (default 80 seems to work well)
    :return: list of pages with headers
    """
    identified_headers = []
    headers_set = set()

    def get_substantial_lines(page_text):
        """
        helper function: if the processed line contains less than 5 characters, or if the line consists solely of digits
        it is considered insubstantial and is skipped
        """
        lines = page_text.split('\n')
        substantial_lines = []
        for line in lines:
            if len(line.strip()) < 5 or line.strip().isdigit():
                continue
            substantial_lines.append(line)
            if len(substantial_lines) == 2:
                break
        return substantial_lines

    for i,page in enumerate(pages):
        current_page_text = page.get('page_text','')
        current_substantial_lines = get_substantial_lines(current_page_text)

        # determine the range of pages to compare with
        start_index = max(0, i - 2)
        end_index = min(len(pages), i + 3)
        if i == len(pages) - 1:  # Special handling for the last page
            start_index = max(0, i - 2)  # Compare with pages before

        for j in range(start_index, end_index):
            if i == j:
                continue

            comparison_page_text = pages[j].get('page_text','')
            comparison_substantial_lines = get_substantial_lines(comparison_page_text)

            for current_line in current_substantial_lines:
                for comparison_line in comparison_substantial_lines:
                    # line comparison logic, considering possible page numbers
                    cleaned_current_line = ''.join(filter(str.isalpha, current_line))
                    cleaned_comparison_line = ''.join(filter(str.isalpha, comparison_line))

                    s = SequenceMatcher(None, cleaned_current_line, cleaned_comparison_line)
                    similarity = s.ratio() * 100

                    if similarity > similarity_threshold:
                        header_key = (i, current_line)
                        if header_key not in headers_set:
                            identified_headers.append(header_key)
                            headers_set.add(header_key)
                        break
    return {hdr for lnnum,hdr in headers_set}



def cleanup_str(txt, use_nltk_tokenizer=False, remove_headers:list=None, **page_attrs):
    """
    Most of the cleanup occurs here. Can be called with a string or a string with page attributes
    """
    orig_txt = txt
    page_text = txt



    # dicts to store specific corrections and their counts
    specific_ocr_corrections = []
    specific_linebreak_corrections = []
    specific_long_s_corrections = []
    correction_rules = load_correction_rules()
    clever_f_s_hack_rules = load_f_s_hack_corrections()

    # add a dictionary for specific f ſ hack corrections
    specific_f_s_hack_corrections = []

    specific_header_corrections = []
        

    # counters for corrections
    linebreak_corrections = 0
    ocr_corrections = 0
    long_s_corrections = 0
    f_s_word_replacements = 0

    # remove headers?
    if remove_headers:
        hdrs=set(remove_headers)
        lines = page_text.split('\n')
        new_lines = [ln for ln in lines if ln.strip() not in hdrs]
        hdr_lines = {ln for ln in lines if ln.strip() in hdrs}
        specific_header_corrections.extend([(hdr,'') for hdr in hdr_lines])
        page_text = '\n'.join(new_lines)
    

    # rejoin line breaks before tokenization and log corrections
    page_text, corrections = rejoin_linebreaks(page_text, specific_linebreak_corrections)
    linebreak_corrections += corrections

    # apply correction for long 's'
    corrected_text, corrections = replace_historic_long_s(page_text, specific_long_s_corrections)
    long_s_corrections += corrections
    page_text = corrected_text

    # tokenization
    tokens = tokenize_agnostic(page_text)

    # apply OCR corrections on tokens and log corrections
    corrected_tokens = []
    for token in tokens:
        if token in correction_rules:
            corrected_token = correction_rules[token]
            ocr_corrections += 1
            specific_ocr_corrections.append((token,corrected_token))
        else:
            corrected_token = token
        corrected_tokens.append(corrected_token)

    # apply f-ſ-s hack corrections on tokens and log corrections
    for i, token in enumerate(corrected_tokens):
        if token in clever_f_s_hack_rules:
            corrected_token = clever_f_s_hack_rules[token]
            f_s_word_replacements += 1
            specific_f_s_hack_corrections.append((token,corrected_token))
            corrected_tokens[i] = corrected_token

    # convert corrected tokens back to text for further processing
    corrected_text = untokenize_agnostic(corrected_tokens)
    corrected_tokens_l = [x.strip().lower() for x in corrected_tokens if x.strip() and x.strip()[0].isalpha()]

    return {
        **{k:v for k,v in page_attrs.items() if k!='page_text'}, 
        'page_text':corrected_text, 
        'page_text_orig':page_text, 
        'page_tokens':corrected_tokens_l,
        'page_corrections_headers':list(set(specific_header_corrections)),
        'page_corrections_linebreaks':list(set(specific_linebreak_corrections)),
        'page_corrections_long_s':list(set(specific_long_s_corrections)),
        'page_corrections_ocr':list(set(specific_ocr_corrections)),
        'page_corrections_f_s':list(set(specific_f_s_hack_corrections)),
    }



def cleanup_page(page_d, remove_headers=None):
    """
    Cleanup a page dictionary
    """
    txt=page_d.get('page_text','')
    odx=cleanup_str(txt, remove_headers=remove_headers, **page_d)
    return odx

def cleanup_pages(pages_ld,remove_headers=True):
    """
    Cleanup a list of pages
    """
    headers = identify_headers(pages_ld, remove_headers=remove_headers)
    pages_ld = [cleanup_page(page_d,remove_headers=headers) for page_d in pages_ld]
    return pages_ld
