import os
from symspellpy.symspellpy import SymSpell, Verbosity

sym_spell = None # Global variable

def init():
    global sym_spell
    # Maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = os.path.join(os.path.dirname(__file__),
                                   "frequency_dictionary_en_82_765.txt")
    term_index = 0  # Column of the term in the dictionary text file
    count_index = 1  # Column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

def checker(input_terms):
    global sym_spell
    # lookup suggestions for single-word input strings
    output_terms = {}
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_edit_distance_dictionary)
    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
    if not sym_spell:
        init()

    for input_term in input_terms:

        suggestions = sym_spell.lookup(
            input_term, suggestion_verbosity, max_edit_distance_lookup)

        if len(suggestions) == 1:
            for suggestion in suggestions:
                output_terms[input_term] = suggestion.term
        else:
            output_terms[input_term] = None

    return output_terms

if __name__ == '__main__':
    input_terms = {'direction', 'cdp_ax_tar','p_apbctrlstandstillvvehthres', 'cdp_request', 'accelerator'}
    results = checker(input_terms)
    print(results)