import re

def read_header(csv_file):
    """
    Read the header of a CSV file.
    """
    with open(csv_file) as f:
        h = f.readline().strip()
    h = [item.strip() for item in h.split(',')]
    return h

def match_in_pattern_list(pattern_list, text):
    """
    Match `text` with all patterns in `pattern_list`.
    """
    for pattern in pattern_list:
        if re.match('^' + pattern + '$', text) is not None:
            return True
    return False


