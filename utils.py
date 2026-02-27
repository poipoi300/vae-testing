import re

def make_path_safe(s):
    # Replace non-alphanumeric characters with underscores
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    # Remove duplicate underscores
    s = re.sub(r'_+', '_', s)
    # Strip leading/trailing underscores
    return s.strip('_')
