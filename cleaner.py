import re
import string


def clean_gutenberg_text(file_path):
    """Cleans a Project Gutenberg book by removing headers, footers, and punctuation."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Remove Gutenberg header and footer
    header_pattern = r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*"
    footer_pattern = r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*"

    text = re.split(header_pattern, text, flags=re.IGNORECASE)[-1]  # Keep only content after header
    text = re.split(footer_pattern, text, flags=re.IGNORECASE)[0]  # Keep only content before footer

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits + "”" + "“" + "—”" + "‘"))

    # Remove single-character words except 'i'
    text = re.sub(r'\b(?!i\b)\w\b', '', text, flags=re.IGNORECASE)

    return text
