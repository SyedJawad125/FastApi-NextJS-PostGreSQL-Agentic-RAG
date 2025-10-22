from PyPDF2 import PdfReader

def extract_text_from_file(file) -> str:
    """Extract text from uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text
