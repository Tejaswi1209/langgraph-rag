from markitdown import MarkItDown

import os
 
md = MarkItDown()
 
def extract_text(file_path: str, file_name: str) -> str:

    """

    Extract text to markdown using MarkItDown.

    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(f"File not found: {file_path}")
 
    # Save markdown to same folder

    md_file_path = file_path + ".md"

    if os.path.exists(md_file_path):

        with open(md_file_path, "r", encoding="utf-8") as f:

            return f.read()
 
    result = md.convert(file_path)

    text = result.text_content or ""
 
    if not text.strip():

        return ""
 
    with open(md_file_path, "w", encoding="utf-8") as f:

        f.write(text)
 
    print(f"✅ Markdown file created: {md_file_path}")

    return text

 