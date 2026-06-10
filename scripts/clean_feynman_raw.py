import os
import re
from pathlib import Path

def clean_feynman_files():
    feynman_dir = Path("data/raw/feynman")
    if not feynman_dir.exists():
        print(f"Error: {feynman_dir} does not exist.")
        return

    txt_files = list(feynman_dir.glob("**/*.txt"))
    print(f"Found {len(txt_files)} text files to clean.")

    # Match LOADING PAGE... up to Editor... New Millennium Edition and any trailing whitespace
    pattern = re.compile(r"LOADING PAGE\.\.\.[\s\S]*?Editor, The Feynman Lectures on Physics New Millennium Edition\s*", re.IGNORECASE)
    
    cleaned_count = 0
    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            if "LOADING PAGE..." in content:
                new_content, count = pattern.subn("", content)
                if count > 0:
                    file_path.write_text(new_content, encoding="utf-8")
                    cleaned_count += 1
        except Exception as e:
            print(f"Error cleaning {file_path}: {e}")

    print(f"Successfully cleaned {cleaned_count} / {len(txt_files)} files.")

if __name__ == "__main__":
    clean_feynman_files()
