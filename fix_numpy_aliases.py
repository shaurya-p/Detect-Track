import os
import re

def fix_numpy_aliases(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace deprecated numpy types
                new_content = re.sub(r'\bnp\.float\b', 'float', content)
                new_content = re.sub(r'\bnp\.int\b', 'int', new_content)
                new_content = re.sub(r'\bnp\.bool\b', 'bool', new_content)
                new_content = re.sub(r'\bnp\.object\b', 'object', new_content)
                new_content = re.sub(r'\bnp\.long\b', 'int', new_content)

                if content != new_content:
                    print(f"Fixing file: {file_path}")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

if __name__ == "__main__":
    fix_numpy_aliases("external/ByteTrack")
    print("✅ Finished fixing all numpy deprecated aliases.")
