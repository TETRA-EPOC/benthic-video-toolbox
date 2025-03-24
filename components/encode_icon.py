from base64 import b64encode
import pyperclip
import sys

filename = sys.argv[1]
with open(filename, "rb") as f:
    pyperclip.copy(b64encode(f.read()).decode("ascii"))
    print("Copied to clipboard.")