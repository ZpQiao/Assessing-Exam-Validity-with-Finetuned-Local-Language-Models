from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch

processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

%%capture
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
from IPython.display import display

pdf_path = "02405_Dec2019.pdf" 
pdf = fitz.open(pdf_path)

images = []
for i in range(pdf.page_count):
    pix = pdf[i].get_pixmap(dpi=144)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    images.append(image)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(images[0])
ax1.axis('off')

ax2.imshow(images[-1])
ax2.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0, left=0, right=1, top=1, bottom=0)
plt.show()
from tqdm import tqdm

all_text = ""

for page_image in tqdm(images):
    pixel_values = processor(images=page_image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        min_length=1,
        max_length=4096,
        eos_token_id=processor.tokenizer.eos_token_id,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )
    text = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    all_text += text + "\n\n"
print('\n'.join(all_text.splitlines()[:50]))

import re

raw_text = all_text

patterns = [
    r"\\\[\\begin\{(?:array|tabular)\}(?:\[\]\{[^\}]*\})?",
    r"\\begin\{(?:array|tabular)\}(?:\[\]\{[^\}]*\})?",
    r"\\end\{(?:array|tabular)\}\\\]",
    r"\\end\{(?:array|tabular)\}",
    r"\\\]",
    r"\\\["
]

for pat in patterns:
    raw_text = re.sub(pat, "", raw_text)
    
raw_text = re.sub(r'\\\\\s*(?=\d+\\\s*\\square)', '\n', raw_text)

import re
import json
lines = raw_text.splitlines()
questions = []
i = 0
while i < len(lines):
   line = lines[i].strip()
   qn_match = re.match(r"Question\s*(\d+)", line)
   if qn_match:
       question_id = qn_match.group(1)
       # find previous non-empty line as context
       j = i - 1
       while j >= 0 and lines[j].strip() == "":
           j -= 1
       context = lines[j].strip() if j >= 0 else ""
       # find next non-empty line as question
       k = i + 1
       while k < len(lines) and lines[k].strip() == "":
           k += 1
       raw_question = lines[k].strip() if k < len(lines) else ""
       # Check if it starts with a number
       if re.match(r"^(\(?\d+\)?|\\\(\d+)", raw_question):
           question = ""
           l = k   # start collecting options directly from this line
       else:
           question = raw_question
           l = k + 1  # start collecting options from next line normally
       # collect the next 6 non-empty lines as options, only those starting with numbers
       options = []
       count = 0
       while l < len(lines) and count < 6:
           opt_line = lines[l].strip()
           # only collect lines starting with numbers
           if re.match(r"^(\(?\d+\)?|\\\(\d+)", opt_line):
               options.append(opt_line)
               count += 1
           l += 1
       questions.append({
           "question_id": question_id,
           "context": context,
           "question": question,
           "options": options
       })
       i = l
       continue
   i += 1

print(questions[0])
with open('questions.json', 'w', encoding='utf-8') as f:
   json.dump(questions, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(questions)} questions to questions.json")