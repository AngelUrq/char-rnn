import re
import torch
from cleantext import clean

def clean_text(text):
  return clean(text,
      fix_unicode=True,               # fix various unicode errors
      to_ascii=True,                  # transliterate to closest ASCII representation
      lower=True,                     # lowercase text
      no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
      no_urls=False,                  # replace all URLs with a special token
      no_emails=False,                # replace all email addresses with a special token
      no_phone_numbers=False,         # replace all phone numbers with a special token
      no_numbers=False,               # replace all numbers with a special token
      no_digits=False,                # replace all digits with a special token
      no_currency_symbols=True,      # replace all currency symbols with a special token
      no_punct=True,                 # remove punctuations
      replace_with_punct="",          # instead of removing punctuations you may replace them
      replace_with_url="",
      replace_with_email="",
      replace_with_phone_number="",
      replace_with_number="",
      replace_with_digit="0",
      replace_with_currency_symbol="",
      lang="en"                       # set to 'de' for German special handling
  )

class CharDataset(torch.utils.data.Dataset):

    def __init__(self, text_file, window_size):
        with open(text_file, encoding='ISO-8859-1') as file:
            self.text = file.read()
        
        self.window_size = window_size

        self.text = self.text.replace('\n', ' ')
        self.text = re.sub(' +', ' ', self.text)
        self.text = self.text.lower()
        self.text = clean_text(self.text)

        print(f'Text has {len(self.text)} characters')

        self.chars = "".join(sorted(set(self.text)))
        print(f'Distinct characters {len(self.chars)}')

        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if index + self.window_size >= len(self.text):
            index = 0
            
        x = self.text[index:index+self.window_size]
        x = torch.tensor([self.char_to_ix[ch] for ch in x]).long()

        y = self.text[index+self.window_size]
        y = torch.tensor([self.char_to_ix[ch] for ch in y]).long()

        return x, y
