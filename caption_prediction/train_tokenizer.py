# Imports
import os
import argparse
import pandas as pd

# Transformers Imports
from transformers import AutoTokenizer



# CLI Interface
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True, help='Data directory')
args = parser.parse_args()

# Build text corpus
text_corpus = []
splits = ['train', 'valid']
for s in splits:
    df = pd.read_csv(os.path.join(args.datadir, f'ImageCLEFmedical_Caption_2023_caption_prediction_{s}_labels.csv'), delimiter='\t')
    for _, row in df.iterrows():
        sent = str(row["caption"])
        text_corpus.append(sent)

print("Total number of examples in text corpus: ", len(text_corpus))



# Function: Get the training corpus
def get_training_corpus():
    return (text_corpus[i : i + 1000]
        for i in range(0, len(text_corpus), 1000)
    )



# Retrain a new tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
training_corpus = get_training_corpus()
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, length=len(text_corpus), vocab_size=500000)
tokenizer.save_pretrained("imageclef23_tokenizer")

# Check new tokenizer
new_tokenizer = AutoTokenizer.from_pretrained("imageclef23_tokenizer")
print(new_tokenizer)
