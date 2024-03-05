import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Installed {package}")

try:
    print("Automatically Installing pandas, nltk, and pytorch pretrained bert")
    install('pandas')
    install('nltk')
    install('pytorch_pretrained_bert')
    install('scikit-learn')
except:
    print("Error Installing, Restart Session and Try again :(")


print("\nImporting necessary libraries")
import os
import numpy as np
import pandas as pd
import nltk
import re
import torch
from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


print("\nLoading Data")
data = []
def load_data(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                data.append({'filename': filename, 'text': text})
    return pd.DataFrame(data)


print("Preprocessing...\n")
def replace_newlines(df):
    df["text"] = df["text"].str.replace("\n", " ")
    df["text"] = df["text"].str.replace(r"\s+", " ")
    df.loc[:,'text'] = df['text'].str.replace(r'\s*https?://\S+(\s+|$)', ' ').str.strip()
    return df

def lower_case_text(df):
    df['text'] = df['text'].str.lower()
    return df

def remove_specific_words(input_string, words_to_remove):
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

# This is the function to generate sentence embeddings
def bertSent_embeding(sentences):
    marked_sent = ["[CLS] " +item + " [SEP]" for item in sentences]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_sent = [tokenizer.tokenize(item) for item in marked_sent]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(item) for item in tokenized_sent]
    tokens_tensor = [torch.tensor([item]) for item in indexed_tokens]
    segments_ids = [[1] * len(item) for ind,item in enumerate(tokenized_sent)]
    segments_tensors = [torch.tensor([item]) for item in segments_ids]
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    assert len(tokens_tensor) == len(segments_tensors)
    encoded_layers_list = []
    for i in range(len(tokens_tensor)):
        with torch.no_grad():
            encoded_layers, _ = bert_model(tokens_tensor[i], segments_tensors[i])
        encoded_layers_list.append(encoded_layers)
    token_vecs_list = [layers[11][0] for layers in encoded_layers_list]
    sentence_embedding_list = [torch.mean(vec, dim=0).numpy() for vec in token_vecs_list]
    return sentence_embedding_list


'''This function makes clusters of sentences and picks one sentence from each cluster
sortes them in order and then returns it.'''
def kmeans_sumIndex(sentence_embedding_list):
    # Here we can control the number of clusters
    n_clusters = np.ceil(len(sentence_embedding_list)**0.5)
    if n_clusters == 0: return False
    kmeans = KMeans(n_clusters=int(n_clusters))
    kmeans = kmeans.fit(sentence_embedding_list)
    sum_index,_ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sentence_embedding_list,metric='euclidean')
    sum_index = sorted(sum_index)
    return sum_index

'''This function generates the summary, first it calles the bertSent_embedding 
   function to generate embeddings and then it calls the kmeans algorithm, 
   finally it generates summary by concatinating the output of Kmeans algorithm'''
def bertSummarize(text, max_sequence_length=512):
    sentences = sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) <= max_sequence_length]
    sentence_embedding_list = bertSent_embeding(sentences)
    sum_index = kmeans_sumIndex(sentence_embedding_list)
    if sum_index == False: return 'Cannot summarize empty file'
    summary = ' '.join([sentences[ind] for ind in sum_index])
    return summary

def summarize_and_save(row, text_column="text", filename_column="filename"):
    summary = bertSummarize(row[text_column])
    filename = row[filename_column]
    splitted_name = filename.split('_')
    company_name = splitted_name[0]
    summary_filename = f"{filename[:-4]}_summary.txt"
    os.makedirs(f"annual_reports/{company_name}/Summary", exist_ok=True)
    with open(f"annual_reports/{company_name}/Summary/{summary_filename}", "w") as f:
        print(f"Saving summary in {company_name} as {summary_filename} ")
        f.write(summary)
    
def driver_function(folder_path):
    df = load_data(folder_path)
    df = replace_newlines(df)
    df = lower_case_text(df)
    words_to_remove = ["a", "an", "at", "the", "only"]
    df["text"] = df["text"].apply(remove_specific_words, args=(words_to_remove,))
    for i, row in df.iterrows():
        print("Summarizing ", row["filename"])
        summarize_and_save(row)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", message=".*DeprecationWarning.*")