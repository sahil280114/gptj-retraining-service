import os
import re
import random

from pathlib import Path

import ftfy
import tensorflow as tf
from lm_dataformat import Reader
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from google.cloud import storage

SERVICE_ACCOUNT_KEY="service_account.json"

def upload_to_gcs(source_file):
    """Uploads a file to the bucket."""
    storage_client = storage.Client.from_service_account_json(
        SERVICE_ACCOUNT_KEY)
    bucket = storage_client.bucket("booste-gpt-j")
    blob = bucket.blob(f"train/data/{source_file}")

    blob.upload_from_filename(source_file)

    print(
        "File {} uploaded to GCS.".format(
            source_file
        )
    )


def get_files(input_dir):
    filetypes = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]
    files = [list(Path(input_dir).glob(f"*{ft}")) for ft in filetypes]
    # flatten list of list -> list and stringify Paths
    return [str(item) for sublist in files for item in sublist]


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    """
    writes data to tfrecord file
    """
    feature = {
        "text": _int64_feature(data)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_tfrecord(sequences, fp):
    with tf.io.TFRecordWriter(fp) as writer:
        for seq in sequences:
            write_to_file(writer, seq)


def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i + n] for i in range(0, len(l), n)]


def enforce_min_unique(seqs, min_unique_tokens, enc, verbose=False):
    #Exclude repetitive documents made up of < min_unique_tokens unique tokens
    for seq in tqdm(seqs, mininterval=1, smoothing=0):
        if len(set(seq)) >= min_unique_tokens:
            yield seq
        elif verbose:
            text = enc.decode(seq)
            print(f"excluding with {len(set(seq))} unique tokens:\n\n{repr(text)}\n\n")


def eot_splitting_generator(string_iterable, encoder):
    """
    Given strings, splits them internally on <|endoftext|> and yields (generally more) strings
    """
    for doc in string_iterable:
        for d in doc.split(encoder.eos_token):
            if len(d) > 0:
                yield d


def prep_and_tokenize_generator(string_iterable, encoder, normalize_with_ftfy, normalize_with_wikitext_detokenize):
    """
    Given strings, does data cleaning / tokenization and yields arrays of tokens
    """
    for doc in string_iterable:
        if normalize_with_ftfy:  # fix text with ftfy if specified
            doc = ftfy.fix_text(doc, normalization='NFKC')
        if normalize_with_wikitext_detokenize:
            doc = wikitext_detokenizer(doc)
        tokens = encoder.encode(doc) + [encoder.eos_token_id]
        yield tokens


def file_to_tokenized_docs_generator(file_path, encoder):
    """
    Given a file path, reads the file and tokenizes the contents
    Yields token arrays of arbitrary, unequal length
    """
    reader = Reader(file_path)
    string_iterable = reader.stream_data(threaded=False)
    string_iterable = eot_splitting_generator(string_iterable, encoder)

    token_list_gen = prep_and_tokenize_generator(string_iterable,
                                                 encoder,
                                                 normalize_with_ftfy=True,
                                                 normalize_with_wikitext_detokenize=True
                                                 )
    return token_list_gen


def read_files_to_tokenized_docs(files, encoder):
    docs = []

    random.shuffle(files)

    for f in tqdm(files, mininterval=10, smoothing=0):
        docs.extend(file_to_tokenized_docs_generator(f, encoder))

    # shuffle at individual document level
    random.shuffle(docs)

    return docs


def arrays_to_sequences(token_list_iterable, sequence_length=2049):
    """
    Given token arrays of arbitrary lengths, concats/splits them into arrays of equal length
    Returns equal-length token arrays, followed by a a final array of trailing tokens (which may be shorter)
    """
    accum = []
    for l in token_list_iterable:
        accum.extend(l)

        if len(accum) > sequence_length:
            chunks = split_list(accum, sequence_length)
            for chunk in chunks[:-1]:
                yield chunk
            accum = chunks[-1]

    if len(accum) > 0:
        yield accum


def chunk_and_finalize(arrays, encoder):
    sequences = list(arrays_to_sequences(arrays))

    full_seqs, trailing_data = sequences[:-1], sequences[-1]

    #200 is a good default value for minimun unique tokens (From GPT-J finetuning guide)
    full_seqs = list(enforce_min_unique(full_seqs, 200, encoder, verbose=False))
    random.shuffle(full_seqs)

    return full_seqs, trailing_data


def create_tfrecords(files,name):
    GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20  # disables a misleading warning
    encoder = GPT2TokenizerFast.from_pretrained('gpt2')

    random.seed(10)

    all_sequences_across_epochs = []

    docs = read_files_to_tokenized_docs(files, encoder)

    full_seqs, trailing_data = chunk_and_finalize(docs, encoder)

    all_sequences_across_epochs.extend(full_seqs)

    print(f"dropped {len(trailing_data)} tokens of trailing data")

    total_sequence_len = len(all_sequences_across_epochs)

    fp = os.path.join('', f"{name}_{total_sequence_len}.tfrecords")
    write_tfrecord(all_sequences_across_epochs, fp)
    upload_to_gcs(f"{name}_{total_sequence_len}.tfrecords")
    return (f"gs://booste-gpt-j/train/data/{name}_{total_sequence_len}.tfrecords",total_sequence_len)