import nltk
import sys
import string
import os
import math

nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    ret_me = {}
    files = os.listdir(directory)
    for file in files:
        f = open(os.path.join(directory, file), "r")
        ret_me[file] = f.read()

    return ret_me

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    tokenized = nltk.tokenize.word_tokenize(document)

    words = []
    for token in tokenized:
        word = ''.join([ch for ch in token.lower() if ch not in string.punctuation])
        if len(word) > 0 and word not in nltk.corpus.stopwords.words("english"):
            words.append(word)

    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_documents = len(documents)

    word_doc_count = {}
    for key in documents:
        unique_words = set()
        for word in documents[key]:
            if word not in unique_words:
                unique_words.add(word)
        for word in unique_words:
            if word not in word_doc_count:
                word_doc_count[word] = 1
            else:
                word_doc_count[word] = word_doc_count[word] + 1

    idfs = {}
    for word in word_doc_count:
        idfs[word] = math.log(num_documents/word_doc_count[word], math.e)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Files ranked according to the sum of the tf-idf for all words in query
    file_ifidf= {}
    for qword in query:

        # If this isn't in our dataset then just skip it
        if qword in idfs:
            #tf idf = the number of times the word appears in the document by the idf value for that term
            idf = idfs[qword]

            # Count how many times it appears in each doc
            for key in files:
                count = 0
                for word in files[key]:
                    if word == qword:
                        count += 1
                if key not in file_ifidf:
                    file_ifidf[key] = count * idf
                else:
                    file_ifidf[key] += (count * idf)

    # Sort the files by their values
    ordered_docs = sorted(file_ifidf, key=file_ifidf.get, reverse=True)

    return ordered_docs[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # Sentences to matching word measure (Sum of idf values for the word in the sentence)
    sent_to_mwm = {}
    for qword in query:

        # If this isn't in our dataset then just skip it
        if qword in idfs:
            idf = idfs[qword]

            for sent in sentences:
                if qword in sentences[sent]:
                    # Add idf value to the total
                    if sent not in sent_to_mwm:
                        sent_to_mwm[sent] = idf
                    else:
                        sent_to_mwm[sent] += idf

    # QTD is Query Term Density, its what percentage of the sentence is made up of words from the query
    sent_to_qtd = {}
    for sent in sentences:
        count = 0
        for qword in query:
            for word in sentences[sent]:
                if word == qword:
                    count += 1
        words_in_sentence = len(sentences[sent])
        sent_to_qtd[sent] = count / words_in_sentence if words_in_sentence > 0 else 0

    # Sort the files by their values and query term density
    ordered_sents = sorted(sent_to_mwm, key=lambda s: (sent_to_mwm[s], sent_to_qtd[s]), reverse=True)

    return ordered_sents[:n]

if __name__ == "__main__":
    main()
