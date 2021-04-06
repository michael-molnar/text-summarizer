import re 
import numpy as np
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# Load the encoder and decoder models 
#encoder_model = load_model('encoder_model.h5')
#decoder_model = load_model('decoder_model.h5')
# Import the tokenizers
#text_tokenizer = pickle.load(open('text_tokenizer.pkl', 'rb'))
#summary_tokenizer = pickle.load(open('summary_tokenizer.pkl', 'rb'))

# Here we will pre-process input text 

# Create the list of contractions and what they will be mapped to
"""
contraction_mappings = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", 
                       "could've": "could have", "couldn't": "could not", "didn't": "did not", 
                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 
                       "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 
                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 
                       "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
                       "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 
                       "mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", 
                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                       "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", 
                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", 
                       "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
                       "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", 
                       "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", 
                       "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                       "you'll've": "you will have", "you're": "you are", "you've": "you have"}
# Import the list of stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
            "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
            'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
            't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', 
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
            "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def clean_text(text):
    """
    Performs all necessary cleaning operations on text input.
    """
    # Lowercase the text
    new_text = text.lower()
    # Remove special characters
    new_text = re.sub(r'\([^)]*\)', '', new_text)
    new_text = re.sub('"', '', new_text)
    # Expand contractions
    new_text = ' '.join([contraction_mappings[x] if x in contraction_mappings else x for x in new_text.split(' ')])
    # Remove 's 
    new_text = re.sub(r"'s\b", '', new_text)
    # Replace non-alphabetic characters with a space
    new_text = re.sub('[^a-zA-Z]', ' ', new_text)
    # Split the text into tokens and remove stopwords
    tokens = [word for word in new_text.split() if word not in stopwords]
    # Keep only tokens that are longer than one letter long 
    words = []
    for t in tokens:
        if len(t) > 1:
            words.append(t)
    # Return a rejoined string 
    return (' '.join(words).strip())

def decode_sequence(input_seq):
    """
    #Take the processed input sequence and generate the summary using the 
    #econder and decoder models.
    """
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (MAX_LEN_SUMMARY-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

# Set the maximum summary and text length 
MAX_LEN_SUMMARY = 10
MAX_LEN_TEXT = 50

# Create mappings with the imported tokenizers
reverse_target_word_index = summary_tokenizer.index_word 
reverse_source_word_index = text_tokenizer.index_word 
target_word_index = summary_tokenizer.word_index
"""

@app.route('/')
def home():
	return render_template('index.html')
"""
@app.route('/predict', methods=['POST'])
def predict():
	# Take user input text
	input_text = request.form['your_text']
	# Clean the text
	review = clean_text(input_text)
	# Condense to the maximum length of 50 tokens
	condensed_review = review.split()[:MAX_LEN_TEXT]
	# Join into a string
	condensed_review = ' '.join(condensed_review)
	# Tokenize the review
	tokenized_review = text_tokenizer.texts_to_sequences([condensed_review])
	# Pad the sequence
	pad_tokenized_review = pad_sequences(tokenized_review, maxlen = MAX_LEN_TEXT, padding = 'post')
	# Generate the summary 
	summary = decode_sequence(pad_tokenized_review.reshape(1, MAX_LEN_TEXT))

	# Show the output 
	pred_text0 = 'Your Text:'
	pred_text1 = 'Summary:'

	return render_template('index.html', your_text_header = pred_text0, your_text = input_text, 
		summary_text = pred_text1, prediction_text = summary)
"""
if __name__ == '__main__':
  app.run(debug=True)