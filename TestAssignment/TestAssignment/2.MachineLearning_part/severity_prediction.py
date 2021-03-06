import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import tensorflow as tf
import tensorflow.python.saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import re


#----------Data Preprocessing----------

# Load data
data=pd.read_csv('commits.csv')
data_issues=pd.read_csv('issues.csv')

# If 'key' is equal append on right, else append on bottom-right
data = data.join(data_issues.set_index('key'), on='key', how = 'outer')

data = data[['message_encoding', 'severity']] # Select 'message_encoding' and 'severity' columns
data = data[data.message_encoding.notnull() & data.severity.notnull()] # Clean columns from NaN valuses
data = data.reset_index(drop=True) # Reset index to start from 0 to n_samples - 1

# Clean feature text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split()) # delete stopwors from text
    return text
    
data['message_encoding'] = data['message_encoding'].apply(clean_text)

# Split data to feature and label
dataX = data.message_encoding
dataY = data.severity

# Convert data to numpy array
dataY = np.array(dataY)
dataX = np.array(dataX)

# Split data to train and test sample
# train 80%, test 20%
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, random_state=np.random.randint(0, 100 ))

# Text vectorization
''' By Google engineers experiments best vectorization strategy for samples / words per sample ratio <1500 
is n-gram in MLP Neural Network. In our case number of samples is 11475, average 108 words per sample, ratio is 106. 
ratio = len(data['message_encoding']) / data['message_encoding'].str.split().str.len().mean() '''

# Basic n_gram parameters
n_gram = (1, 2) # n-gram sizes for tokenizing text(unigram + bigram)
token_mode = 'word' # whether text should be split into word or character n-grams
min_frequency = 2 # minimum n_gram frequency below which a token will be discarded.

vectorizer = TfidfVectorizer(ngram_range = n_gram,
                             strip_accents = 'unicode', 
                             decode_error = 'replace', 
                             analyzer = token_mode,
                             min_df = min_frequency)

# Learn vocabulary from training texts
vectorizer_vocabulary = vectorizer.fit(trainX)

# Save vocabulary for further predictions usage
pickle.dump(vectorizer_vocabulary, open("vectorizer_vocabulary.pickle", "wb"))

# Vectorize train and test features
trainX = vectorizer.transform(trainX)
testX = vectorizer.transform(testX)


# Select top 'k' of the vectorized features (reduce num of words in vectors to best 'k' for prediction labels)
'''This will: Reduce Overfitting (Less redundant data means less possibility of making decisions 
based on redundant data/noise), Improve Accuracy (Less misleading data means modeling accuracy improves)
Reduce Training Time (Less data means that algorithms train faster)'''
selector = SelectKBest(f_classif, k=min(20000, trainX.shape[1]))


# Calculate feature word importance
selector.fit(trainX, trainY)

# Save selector for further predictions usage
pickle.dump(selector, open("selector.pickle", "wb"))

# Select features
trainX = selector.transform(trainX).astype('float32')
testX = selector.transform(testX).astype('float32')

# Creating the One Hot Encoder for label
dataY = dataY.reshape(dataY.size, 1)
trainY = trainY.reshape(trainY.size, 1)
testY = testY.reshape(testY.size, 1)

oneHot = OneHotEncoder(categories='auto')
oneHot.fit(dataY)
trainY = oneHot.transform(trainY).toarray()
testY = oneHot.transform(testY).toarray()

trainX = trainX.toarray()
testX = testX.toarray()


#----------Create MLP Neural Network model----------

# Basic model parameters
learning_rate = 0.01
epochs = 3000
step = 500
batch_size = 512

n_features = trainX.shape[1]
n_categories = trainY.shape[1]

# Create placeholders for input/features and labels
inputs = tf.placeholder(tf.float32, [None, n_features], name='features')
labels = tf.placeholder(tf.float32, [None, n_categories], name='labels')

# Create 3 hidden and output layers
hidden1 = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu)
hidden2 = tf.layers.dense(inputs=hidden1, units=1024, activation=tf.nn.relu)
hidden3 = tf.layers.dense(inputs=hidden2, units=1024, activation=tf.nn.relu)
output = tf.layers.dense(inputs=hidden3, units=n_categories)
prediction = tf.nn.softmax(output, name='prediction')

# Calculating loss/cost function
loss = tf.losses.softmax_cross_entropy(labels, output)

# Optimizer for reduce loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Measure accuracy
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        # Start training
        for epoch in range(epochs):
            # Create random batch 
            index = np.random.choice(len(trainX), batch_size, replace=False)
            batch_X = trainX[index]
            batch_Y = trainY[index]
            # Learn model
            _, l = sess.run([optimizer, loss], feed_dict={inputs: batch_X  , labels: batch_Y })
            # Print training parameters
            if (epoch+1) % step == 0:
                average_loss = sess.run(tf.reduce_mean(l))
                train_accuracy = accuracy.eval(feed_dict={inputs: trainX, labels: trainY})
                print("Epoch: {}".format(epoch + 1), " cost={}".format(average_loss), " training accuracy %g"%(train_accuracy))
        
        print("Optimization Finished!")

        # Test model
        print("Test accuracy: %g"%accuracy.eval(feed_dict={inputs: testX, labels: testY}))
         
        # Save model        
        builder = tf.saved_model.builder.SavedModelBuilder('saved_model')
        signature = predict_signature_def(inputs={'inputs': inputs},
                                  outputs={'prediction': prediction})
        builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
        builder.save()

