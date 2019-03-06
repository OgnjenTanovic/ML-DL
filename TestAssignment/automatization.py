import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import tensorflow as tf

lr = [0.01, 0.001, 0.0001, 0.00001]
epochs = [5000, 10000]
bs = [50, 200, 500]
lu = [10, 50, 100]
lf = ['relu', 'softmax']
lo = ['y_', 'prediction']
cf = ['softmax', 'sigmoid']

#----------Data Preprocessing----------

# Load data
data=pd.read_csv('commits.csv')
data_issues=pd.read_csv('issues.csv')

# If 'key' is equal append on right, else append on bottom-right
data = data.join(data_issues.set_index('key'), on='key', how = 'outer')

data = data[['message_encoding', 'severity']] # Select 'message_encoding' and 'severity' columns
data = data[data.message_encoding.notnull() & data.severity.notnull()] # Clear columns from NaN valuses
data = data.reset_index(drop=True) # Reset index to start from 0 to n_samples - 1

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


# Select top 'k' of the vectorized features (reduce num of words in vectors to best 'k' ones for prediction labels)
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

#trainY = np.array(trainY).reshape(trainY.size, 1)
#testY = np.array(testY).reshape(testY.size, 1)


for learning_rate in lr:

    for epoch in epochs:
        
        for batch_size in bs:
            
            for layer_units in lu:
                
                for layer_function in lf:
                    
                    for logits in lo:
                
                        for cost_function in cf:
                            #Here


                            #----------Create MLP Neural Network model----------
                            
                            tf.reset_default_graph()

                            
                            step = 500
                            

                            n_features = trainX.shape[1]
                            n_categories = trainY.shape[1]

                            # Create placeholders
                            # X is placeholdre for features
                            X = tf.placeholder(tf.float32, [None, n_features], name='features')
                            # y is placeholder for label
                            y = tf.placeholder(tf.float32, [None, n_categories], name='label')

                            # Create input, 2 hidden and output variables
                            input_units = n_features
                            hidden1_units = layer_units
                            hidden2_units = layer_units
                            output_units = n_categories

                            hidden1W = tf.Variable(tf.zeros([input_units, hidden1_units]), name='hidden1W')
                            hidden1b = tf.Variable(tf.zeros([hidden1_units]), name='hidden1b')

                            hidden2W = tf.Variable(tf.zeros([hidden1_units, hidden2_units]), name='hidden2W')
                            hidden2b = tf.Variable(tf.zeros([hidden2_units]), name='hidden2b')

                            outputW = tf.Variable(tf.zeros([hidden2_units, output_units]), name='outputW')
                            outputb = tf.Variable(tf.zeros([output_units]), name='outputb')

                            # Declare layers hypothesis functions
                            # Input layer
                            input_layer = X    
                            # Hidden layers
                            if layer_function == 'relu':
                                hidden1_layer = tf.nn.relu(tf.add(tf.matmul(input_layer, hidden1W), hidden1b))
                                hidden2_layer = tf.nn.relu(tf.add(tf.matmul(hidden1_layer, hidden2W), hidden2b))
                            else:
                                hidden1_layer = tf.nn.softmax(tf.add(tf.matmul(input_layer, hidden1W), hidden1b))
                                hidden2_layer = tf.nn.softmax(tf.add(tf.matmul(hidden1_layer, hidden2W), hidden2b))
                            # Output layer - our prediction function
                            y_ = tf.add(tf.matmul(hidden2_layer, outputW), outputb)
                            prediction = tf.nn.softmax(y_, name='prediction')

                            # Calculating cost
                            if logits == 'y_':
                                if cost_function == 'softmax':
                                    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)
                                else:
                                    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_)
                            else:
                                if cost_function == 'softmax':
                                    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
                                else:
                                    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)

                            # Optimizer
                            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

                            with tf.Session() as sess:

                                    # Initialize all variables
                                    sess.run(tf.global_variables_initializer())

                                    for epo in range(epoch):

                                        # Create random batch 
                                        index = np.random.choice(len(trainX), batch_size, replace=False)
                                        batch_X = trainX[index]
                                        batch_Y = trainY[index]

                                        # Start training
                                        _, c = sess.run([optimizer, cost], feed_dict={X: batch_X  , y: batch_Y })
                                        
                                        if (epo+1) % step == 0:
                                            mse = sess.run(tf.reduce_mean(c))
                                            print("Epoch: {}".format(epo + 1), "cost={}".format(mse), trainX.shape)
                                    
                                    print("Optimization Finished!")

                                    # Test model
                                    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                                    predict = sess.run(prediction, feed_dict={X: testX})
                                    position = sess.run(tf.argmax(predict, 1))
                                    correct = testY
                                    decode = oneHot.inverse_transform(testY)
                                    counter=0
                                    for a, b, c, d in zip(predict, position, correct, decode):
                                        if counter<20:
                                            print(np.round(a,  decimals=2), b, c, d)
                                            counter+=1
                                    #print(sess.run(correct_prediction, feed_dict={X: testX, y: testY}))
                                    
                                    string_line = '\n' + '*'*50 + '\n'

                                    # Calculate accuracy for 3000 examples
                                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                                    a = accuracy.eval({X: testX, y: testY})
                                    print("Accuracy test:", a )
                                    
                                    string_line+='\nAccuracy test:' + str(a)
                                    
                                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                                    b = accuracy.eval({X: trainX, y: trainY})
                                    print("Accuracy train:", b)
                                    
                                    string_line+='\nAccuracy train:' + str(b)
                                    string_line+='\nLearning rate:' + str(learning_rate) + ' Epochs:' + str(epoch) + ' Batch size:' + str(batch_size) + ' Hidden units:' + str(layer_units) + ' Layer function:' + str(layer_function) + ' Logits:' + str(logits) + ' Cost function:' + str(cost_function)
                                    
                                    f=open("log.txt","a+")
                                    f.write(string_line)
                                    f.close() 
               

