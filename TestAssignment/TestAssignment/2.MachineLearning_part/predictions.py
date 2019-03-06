import tensorflow as tf
import pickle
import re
import os.path as pt

# Clean input text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split()) # delete stopwors from text
    
    return [text]



    
def make_prediction(mystr):

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], pt.join(pt.dirname(__file__),'saved_model'))
        graph = tf.get_default_graph()        
            
        inputs = graph.get_tensor_by_name("features:0")
        predict = graph.get_tensor_by_name("prediction:0")
        
        vectorizer = pickle.load(open(pt.join(pt.dirname(__file__),'vectorizer_vocabulary.pickle'), 'rb'))
        selector = pickle.load(open(pt.join(pt.dirname(__file__),'selector.pickle'), 'rb'))


        if mystr:
            mystr = clean_text(mystr)# Clean input
            myx = vectorizer.transform(mystr)# Vectorize input
            myx = selector.transform(myx).astype('float32')# Select input
            
            myx = myx.toarray()
            
            prediction = sess.run(predict, feed_dict={inputs:myx})
            prediction = sess.run(tf.argmax(prediction, 1))
            
            # Decode prediction
            classes = [0, 1, 3, 5, 10]
            positions = range(5)
            for i, p in zip(positions, classes):
                if prediction==i:
                    prediction=p
                    return prediction       
        else:
            return 'Empty field'

