#!/usr/bin/env python
# coding: utf-8

# ### Importing the Required Packages for the Project

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM, Activation, Dense,Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report


# ### Loading the features and Labels

# In[ ]:


data = pd.read_csv('X_data.txt', sep=',', header = None, names=['Sentences'])


# In[ ]:


label = pd.read_csv('y_data.txt', sep=',', header = None, names=['Labels'])


# In[ ]:


data = data.Sentences.values
label = label.Labels.values


# In[ ]:


len(data)


# In[ ]:


len(label)


# ### Tokenizing the features and padding the dataset

# In[ ]:


tok = Tokenizer()
tok.fit_on_texts(data)
max_len = max([len(s.split()) for s in data])
max_words = 45000
sequences = tok.texts_to_sequences(data)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len) #padding = 'post'


# In[ ]:


print(max_words)
print(max_len)


# ### Splitting the Data Set into Train and Test using Stratification

# In[ ]:


train_data, test_data, train_label, test_label = train_test_split(sequences_matrix,label,stratify = label,test_size = 0.15, random_state = 650) 


# ### Using the Sequential Neural Network for training and validating the Model

# In[ ]:


model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=3,padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size = 4))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


# ### Splitting the Data into Train and Validation Set and fitting the Model using EarlyStop to prevent Overfitting

# In[ ]:


model.fit(train_data,train_label,batch_size=128,epochs = 10,validation_split = 0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta = 0.0001)])


# In[ ]:


Metrics = pd.DataFrame(model.history.history)
Metrics


# In[ ]:


Metrics[['loss', 'val_loss']].plot()
plt.show()


# In[ ]:


Metrics[['accuracy', 'val_accuracy']].plot()
plt.show()


# In[ ]:


accuracy = model.evaluate(test_data,test_label)


# In[ ]:


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accuracy[0],accuracy[1]*100))


# In[ ]:


y_pred = model.predict(test_data)


# In[ ]:


y_pred = np.around(y_pred)


# In[ ]:


print(classification_report(test_label, y_pred))


# ### Loading the Test Dataset for Prediction

# In[ ]:


test = pd.read_csv('X_test.txt', sep=',', header = None, names=['Test_Data'])


# In[ ]:


test = test.Test_Data.values
test


# In[ ]:


test = test.astype(str)


# In[ ]:


tok.fit_on_texts(test)
test = tok.texts_to_sequences(test)
test = sequence.pad_sequences(test, maxlen=max_len)


# ### Model Prediction on Test Data

# In[ ]:


output_test = model.predict(test)


# In[ ]:


x = np.around(output_test)
x = (x.astype(int))
print(x)


# In[ ]:


output_label = (["%d" % a for a in x])
output_label


# ### Saving the Output in Requested format to a text file

# In[ ]:


with open('final.txt', 'w', newline = '\n') as file:
    for item in output_label:
        file.write(item + '\n')

