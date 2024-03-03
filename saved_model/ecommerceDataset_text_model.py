#1. Setup - import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn,os,pickle
import h5py

# Import the data
filename = 'ecommerceDataset_normalized.h5'
df = pd.read_hdf(filename,key='ecommerce')

#3. Data inspection
df = df.dropna()
print("Shape of data", df.shape)
print("Data description:\n", df.describe().transpose())
print("Example data:\n", df.head())
print("NA values:\n", df.isna().sum())
print("Duplicate values:\n", df.duplicated().sum())

#4. Data preprocessing
#(A) Isolate the features and labels
normalized = df['normalized description'].values
labels = df['label'].values

#(B) Perform label encoding on category column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

#5. Perform train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(normalized, labels_encoded, train_size=0.8, random_state=42)

#6. Start with tokenization
# Define hyperparameters 
vocab_size = 100000
oov_token = "<OOV>"
max_length = 200
embedding_dim = 64

# Define the Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words = vocab_size,
    oov_token = oov_token,
    split = " "
)

tokenizer.fit_on_texts(X_train)

# Inspection on the Tokenizer
word_index = tokenizer.word_index
word_index

# Use the Tokenizer to transform text to tokens
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
print(X_train[0])
print(X_test_tokens[0])

#7. Perform padding and truncating
X_train_padded = keras.utils.pad_sequences(
    X_train_tokens,
    maxlen = max_length,
    padding = 'post',
    truncating = 'post'
)
X_test_padded = keras.utils.pad_sequences(
    X_test_tokens,
    maxlen = max_length,
    padding = 'post',
    truncating = 'post'
)
print(X_train_padded.shape)

# Create a function that can decode the tokens
#(A) Create a reverse word index
reverse_word_index = [(value,key) for (key,value) in word_index.items()]
reverse_word_index

# Create the function for the decoding
def decode_tokens(tokens):
    return " ".join([reverse_word_index.get(i,"?") for i in tokens])

print(X_train[3])
print("------------------")
print()

#8. Model development
model = keras.Sequential()
#(A) Create the Embedding layer to perform token embedding
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Proceed to build the RNN as the subsequent layers
model.add(keras.layers.Bidirectional(keras.layers.LSTM(32,return_sequences=False)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(len(np.unique(labels)),activation='softmax'))
model.summary()

#9. Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#10. Model training
early_stopping = keras.callbacks.EarlyStopping(patience=2)
max_epoch = 10
history = model.fit(X_train_padded,y_train,validation_data=(X_test_padded,y_test),epochs=max_epoch,callbacks=[early_stopping])

# Training results
# Plot the graphs of loss and accuracy
#(A) Loss graph
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss','Validation Loss'])
plt.show()

#(B) Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.show()

#10. Model deployment
#(A) Have an example input
test_string = r"mini tripod stand monopod mount adapter camera want camera range pocketwe gorillapod tripod camera people love capture landscape scenery people key put pursebackpack jacket pocket feature wire leg rubberize feet ball socket rotate degreemount camera anywhere wantonly wrap segment leg securely branch fence park bench anything convenient rrubberized foot grip provide enhance stability terrain come attachment use wellonce try never leave home stand phonescamera dslr gorilla pod leg attach tripod pole table always get angle photo videos gorilla pod stand go well use photography video graphy supply stand use book watch movie keep desk reference screen writing copy research multi purposed remove mount attach phone mount accessoryget well angle mobility use skype smartphone stand tripod imagination photography movie book use gorilla pod stand come clip long screw accessory"

#(B) Apply the preprocessing to the string (convert to tokens > padding > embedding(don't need because it's in the model))
# Convert to tokens
test_token = tokenizer.texts_to_sequences(test_string)
test_token

# Create a function to remove the empty arrays
def remove_empty(tokens):
    temp = []
    for i in tokens:
        if i!=[]:
           temp.append(i[0])
    return temp

test_token_processed = np.expand_dims(np.array(remove_empty(test_token)),axis=0)

# Perform padding
test_token_padded = keras.utils.pad_sequences(
    test_token_processed,
    maxlen = max_length,
    padding = 'post',
    truncating = 'post'
)

#(C) Perform prediction using the model
y_pred = np.argmax(model.predict(test_token_padded))

# Use the label encoder to find the class
class_prediction = label_encoder.inverse_transform(y_pred.flatten())
class_prediction

#11. Save the important components so that we can deploy them in another application
#(A) Label Encoder
label_encoder_save_path = 'label_encoder_sa.pkl'
with open(label_encoder_save_path, "wb") as f:
    pickle.dump(label_encoder,f)

#(B) Tokenizer
tokenizer_save_path = 'tokenizer_sa.pkl'
with open(tokenizer_save_path, "wb") as f:
    pickle.dump(tokenizer,f)

#(C) Keras model
model_save_path = "nlp_model_sa"
keras.models.save_model(model,model_save_path)

filename = 'ecommerceDataset_normalized_model.h5'
df.to_hdf(filename, key='model',mode='w')