so some good results after trial and error

![alt text](images/image.png)
![alt text](images/image-1.png)

for some reason, the epoch stopped at 23/50, no indication of early stopping happened or my custom callback. still looks good. Werid is that loss graph, what is up with those bumps...


This really is a luck game

Test Loss: 1.3155463933944702
Test Accuracy: 0.5541125535964966

    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(127, kernel_size=5),

    Flatten(),
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("relu6"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))



Test Loss: 0.8309436440467834
Test Accuracy: 0.6666666865348816

    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(127, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    Flatten(),
   
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("relu6"),
    Dropout(0.5),
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))

Test Loss: 0.8621662259101868
Test Accuracy: 0.7792207598686218
    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(127, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    Flatten(),
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))

Test Loss: 0.8409031629562378
Test Accuracy: 0.7186146974563599
    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),

    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),

    GlobalMaxPooling1D(),  # Instead of Flatten
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))

Test Loss: 0.9042438864707947
Test Accuracy: 0.6969696879386902
    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.123),

    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),

    GlobalMaxPooling1D(),  # Instead of Flatten
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))
![alt text](images/image-2.png)
![alt text](images/image-3.png)


Test Loss: 0.8379583358764648
Test Accuracy: 0.7359307408332825
    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.123),
    
    Conv1D(128, kernel_size=3),
    LayerNormalization(),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    

    GlobalMaxPooling1D(),  # Instead of Flatten

    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),

    Dense(96, bias_regularizer=L2(0.00001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.08),

    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))
![alt text](images/image-4.png)
![alt text](images/image-5.png)



NOTE: according to the GPT, the GlobalMaxPooling1D shows the model elikes a moer focused and less param-heavy representation of the temporal data

Test Loss: 0.7620430588722229
Test Accuracy: 0.761904776096344
 InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.123),

    # Dense(100, kernel_regularizer=L2(0.000001)),
    # Activation("tanh"),
    
    Conv1D(128, kernel_size=3),
    LayerNormalization(),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),

    Bidirectional(GRU(128, return_sequences=True)),   
    LayerNormalization(),
    Activation("tanh"),
    BatchNormalization(),
    Dropout(0.01),

    # Bidirectional(GRU(128, return_sequences=True)),   
    # LayerNormalization(),

    # GRU(96, return_sequences=True),
    # BatchNormalization(),

    GlobalMaxPooling1D(),  # Instead of Flatten
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),

    Dense(96, bias_regularizer=L2(0.00001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.08),

    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))
![alt text](images/image-6.png)
![alt text](images/image-7.png)

Test Loss: 0.8189589977264404
Test Accuracy: 0.7316017150878906
NOTE: this isn't really an improvement, but the matrix is much better, the potty is acutally being detected
    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.123),

    # Dense(100, kernel_regularizer=L2(0.000001)),
    # Activation("tanh"),
    
    Conv1D(128, kernel_size=3),
    LayerNormalization(),
    Activation("relu6"),
    MaxPooling1D(3),
    BatchNormalization(),

    Bidirectional(GRU(128, return_sequences=True)),   
    LayerNormalization(),
    Activation("tanh"),
    BatchNormalization(),
    Dropout(0.1),

    # Bidirectional(GRU(128, return_sequences=True)),   
    # LayerNormalization(),

    # GRU(96, return_sequences=True),
    # BatchNormalization(),

    GlobalMaxPooling1D(),  # Instead of Flatten
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),

    Dense(96, bias_regularizer=L2(0.00001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.08),

    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))
![alt text](images/image-8.png)
![alt text](images/image-9.png)



Test Loss: 0.71469646692276
Test Accuracy: 0.761904776096344
    InputLayer(shape=(sequence_length, 1666)),
    
    Conv1D(128, kernel_size=3),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.153),

    # Dense(100, kernel_regularizer=L2(0.000001)),
    # Activation("tanh"),
    
    Conv1D(128, kernel_size=3),
    LayerNormalization(),
    Activation("relu6"),
    MaxPooling1D(3),
    BatchNormalization(),

    Bidirectional(GRU(128, return_sequences=True)),   
    LayerNormalization(),
    Activation("tanh"),
    BatchNormalization(),
    Dropout(0.125),

    # Bidirectional(GRU(128, return_sequences=True)),   
    # LayerNormalization(),

    # GRU(96, return_sequences=True),
    # BatchNormalization(),

    GlobalMaxPooling1D(),  # Instead of Flatten
    
    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(128, kernel_regularizer=L2(0.0001)),
    BatchNormalization(),

    Dense(96, bias_regularizer=L2(0.00001)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.08),

    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))

![alt text](images/image-10.png)
![alt text](images/image-11.png)


model quatre v3 has the things I like, not as good as the previous best, but the 
Test Loss: 0.7827021479606628
Test Accuracy: 0.7402597665786743
    Conv1D(128, kernel_size=3),
    Activation("relu6"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.153),

    Conv1D(128, kernel_size=3),
    LayerNormalization(),
    Activation("relu6"),
    MaxPooling1D(3),
    BatchNormalization(),

    Bidirectional(GRU(128, return_sequences=True)),   
    LayerNormalization(),
    Activation("tanh"),
    BatchNormalization(),
    Dropout(0.135),

    # Bidirectional(GRU(128, return_sequences=True)),   
    # LayerNormalization(),

    # GRU(96, return_sequences=True),
    # BatchNormalization(),

    GlobalMaxPooling1D(),  # Instead of Flatten

    Dense(128, kernel_regularizer=L2(1e-4)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(128, kernel_regularizer=L2(1e-4)),
    BatchNormalization(),

    Dense(96, bias_regularizer=L2(1e-5)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.08),

    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-7))


![alt text](images/image-12.png)
![alt text](images/image-13.png)

Test Loss: 1.034582495689392
Test Accuracy: 0.6190476417541504
going forward, we have doubled the training data, different model because the current doesn't work with that much data...


    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(256, kernel_size=3),
    LayerNormalization(),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.153),

    Conv1D(128, kernel_size=3),
    LayerNormalization(),
    Activation("relu"),
    MaxPooling1D(3),
    BatchNormalization(),
    Dropout(0.035),

    Bidirectional(GRU(128, return_sequences=True)),   
    LayerNormalization(),
    Activation("tanh"),
    BatchNormalization(),
    Dropout(0.135),

    Bidirectional(GRU(128, return_sequences=True)),   
    LayerNormalization(),
    Activation("tanh"),
    BatchNormalization(),
    Dropout(0.135),

    GlobalMaxPooling1D(),

    Dense(256, kernel_regularizer=L2(1e-4)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(256, kernel_regularizer=L2(1e-4)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.55),

    Dense(128),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.08),

    Dense(64, kernel_regularizer=L2(1e-3)),
    BatchNormalization(),
    Activation("tanh"),
    Dropout(0.2),
    
    Activation("tanh"),
    Dropout(0.5),
    Dense(len(class_labels), activation="softmax", kernel_regularizer=L2(1e-7))

so this model got the following:
Test Loss: 1.135124921798706
Test Accuracy: 0.6796537041664124
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Flatten(),

    Dense(64, kernel_regularizer=L2(1e-5)),
   
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-6),  bias_regularizer=L2(1e-5))
![alt text](images/image-14.png)
![alt text](images/image-15.png)
so like the graph is good, but that loss. well the graph is good in only some parts



yooooooo
Test Loss: 0.5579075217247009
Test Accuracy: 0.8311688303947449
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    GRU(126, kernel_regularizer=L2(1e-5)),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.5),

    Dense(64, kernel_regularizer=L2(1e-5)),
   
    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-6),  bias_regularizer=L2(1e-5))

![alt text](images/image-16.png)
![alt text](images/image-17.png)

really good... almost too good, I don't trust it. Will do a Kfold style of validation. TODO: Might try an ensemble 

Test Loss: 0.4607835114002228
Test Accuracy: 0.8787878751754761
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(160, 3, activation="relu6"),
    BatchNormalization(),

    GRU(128, kernel_regularizer=L2(1e-5)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),


    Dense(64, kernel_regularizer=L2(1e-5)),
    Activation("tanh"),
    Dropout(0.165),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-6),  bias_regularizer=L2(1e-5))

![alt text](images/image-18.png)
![alt text](images/image-19.png)

this model is really good, potty finally got some points. None did take a hit though
this is with the model_5trois.keras file

Test Loss: 1.0815975666046143
Test Accuracy: 0.9090909361839294
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(164, 4, activation="relu6", kernel_regularizer=L2(1e-2),),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.065),

    GRU(128, kernel_regularizer=L2(1e-5)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(64, kernel_regularizer=L2(1e-4)),
    Activation("tanh"),
    Dropout(0.165),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-6),  bias_regularizer=L2(1e-5))

![alt text](images/image-20.png)
![alt text](images/image-21.png)

model while loss is bad, has a better confusion matrix then the others. save on the 5_quatre keras model

Test Loss: 0.998706579208374
Test Accuracy: 0.9307359457015991
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(170, 4, activation="relu6", kernel_regularizer=L2(1e-2)),
    BatchNormalization(),
    MaxPooling1D(3),
    Dropout(0.055),

    GRU(128, kernel_regularizer=L2(1e-5)),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    Dense(90, kernel_regularizer=L2(1e-4)),
    Activation("tanh"),
    Dropout(0.165),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-6),  bias_regularizer=L2(1e-5))
![alt text](images/image-22.png)
![alt text](images/image-23.png)

model performed slightly better, could honestly be margin or error but it is better
NOTE: after doing some reading, and GPT work. there's this thing called natural logarithm (ln) and the closer that number our loss is, means the we are essentially performing a random guess. in our case since we have 11 classes, the ln is ~2.4. Our loss here is closer to that 2.4 at 1

Test Loss: 0.5885689854621887
Test Accuracy: 0.9004328846931458
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(170, 3, activation="tanh", kernel_regularizer=L2(1e-3)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(340, 3, activation="tanh"),
    MaxPooling1D(2),
    Dropout(0.055),

    GRU(128, kernel_regularizer=L2(1e-5), return_sequences=True),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.5),

    GRU(64),
    BatchNormalization(),

    Dense(64, kernel_regularizer=L2(1e-4)),
    Activation("tanh"),
    Dropout(0.165),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L2(1e-6),  bias_regularizer=L2(1e-5))
![alt text](images/image-24.png)
![alt text](images/image-25.png)

This model is substantially better than the last. While we lose some accuracy with the problem gestures, that loss is way better

Test Loss: 0.7382153868675232
Test Accuracy: 0.887445867061615
 InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(170, 3, activation="relu6", kernel_regularizer=L1L2(1e-5, 1e-3)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(340, 3, activation="tanh"),
    MaxPooling1D(2),
    Dropout(0.055),

    Conv1D(680, 3, activation="tanh"),
    Dropout(0.1),

    Bidirectional(GRU(128, kernel_regularizer=L2(1e-4))),
    BatchNormalization(),
    Activation("relu6"),
    Dropout(0.5),

    Dense(96),
    BatchNormalization(),

    Dense(64, kernel_regularizer=L2(1e-4)),
    Activation("tanh"),
    Dropout(0.2),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L1L2(1e-5, 1e-6),  bias_regularizer=L2(1e-5))
![alt text](images/image-26.png)
![alt text](images/image-27.png)

This model has potential, but like it's the learning rate that is causing a problem. gets too low - local minima 

Test Loss: 0.6565546989440918
Test Accuracy: 0.9090909361839294
model = Sequential([
    # NOTE: if we where to just run this as is, we will get an error because of our Y datasets
    # They are a 2D shape, not the 1D it is expecting
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(170, 3, activation="leaky_relu", kernel_regularizer=L1L2(1e-7, 1e-3)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(340, 3, activation="leaky_relu"),
    MaxPooling1D(2),
    Dropout(0.1),

    Conv1D(680, 3, activation="tanh"),

    Bidirectional(GRU(128, kernel_regularizer=L2(1e-4))),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.35),

    Dense(96),
    BatchNormalization(),

    Dense(64, kernel_regularizer=L2(1e-4)),
    Activation("tanh"),
    Dropout(0.25),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L1L2(1e-7, 1e-6),  bias_regularizer=L2(1e-6))
])

model.compile(optimizer=AdamW(learning_rate=1.0e-4, weight_decay=1e-5, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
![alt text](images/image-28.png)
![alt text](images/image-29.png)
model 8 btw


model 9
Test Loss: 0.6769238710403442
Test Accuracy: 0.9090909361839294
model = Sequential([
    # NOTE: if we where to just run this as is, we will get an error because of our Y datasets
    # They are a 2D shape, not the 1D it is expecting
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),
    
    Conv1D(170, 3, activation="leaky_relu", kernel_regularizer=L1L2(1e-7, 1e-3)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(340, 3, activation="leaky_relu"),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(340, 3, activation="tanh"),

    Bidirectional(GRU(128, kernel_regularizer=L2(1e-4))),
    BatchNormalization(),
    Activation("leaky_relu"),
    Dropout(0.35),

    Dense(96),
    BatchNormalization(),

    Dense(64, kernel_regularizer=L2(1e-4)),
    Activation("tanh"),
    Dropout(0.25),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L1L2(1e-7, 1e-6),  bias_regularizer=L2(1e-6))
])

model.compile(optimizer=AdamW(learning_rate=1.0e-4, weight_decay=1e-5, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
![alt text](images/image-30.png)
![alt text](images/image-31.png)

Test Loss: 0.5824358463287354
Test Accuracy: 0.9090909361839294
model = Sequential([
    # NOTE: if we where to just run this as is, we will get an error because of our Y datasets
    # They are a 2D shape, not the 1D it is expecting
    InputLayer(shape=(sequence_length, X_train_sequences.shape[2])),

    Conv1D(64, 3, kernel_regularizer=L2(1e-3)),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),

    Dense(90, kernel_regularizer=L2(1e-3)),
    BatchNormalization(),
    Activation("relu"),

    Dense(len(class_labels), activation='softmax', kernel_regularizer=L1L2(1e-5, 1e-4))
   

])

model.compile(optimizer=AdamW(learning_rate=1.0e-4, weight_decay=1e-5, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
![alt text](images/image-32.png)
![alt text](images/image-33.png)


so this is probably the final stable model, 11 v3 - didn't save it here, but basically the same as above but with a lower test loss. The high overfitting can lead to non-optimally performance, but it has high acc so it should be fine - monitor it still (problems seems to be from the None side which is fine)