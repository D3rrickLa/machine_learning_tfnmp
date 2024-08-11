so some good results after trial and error

![alt text](image.png)
![alt text](image-1.png)

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
![alt text](image-2.png)
![alt text](image-3.png)


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
![alt text](image-4.png)
![alt text](image-5.png)