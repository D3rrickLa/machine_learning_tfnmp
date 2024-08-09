the focus of v2 over v1 is bringing down val loss
will be trying to augment more data along with hp and adding more layers


# Conv1D(filters=390, kernel_size=5, activation="relu"),
    # MaxPooling1D(pool_size=3),
    # Dropout(0.1),
    # Conv1D(filters=128, kernel_size=3, activation="relu"),
    # MaxPooling1D(pool_size=3),
    # BatchNormalization(),
    
    # GRU(132, return_sequences=True), 
    # Dropout(0.4),
    # Bidirectional(GRU(96)), 
    # Dense(128, activation="relu"),
    # Dropout(0.5),
    # Dense(48, activation="relu"),
    # Dropout(0.5),
    # Activation("tanh"),
    # Dense(len(class_labels), activation='softmax', kernel_regularizer=L1L2(), bias_regularizer=L2(), activity_regularizer=L2())  # Output layer for classification

    # Conv1D(filters=384, kernel_size=3, activation='leaky_relu', kernel_regularizer=L1L2(0.01, 0.01), bias_regularizer=L2()),
    # MaxPooling1D(pool_size=2),
    # BatchNormalization(), 
    # Dense(96, activation="relu6"),
    # BatchNormalization(),
    # Conv1D(filters=128, kernel_size=3, activation='tanh'),
    # MaxPooling1D(pool_size=2),
    # Conv1D(filters=64, kernel_size=3, activation='tanh'),
    # MaxPooling1D(pool_size=2),
    # BatchNormalization(),
    # Bidirectional(GRU(376, return_sequences=True, kernel_regularizer=L2(1.0848960662999816e-05))),
    # Dropout(0.354569645899713),
    # Bidirectional(GRU(385, return_sequences=True, dropout=0.12, recurrent_dropout=0.12)),
    # Bidirectional(GRU(352, return_sequences=True)),
    # LayerNormalization(),
    # Dropout(0.13),
    # GRU(120, kernel_regularizer=L2(6.142117138252424e-05)),
    # BatchNormalization(),
    # Activation('relu6'),
    # Dense(100, activation='tanh'),
    # Dropout(0.5),
    # Dense(len(class_labels), activation='softmax', kernel_regularizer=L1L2(0.0001, 0.0001), bias_regularizer=L2(0.00025))  # Output layer for classification

note: v2 technically worst, but has a good val loss


All DONE - 140 ✅
BED - 142 ✅
DAD - 140 ✅
DRINK - 140 ✅
EAT - 204 ✅
ILU - 138 ✅
MOM - 138 
MORE - 138 
NONE - 134 ✅
POTTY - 140 ✅
THX - 148 ✅

all are at 210, at least 70 are in 1366 x 768, thought different sizing would be good, but switched back
to 1280 x 720