from keras._tf_keras.keras.callbacks import Callback
from keras.src.utils import io_utils
import tensorflow as tf

class CustomEarlyStopping(Callback):
    def __init__(self, patience=0, threshold=0.15):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience 
        self.threshold = threshold 
        self.wait = 0 
        self.best_weights = None 
    
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if val_loss > loss * (1 + self.threshold):
            self.wait +=1 
            if self.wait >= self.patience:
                self.model.stop_training = True 
                if self.best_weights is not None: 
                    self.model.set_weights(self.best_weights)
                    print(" - Loss and Val Loss are too far apart. Restoring model weights")

        else:
            self.wait = 0 
            self.best_weights = self.model.get_weights()

class BoostLearningRateOnPlateau(Callback):
    def __init__(self, monitor='loss', factor=2, patience=10, min_lr=1e-6, verbose=1):
        super(BoostLearningRateOnPlateau, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.wait = 0
        self.best = float('inf')
        self.lr_boosted = False
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
            self.lr_boosted = False
        else:
            self.wait += 1
            if self.wait >= self.patience and not self.lr_boosted:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose:
                        print(f"\nEpoch {epoch+1}: Boosting learning rate to {new_lr}.")
                    self.lr_boosted = True
                    self.wait = 0