from keras._tf_keras.keras.callbacks import Callback
from keras.src.utils import io_utils
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