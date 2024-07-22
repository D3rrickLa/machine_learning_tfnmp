model 7 was okay, still had some problem like having to wait holding the gestuer position for it to recognize, also don't know if more gesture would mess with it... 
that is what this model will do
- more gestures
- feature engineer? maybe, atleast the easy ones like velocity and such






training vs val acc 
training acc > val acc = overfitting 

training loss vs val loss
loss < val loss - model too simple/high regularization 
loss = val loss - generalizes well

Underfitting – Validation and training error high

Overfitting – Validation error is high, training error low
            Overfitting would be when acc is higher than val_acc and loss lower than val_loss.

Good fit – Validation error low, slightly higher than the training error

Unknown fit - Validation error low, training error 'high'


dense
- sigmoid: binary 
- softmax: multi class 
- linear/relu: regression




we now gotta try and add more data into this model
will switch over to the dev can branch once we know it can do 6+ gestures
with relative acc


with 5 gestures, we get such bad scores that we can only consider augmentation or more data 