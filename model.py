import tensorflow.keras.layers as _layers 
from tensorflow.keras.models import Model as _Model


def encoding(layer, n_conv, drop = False):
    
    conv = _layers.Conv2D(n_conv, 3, activation = 'relu', padding = 'same')(layer)
    conv = _layers.Conv2D(n_conv, 3, activation = 'relu', padding = 'same')(conv)
    
    if drop == False:
        
        pool = _layers.MaxPooling2D(pool_size=(2, 2))(conv)
    
    else:
        
        conv = _layers.Dropout(0.5)(conv)
        pool = _layers.MaxPooling2D(pool_size=(2, 2))(conv)
        
    return pool, conv

def decoding(layer, layer_encoder, n_conv):
    
    upsa = _layers.UpSampling2D(size = (2,2))(layer)
    conv_upsa = _layers.Conv2D(n_conv, 2, activation = 'relu', padding = 'same')(upsa)
    conc = _layers.concatenate([layer_encoder,conv_upsa], axis = 3)
    conv = _layers.Conv2D(n_conv, 3, activation = 'relu', padding = 'same')(conc)
    conv = _layers.Conv2D(n_conv, 3, activation = 'relu', padding = 'same')(conv)
    
    return conv

def unet():
    
    inputs = _layers.Input(shape=(128, 128, 1))
    
    layer1, skip_layer1 = encoding(inputs, 64)
    layer2, skip_layer2 = encoding(layer1, 128)
    layer3, skip_layer3 = encoding(layer2, 256)
    layer4, skip_layer4 = encoding(layer3, 512, drop = True)
    
    bottleneck = _layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(layer4)
    bottleneck = _layers.Conv2D(1024, 3, activation = 'relu', padding = 'same')(bottleneck)
    bottleneck = _layers.Dropout(0.5)(bottleneck)

    layer6 = decoding(bottleneck, skip_layer4, 512)
    layer7 = decoding(layer6, skip_layer3, 256)
    layer8 = decoding(layer7, skip_layer2, 128)
    layer9 = decoding(layer8, skip_layer1, 64)
    
    conv_out = _layers.Conv2D(2, 3, activation = 'relu', padding = 'same')(layer9)
    conv_out = _layers.Conv2D(1, 1, activation = 'tanh')(conv_out)
    
    model = _Model(inputs,conv_out)

    return model