# ORGANIZATION OF THE CODE

Three main folders:

# base

head.py contains the generic definition of the SegmentationHead class used to specify the number of classes used in your dataset
inizialization.py initializes the weights of the part of the architecture not corresponding to the encoder (e.g. which has its weights loaded from Imagenet)
model.py contains the forward pass, in which it iterates throught each attribute of the model such as the encoder, the aspp module, the decoders, etc.
model2.py does the same but it is modified in order to only work with part of the ResNet encoder (until feature map F3 instead of F5)
modules.py contains the definitions of some useful classes or blocks, such as the different activations

# deeplabv3

decoder.py contains the definition of the decoder that takes as input the output of the ASPP module, and its output enters the SegmentationHead
decoder2.py does the same but it is modified in order to only work with part of the ResNet encoder (until feature map F3 instead of F5)
model.py contains the definition of all the attributes of the model that form the architecture (i.e. encoder, aspp, sssr_decoder, sisr_decoder, extra upsampling modules, etc.)
model2.py does the same but it is modified in order to only work with part of the ResNet encoder (until feature map F3 instead of F5)

# encoders

this folder contains the different scripts used to get an specific encoder as a backbone such as ResNet101. The specific encoder is selected in deeplabv3/model.py

