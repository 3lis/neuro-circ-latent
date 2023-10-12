"""
#####################################################################################################################

    Alice   2022

    Definition of the neural network architectures

#####################################################################################################################
"""

SEED = 1
import      os
import      numpy               as np
import      random              as rn
import      tensorflow          as tf
from        keras               import backend  as K
from        tensorflow.keras    import models, layers, utils, optimizers, losses, applications
from        distutils.version   import LooseVersion
os.environ[ 'PYTHONHASHSEED' ] = str( SEED )
np.random.seed( SEED )
rn.seed( SEED )
# -------------------------------------------------------------------------------------------------------------------

import      data_gen
from        print_msg       import print_err, print_wrn


#####################################################################################################################
#
#   GLOBALs
#
#####################################################################################################################

layer_code      = {                                     # one of the accepted type of layers
        'CONV':         'C',
        'DCNV':         'T',
        'DNSE':         'D',
        'LSTM':         'L',
        'GRUU':         'G',
        'SRNN':         'S',
        'FLAT':         'F',
        'POOL':         'P',
        'RSHP':         'R',
        'STOP':         '-'
}

optimiz_code    = {                                     # one of the accepted keras.optimizers.Optimizer
        'ADAM':         optimizers.Adam,
        'ADAGRAD':      optimizers.Adagrad,
        'SDG':          optimizers.SGD,
        'RMS':          optimizers.RMSprop
}

loss_code       = {                                     # one of the accepted losses functions
        'MSE':          losses.mean_squared_error,
        'BXE':          losses.binary_crossentropy,
        'W-BXE':        K.binary_crossentropy,          # case of loss weighted by distance matrix
        'CXE':          losses.categorical_crossentropy
}

preprocess_code = {                                     # one of the accepted preprocessing functions
        'ResNet50':         applications.resnet.preprocess_input,
        'InceptionV3':      applications.inception_v3.preprocess_input,
        'EfficientNetB0':   applications.efficientnet.preprocess_input,
        'EfficientNetB4':   applications.efficientnet.preprocess_input,
        'EfficientNetB5':   applications.efficientnet.preprocess_input,
        'EfficientNetB6':   applications.efficientnet.preprocess_input
#       'EfficientNetV2B0': applications.efficientnet_v2.preprocess_input
}


#####################################################################################################################
#
#   Classes
#
#   - Encoder
#   - Decoder
#   - EncoderShiftDecoder
#
#####################################################################################################################

# ===================================================================================================================
#
#   Class for the generation of an encoder network
#
# ===================================================================================================================

class Encoder( object ):

    def __init__( self, kwargs, summary=False ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                    [str] path where to save plot if you want a summary+plot of model, False otherwise
        
        Expected parameters in kwargs:
            arch_layout:            [str] code describing the order of layers in the model
            input_size:             [list] height, width, channels
            net_indx:               [int] a unique index identifying the network

            conv_kernel_num:        [list of int] number of kernels for each convolution
            conv_kernel_size:       [list of int] (square) size of kernels for each convolution
            conv_strides:           [list of int] stride for each convolution
            conv_padding:           [list of str] padding (same/valid) for each convolution
            conv_activation:        [list of str] activation function for each convolution
            conv_train:             [list of bool] False to lock training of each convolution

            pool_size:              [list of int] pooling size for each MaxPooling
            
            enc_dnse_size:          [list of int] size of each dense layer
            enc_dnse_dropout:       [list of int] dropout of each dense layer
            enc_dnse_activation:    [list of str] activation function for each dense layer
            enc_dnse_train:         [list of bool] False to lock training of each dense layer
        --------------------------------------------------------------------------------------------------------- """
        for key, value in kwargs.items():
            setattr( self, key, value )

        # total num of convolutions, pooling and dense layers
        self.n_conv             = self.arch_layout.count( layer_code[ 'CONV' ] )
        self.n_pool             = self.arch_layout.count( layer_code[ 'POOL' ] )
        self.n_dnse             = self.arch_layout.count( layer_code[ 'DNSE' ] )

        # the last 3D shape before flattening, filled by _define_layers()
        self.last_3d_shape      = None

        # the keras.layers.Input object, filled by define_model()
        self.input_layer        = None

        # the layout description must contain meaningful codes
        assert set( self.arch_layout ).issubset( layer_code.values() )

        # only a single flatten layer is accepted in the architecture
        assert self.arch_layout.count( layer_code[ 'FLAT' ] ) == 1

        assert self.n_dnse == len( self.enc_dnse_size ) == len( self.enc_dnse_dropout ) == \
                len( self.enc_dnse_activation ) == len( self.enc_dnse_train )
        assert self.n_conv == len( self.conv_kernel_num ) == len( self.conv_kernel_size ) == len( self.conv_strides ) == \
                len( self.conv_padding ) == len( self.conv_activation ) == len( self.conv_train )
        assert self.enc_dnse_size[ -1 ] == data_gen.n_rot, "Error: mismatch between latent dimension and rotations"

        # create the network
        self.model_name     = 'encoder_{}'.format( self.net_indx )
        self.model          = self.define_model()
        if summary:         model_summary( self.model, fname=os.path.join( summary, self.model_name ) )



    def define_model( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the encoder model

        summary:        [bool] if True produce a summary of the model

        return:         [keras.models.Model] decoder model
        --------------------------------------------------------------------------------------------------------- """
        self.input_layer    = layers.Input( shape=self.input_size )
        model               = models.Model( 
                    inputs      = self.input_layer,
                    outputs     = self._define_layers( self.input_layer ),
                    name        = self.model_name
        )

        return model



    def _define_layers( self, x ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the network of layers

        x:              [tf.Tensor] input of the layers

        return:         [tf.Tensor] output of the layers
        --------------------------------------------------------------------------------------------------------- """
        i_conv = i_pool = i_dnse  = 0                             # to keep count

        for layer in self.arch_layout:

            # convolutional layer
            if layer == layer_code[ 'CONV' ]:
                x       = self._conv2D( x, i_conv )
                i_conv  += 1

            # pooling layer
            elif layer == layer_code[ 'POOL' ]:
                x       = self._maxpool2D( x, i_pool )
                i_pool  += 1

            # dense layer
            elif layer == layer_code[ 'DNSE' ]:
                x       = self._dense( x, i_dnse )
                i_dnse  += 1

            # flat layer
            elif layer == layer_code[ 'FLAT' ]:
                self.last_3d_shape      = K.int_shape( x )[ 1: ]        # save the last 3D shape before flattening
                x       = layers.Flatten( name='flat_{}'.format( self.net_indx ) )( x )

            else:
                print_err( "Layer code '{}' not valid".format( layer ) )
                
        return x



    def _conv2D( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Conv2D layer, using the parameters associated to the index passed as argument

            NOTE be aware of the difference between kernel_regularizer and activity_regularizer
            NOTE be aware there are biases also in convolutions

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        return layers.Conv2D(
                    self.conv_kernel_num[ indx ],                           # number of filters
                    kernel_size         = self.conv_kernel_size[ indx ],    # size of window
                    strides             = self.conv_strides[ indx ],        # stride (window shift)
                    padding             = self.conv_padding[ indx ],        # zero-padding around the image
                    activation          = self.conv_activation[ indx ],     # activation function
                    #kernel_initializer = self.conv_initializer,            # kernel initializer
                    #kernel_regularizer = self.conv_regularizer,            # kernel regularizer
                    #use_bias           = True,                             # convolutional biases
                    trainable           = self.conv_train[ indx ],
                    name                = 'conv_{}_{}'.format( self.net_indx, indx )
        )( x )



    def _maxpool2D( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a MaxPooling2D layer, using the parameters associated to the index passed as argument

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        return layers.MaxPooling2D(                          
                    pool_size       = self.pool_size[ indx ],               # pooling size
                    padding         = self.conv_padding[ indx ],            # zero-padding around the image
                    name            = 'pool_{}_{}'.format( self.net_indx, indx )
        )( x )



    def _dense( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Dense layer, using the parameters associated to the index passed as argument

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        x   = layers.Dense(                          
                    self.enc_dnse_size[ indx ],                             # dimensionality of the output
                    activation      = self.enc_dnse_activation[ indx ],     # activation function
                    trainable       = self.enc_dnse_train[ indx ],
                    name            = 'dnse_{}_{}'.format( self.net_indx, indx )
        )( x )

        if self.enc_dnse_dropout[ indx ] > 0:                               # dropout
            x   = layers.Dropout( self.enc_dnse_dropout[ indx ] )( x )

        return x



# ===================================================================================================================
#
#   Class for the generation of a decoder network
#
# ===================================================================================================================

class Decoder( object ):

    def __init__( self, kwargs, summary=False ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                    [str] path where to save plot if you want a summary+plot of model, False otherwise

        Expected parameters in kwargs:
            arch_layout:            [str] code describing the order of layers in the model
            input_size:             [list] height, width, channels
            target_size:            [list] height, width, channels
            net_indx:               [int] a unique index identifying the network
            first_3d_shape:         [list] the first 3D shape after reshaping (height, width, channels)

            dcnv_kernel_num:        [list of int] number of kernels for each deconvolution
            dcnv_kernel_size:       [list of int] (square) size of kernels for each deconvolution
            dcnv_strides:           [list of int] stride for each deconvolution
            dcnv_padding:           [list of str] padding (same/valid) for each deconvolution
            dcnv_activation:        [list of str] activation function for each deconvolution
            dcnv_train:             [list of bool] False to lock training of each deconvolution

            dec_dnse_size:          [list of int] size of each dense layer
            dec_dnse_dropout:       [list of int] dropout of each dense layer
            dec_dnse_activation:    [list of str] activation function for each dense layer
            dec_dnse_train:         [list of bool] False to lock training of each dense layer
        --------------------------------------------------------------------------------------------------------- """
        for key, value in kwargs.items():
            setattr( self, key, value )

        # total num of deconvolutions and dense layers
        self.n_dcnv     = self.arch_layout.count( layer_code[ 'DCNV' ] )
        self.n_dnse     = self.arch_layout.count( layer_code[ 'DNSE' ] )

        # the layout description must contain meaningful codes
        assert set( self.arch_layout ).issubset( layer_code.values() )

        # only a single reshape layer is accepted in the architecture
        assert self.arch_layout.count( layer_code[ 'RSHP' ] ) == 1

        assert self.n_dnse == len( self.dec_dnse_size ) == len( self.dec_dnse_activation ) == \
                len( self.dec_dnse_dropout ) == len( self.dec_dnse_train )
        assert self.n_dcnv == len( self.dcnv_kernel_num ) == len( self.dcnv_kernel_size ) == len( self.dcnv_strides ) == \
                len( self.dcnv_padding ) == len( self.dcnv_activation ) == len( self.dcnv_train )

        # the last deconvolution should match with the number of channels of the target
        assert self.dcnv_kernel_num[ -1 ] == self.target_size[ -1 ]

        # create the network
        self.model_name     = 'decoder_{}'.format( self.net_indx )
        self.model          = self.define_model()
        if summary:         model_summary( self.model, fname=os.path.join( summary, self.model_name ) )



    def define_model( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the decoder model

        summary:        [bool] if True produce a summary of the model

        return:         [keras.models.Model] decoder model
        --------------------------------------------------------------------------------------------------------- """
        x       = layers.Input( shape=self.input_size )
        model   = models.Model( 
                    inputs      = x,
                    outputs     = self._define_layers( x ),
                    name        = self.model_name
        )

        return model



    def _define_layers( self, x ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the network of layers

        x:              [tf.Tensor] input of the layers

        return:         [tf.Tensor] output of the layers
        --------------------------------------------------------------------------------------------------------- """
        i_dcnv, i_dnse          = 2 * [ 0 ]                             # to keep count

        for layer in self.arch_layout:

            # deconvolutional layer
            if layer == layer_code[ 'DCNV' ]:
                x       = self._deconv2D( x, i_dcnv )
                i_dcnv  += 1

            # dense layer
            elif layer == layer_code[ 'DNSE' ]:
                x       = self._dense( x, i_dnse )
                i_dnse  += 1

            # reshape layer
            elif layer == layer_code[ 'RSHP' ]:
                ts      = self.first_3d_shape
                x       = layers.Reshape( target_shape=ts, name='rshp_{}'.format( self.net_indx ) )( x )

            else:
                print_err( "Layer code '{}' not valid".format( layer ) )
                
        return x



    def _deconv2D( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Conv2DTranspose layer, using the parameters associated to the index passed as argument

            NOTE be aware of the difference between kernel_regularizer and activity_regularizer
            NOTE be aware there are biases also in deconvolutions

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        return layers.Conv2DTranspose(
                    self.dcnv_kernel_num[ indx ],                           # number of filters
                    kernel_size         = self.dcnv_kernel_size[ indx ],    # size of window
                    strides             = self.dcnv_strides[ indx ],        # stride (window shift)
                    padding             = self.dcnv_padding[ indx ],        # zero-padding around the image
                    activation          = self.dcnv_activation[ indx ],     # activation function
                    #kernel_initializer = self.dcnv_initializer,            # kernel initializer
                    #kernel_regularizer = self.dcnv_regularizer,            # kernel regularizer
                    #use_bias           = True,                             # deconvolutional biases
                    trainable           = self.dcnv_train[ indx ],
                    name                = 'dcnv_{}_{}'.format( self.net_indx, indx )
        )( x )



    def _dense( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Dense layer, using the parameters associated to the index passed as argument

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        x   = layers.Dense(                          
                    self.dec_dnse_size[ indx ],                             # dimensionality of the output
                    activation      = self.dec_dnse_activation[ indx ],     # activation function
                    trainable       = self.dec_dnse_train[ indx ],
                    name            = 'dnse_{}_{}'.format( self.net_indx, indx )
        )( x )

        if self.dec_dnse_dropout[ indx ] > 0:                               # dropout
            x   = layers.Dropout( self.dec_dnse_dropout[ indx ] )( x )

        return x

 

# ===================================================================================================================
#
#   Class for the generation of an encoding network followed by latent, shift, and a shared decoder for two outputs
#   for the first output the decoder takes as input the latent, and reconstruct the original image (autoencoder)
#   for the second output the decoder takes as input the shifted latend, and reconstruct a rotation of the image
#
# ===================================================================================================================

class EncoderShiftDecoder( object ):

    def __init__( self, arch_kwargs, other_kwargs, summary=False, model_name='encshiftdec' ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                    [str] path where to save plot if you want a summary+plot of model, False otherwise
        model_name:                 [str] name of the model

        Expected in other_kwargs:   
            enc_kwargs:             [dict] agruments to build encoder
            dec_kwargs:             [dict] agruments to build decoder

        Expected parameters in arch_kwargs:
            arch_layout:            [str] code describing the order of layers in the model
            optimiz:                [str] code of the optimizer
            loss:                   [str] code of the loss function
            lrate:                  [float] learning rate
        --------------------------------------------------------------------------------------------------------- """
        self.model_name         = model_name
        enc_kwargs, dec_kwargs  = other_kwargs

        for key, value in arch_kwargs.items():
            setattr( self, key, value )

        assert layer_code[ 'STOP' ] in self.arch_layout

        channels            = 3 if data_gen.color   else 1
        self.input_size     = ( *data_gen.isize, channels )         # should be the image size, plus channel
        self.target_size    = self.input_size                       # target is the same sized image
        self.shift_size     = ( data_gen.n_rot, data_gen.n_rot )    # the input shift square matrix

        self.loss_func      = [ loss_code[ self.loss ], loss_code[ self.loss ] ]

        self.model          = self.define_model( enc_kwargs, dec_kwargs, summary=summary )
        if summary:         model_summary( self.model, fname=os.path.join( summary, self.model_name ) )



    def define_model( self, enc_kwargs, dec_kwargs, summary=False ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the encoder-decoder model

        enc_kwargs:     [dict] parameters for Encoder
        dec_kwargs:     [dict] parameters for Decoder
        summary:        [str] path where to save plot if you want a summary+plot of model, False otherwise

        return:         [keras.models.Model] model
        --------------------------------------------------------------------------------------------------------- """
        enc_kwargs[ 'net_indx' ]        = 1
        dec_kwargs[ 'net_indx' ]        = 2
        enc_kwargs[ 'arch_layout' ]     = self.arch_layout.split( layer_code[ 'STOP' ] )[ 0 ]
        dec_kwargs[ 'arch_layout' ]     = self.arch_layout.split( layer_code[ 'STOP' ] )[ -1 ]
        enc_kwargs[ 'input_size' ]      = self.input_size
        dec_kwargs[ 'input_size' ]      = ( enc_kwargs[ 'enc_dnse_size' ][ -1 ], )
        dec_kwargs[ 'target_size' ]     = self.target_size

        # create auxiliary input for the shift matrix
        shift   = layers.Input( shape=self.shift_size )

        # create Encoder and Decoder objects
        self.encoder                    = Encoder( enc_kwargs, summary=summary )
        self.decoder                    = Decoder( dec_kwargs, summary=summary )

        # from Encoder to latent space
        latent1 = self.encoder.model( self.encoder.input_layer )

        # shifted latent space for rotation
        latent2 = layers.Dot( axes=(1,2) )( [ latent1, shift ] )

        # autoencoded original image
        y1      = self.decoder.model( latent1 )

        # reconstruction of rotated image, with shared decoder weights
        y2      = self.decoder.model( latent2 )

        model   = models.Model( 
                    inputs      = [ self.encoder.input_layer, shift ],
                    outputs     = [ y1, y2 ],
                    name        = self.model_name
        )

        return model

 

# ===================================================================================================================
#
#   Class for the generation of a baseline model made with a standard CNN architecture
#   the same CNN takse as input two images, the two CNN outputs are concatenated, and the
#   output layer is a denze softmax vector with the number of possible rotations as dimension
#   the model name is in enc_kwargs[ 'baseline' ] (in other_kwargs) and should be one of the
#   available Keras applications models, see list in
#   https://keras.io/api/applications/
#
# ===================================================================================================================

class BaselineCNN( object ):

    def __init__( self, arch_kwargs, other_kwargs, summary=False, ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                    [str] path where to save plot if you want a summary+plot of model, False otherwise

        Expected in other_kwargs:   
            enc_kwargs:             [dict] arguments to build encoder
            dec_kwargs:             [dict] not used

        Expected parameters in arch_kwargs:
            arch_layout:            [str] code describing the order of layers in the model
            optimiz:                [str] code of the optimizer
            loss:                   [str] code of the loss function
            lrate:                  [float] learning rate
        --------------------------------------------------------------------------------------------------------- """
        enc_kwargs, dec_kwargs  = other_kwargs
        self.baseline           = enc_kwargs[ 'baseline' ]
        self.model_name         = 'base_' + self.baseline

        for key, value in arch_kwargs.items():
            setattr( self, key, value )

        assert data_gen.color                                       # baseline CNN expect color images

        self.input_size     = ( *data_gen.isize, 3 )                # should be the image size, plus channel
#       self.target_size    = data_gen.n_rot                        # target is the number of rotations
        self.target_size    = 2                                     # target is [ sin, cos ]

        self.loss_func      = loss_code[ self.loss ]

        self.model          = self.define_model( enc_kwargs, summary=summary )
        if summary:         model_summary( self.model, fname=os.path.join( summary, self.model_name ) )


    def preprocess( self, x ):
        """ ---------------------------------------------------------------------------------------------------------
        preprocess images in the way appropriate for the baseline used

        baseline:       [str] name of the Keras applications model to use
        x               [keras.engine.keras_tensor.KerasTensor] input image

        return:         [keras.engine.keras_tensor.KerasTensor] output image
        --------------------------------------------------------------------------------------------------------- """

        # the Keras preprocessing function, NOTE that it expect values in range 0-255
        pre     = preprocess_code[ self.baseline ]
        scale   = layers.Rescaling( 255. )
        x       = scale( x )
        return pre( x )


    def define_cnn( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the CNN for the baseline model

        return:         [keras.models.Model] model
        --------------------------------------------------------------------------------------------------------- """

        # the Keras model
        cnn     = eval( "applications." + self.baseline )

        # create Encoder and Decoder objects
        model   = cnn(
                include_top = False,
                weights     = "imagenet",
                input_shape = self.input_size
        )
        model.trainable     = False                             # freeze the CNN weights

        return model


    def define_model( self, enc_kwargs, summary=False ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the baseline model

        enc_kwargs:     [dict] parameters for Encoder (NOTE: only enc_dnse_size, and its last value, is used)
        summary:        [str] path where to save plot if you want a summary+plot of model, False otherwise

        return:         [keras.models.Model] model
        --------------------------------------------------------------------------------------------------------- """

        # create inputs
        img1        = layers.Input( shape=self.input_size )
        img2        = layers.Input( shape=self.input_size )
        i1          = self.preprocess( img1 )
        i2          = self.preprocess( img2 )

        # create Encoder and Decoder objects
        cnn         = self.define_cnn()
        feat1       = cnn( i1 )
        feat2       = cnn( i2 )
        feat1       = layers.Flatten( name='flat_1' )( feat1 )
        feat2       = layers.Flatten( name='flat_2' )( feat2 )
        features    = layers.Concatenate( axis=1 )( [ feat1, feat2 ] )
        # only one additional dense between the output of the CNN, and the last softmax dense, is allowed
        if "enc_dnse_size" in enc_kwargs.keys() and len( enc_kwargs[ "enc_dnse_size" ] ) > 0:
            features    = layers.Dense(                          
                        enc_kwargs[ "enc_dnse_size" ][ -1 ],    # dimensionality of the output
                        activation      = 'relu',               # activation function
                        name            = 'dnse_butlast'
            )( features )
#       y           = layers.Dense( self.target_size, activation='softmax', name='final' )( features )
        y           = layers.Dense( self.target_size, activation='tanh', name='final' )( features )     # for [sin, cos ]

        model   = models.Model( 
                    inputs      = [ img1, img2 ],
                    outputs     = y,
                    name        = self.model_name
        )

        return model




#####################################################################################################################
#
#   GLOBALs and other FUNCTIONs
#
#   - model_summary
#   - create_model
#
#####################################################################################################################


def model_summary( model, fname ):
    """ -------------------------------------------------------------------------------------------------------------
    Print a summary of the model, and plot a graph of the model

    model:          [keras.engine.training.Model]
    fname:          [str] name of the output image with path but without extension
    ------------------------------------------------------------------------------------------------------------- """
    model.summary()
    fname   += '.png'
    utils.plot_model( model, to_file=fname, show_shapes=True, show_layer_names=True )



def create_model( arch_kwargs, other_kwargs, summary=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Create the model

    arch_kwargs:    [dict] parameters of the overall architecture
    other_kwargs:   [dict or list of dict] parameters of specific parts of the architecture
    summary:        [str] path where to save plot if you want a summary+plot of model, False otherwise

    return:         the model object
    ------------------------------------------------------------------------------------------------------------- """
    if 'baseline' in other_kwargs[ 0 ].keys():
        nn  = BaselineCNN( arch_kwargs, other_kwargs, summary=summary )
    else:
        nn  = EncoderShiftDecoder( arch_kwargs, other_kwargs, summary=summary )
    nn.model.compile(
            optimizer       = optimiz_code[ nn.optimiz ]( learning_rate=nn.lrate ),
            loss            = nn.loss_func,
            loss_weights    = nn.loss_wght if hasattr( nn, 'loss_wght' ) else None
    )

    return nn
