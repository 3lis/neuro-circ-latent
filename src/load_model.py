"""
#####################################################################################################################

    Alice   2022

    Load parts of trained models from HDF5 binary files

#####################################################################################################################
"""

import  h5py
import  keras.backend   as K
import  numpy           as np

from    print_msg       import print_err, print_wrn


def load_h5( nn, h5_file, part ):
    """ -------------------------------------------------------------------------------------------------------------
    Load weights of a submodel in a full HDF5 file into a neural network.

    NOTE: the function supposes that the network and the HDF5 dataset share the same layer names
    WARNING: this functions currenty does not work, there are problems with batch_set_value()
            found in
            https://stackoverflow.com/questions/58364974/how-to-load-trained-autoencoder-weights-for-decoder
            but not ufficialy documented in Keras

    nn:             [keras.models.Model] neural network to be filled
    h5_file:        [str] pathname to the HDF5 file
    part:           [str] name of a submodel - also just a part
    ------------------------------------------------------------------------------------------------------------- """
    try:        h5  = h5py.File( h5_file, 'r' )
    except:     print_err( "Error opening HDF5 file {}".format( h5_file ) )

    h5_part         = None
    for k in h5.keys():
        if part in k:
            h5_part = k
            break
    if part is None:
        print_err( "'{}' is not a part of the HDF5 file {}".format( h5_part, h5_file ) )
    h5              = h5[ h5_part ]
    h5_w_names      = h5.attrs['weight_names']
    nn_weights      = nn.trainable_weights
    nn_w_names      = [ w.name for w in nn_weights ]
    for n in h5_w_names:
        if n not in nn_weights:
            print_err( "No layers {} found in the model".format( n ) )
    h5_weights      = [ np.array( h5[ w ] ) for w in h5_w_names ]

    K.batch_set_value( zip ( nn_weights, h5_weights ) )


def load_encoder( nn, h5_file ):
    """ -------------------------------------------------------------------------------------------------------------
    Load weights of the encoder submodel in a full HDF5 file into a neural network.

    nn:             [keras.models.Model] neural network to be filled
    h5_file:        [str] pathname to the HDF5 file
    ------------------------------------------------------------------------------------------------------------- """

    load_h5( nn, h5_file, 'encoder' )
