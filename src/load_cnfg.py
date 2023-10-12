"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Configuration class, checking all the parameters used in the software

#####################################################################################################################
"""

import  os
from    argparse        import ArgumentParser

from    arch_net        import layer_code, optimiz_code, loss_code
from    print_msg       import print_err, print_wrn


dir_models      = 'log'     # folder of (manually) saved models


class Config( object ):
    """ -------------------------------------------------------------------------------------------------------------
    LIST of all PARAMETERS accepted by the software (* indicates compulsory parameters)

    CONFIG:               * [str] name of configuration file (without path nor extension)
    GPU:                  * [int or list of int] number of GPUs to use (0=CPU) or list of GPU indices
    LOAD:                   [str] pathname of HDF5 file to load as weights
    REDIRECT:               [bool] redirect stderr and stdout to log files (DEFAULT=False)
    ARCHIVE:                [bool] archive python scripts (DEFAULT=False)
    TRAIN:                  [bool] execute training of the model (DEFAULT=False)
    TEST:                   [bool] execute testing of the model (DEFAULT=False)
    INVERTED:               [bool] use the inverted model (DEFAULT=False)
    IDX:                    [int*] indexes into the dataset of samples to test
    OBJECT:                 [str] name of an object for which all shifted latents are saved as images

    n_epochs:             * [int] number of epochs
    n_epochs:             * [int] number of epochs
    batch_size:           * [int] batch size
    save_best:              [bool] save best model during training
    n_rot:                  [int] numer of rotations in 2Pi, should be equal to the latent vector dimension
    every_rot:            * [int] how many rotations to skip as second image in the dataset (CARE memory!)
    train_val:            * [int] ratio training set validation set or None for v_classes list of validation objects
    dataset:                [str] code for dataset:
                                    "int"=internal,
                                    "coilBW"    = coil B/W with all dataset in memory,
                                    "coilRGB"   = coil RGB with all dataset in memory,
                                    "COILBW"    = coil B/W using tf.data.Dataset,
                                    "COILRGB"   = coil RGB using tf.data.Dataset,
                                    "amazonBW"  = amazon B/W with all dataset in memory,
                                    "amazonRGB" = amazon RGB with all dataset in memory,
                                    "AMAZONBW"  = amazon B/W using tf.data.Dataset,
                                    "AMAZONRGB" = amazon RGB using tf.data.Dataset,
                                     default internal
    amazon_cam              [int] number of camera in the case of amazon dataset, defaults to 5
    dset_prefetch           [bool] uses  tf.data.Dataset.prefetch(), useful only for tf.data.Dataset
    dset_cache              [bool] uses  tf.data.Dataset.cache(), useful only for tf.data.Dataset


    arch_kwargs:          * [dict] general parameters of architecture, containing:
        arch_layout:          * [str] code describing the order of layers in the model (using 'layer_code' in arch_net.py)
        optimiz:              * [str] code of the optimizer (one of 'optimiz_code' in arch_net.py)
        loss:                 * [str] code of the loss function (one of 'loss_code' in arch_net.py)
        lrate:                * [float] learning rate

        kl_weight:            * [float] weight of KL component in loss function (ONLY FOR VARIATIONAL MODELS)
        loss_wght:              [list of float] weight the loss contributions of different model outputs (TEMPORAL MODELS)

    enc_kwargs:             [dict] parameters of encoder network, containing:
        conv_kernel_num:      * [list of int] number of kernels for each convolution
        conv_kernel_size:     * [list of int] (square) size of kernels for each convolution
        conv_strides:         * [list of int] stride for each convolution
        conv_padding:           [list of str] padding (same/valid) for each convolution (DEFAULT=same)
        conv_activation:        [list of str] activation function for each convolution (DEFAULT=relu)
        conv_train:             [list of bool] False to lock training of each convolution (DEFAULT=True)
        pool_size:              [list of int] pooling size for each max-pooling
        enc_dnse_size:        * [list of int] size of each dense layer
        enc_dnse_dropout:       [list of int] dropout of each dense layer (DEFAULT=0)
        enc_dnse_activation:    [list of str] activation function for each dense layer (DEFAULT=relu)
        enc_dnse_train:         [list of bool] False to lock training of each dense layer (DEFAULT=True)

    dec_kwargs:             [dict] parameters of encoder network, containing:
        first_3d_shape:       * [list] the first 3D shape after reshaping (height, width, channels)
        dcnv_kernel_num:      * [list of int] number of kernels for each deconvolution
        dcnv_kernel_size:     * [list of int] (square) size of kernels for each deconvolution
        dcnv_strides:         * [list of int] stride for each deconvolution
        dcnv_padding:           [list of str] padding (same/valid) for each deconvolution (DEFAULT=same)
        dcnv_activation:        [list of str] activation function for each deconvolution (DEFAULT=relu)
        dcnv_train:             [list of bool] False to lock training of each deconvolution (DEFAULT=True)
        dec_dnse_size:        * [list of int] size of each dense layer
        dec_dnse_dropout:       [list of int] dropout of each dense layer (DEFAULT=0)
        dec_dnse_activation:    [list of str] activation function for each dense layer (DEFAULT=relu)
        dec_dnse_train:         [list of bool] False to lock training of each dense layer (DEFAULT=True)
    ------------------------------------------------------------------------------------------------------------- """

    def load_from_line( self, line_kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Load parameters from arguments passed in command line. Check the existence and correctness of all
        required parameteres

        line_kwargs:        [dict] parameteres read from arguments passed in command line
        --------------------------------------------------------------------------------------------------------- """
        for key, value in line_kwargs.items():
            setattr( self, key, value )

        # load an existing model, pass a new configuration, and create a new folder
        if self.LOAD is not None:
            assert os.path.isfile( self.LOAD ), "File '{}' is not an HDF5 file.".format( self.LOAD )


    def load_from_file( self, file_kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Load parameters from a python file. Check the existence and correctness of all required parameteres

        In some cases, if a parameter is not passed as argument, it is set to a default value.
        In other cases, if a parameter is considered fundamental, it must be specified as argument.

        file_kwargs:        [dict] parameteres coming from a python module (file)
        --------------------------------------------------------------------------------------------------------- """
        for key, value in file_kwargs.items():
            setattr( self, key, value )


        # -------- GENERAL -------- 
        assert hasattr( self, 'n_epochs' )                  and isinstance( self.n_epochs, int )
        assert hasattr( self, 'batch_size' )                and isinstance( self.batch_size, int )
        assert hasattr( self, 'every_rot' )                 and isinstance( self.every_rot, int )
        assert hasattr( self, 'train_val' )                 and (
                isinstance( self.train_val, int ) or self.train_val is None )

        if not hasattr( self, 'n_rot' ):
            self.n_rot              = 72                    # backward compatibility

        if not hasattr( self, 'save_best' ):
            self.save_best          = False                 # backward compatibility

        if not hasattr( self, 'dataset' ):
            if hasattr( self, 'use_coil' ):                 # backward compatibility
                if self.use_coil:
                    self.dataset   = "coilBW"
                else:
                    self.dataset   = "int"
            else:
                self.dataset   = "int"
        assert self.dataset in (
                "int",
                "coilBW",
                "coilRGB",
                "COILBW",
                "COILRGB",
                "amazonBW",
                "amazonRGB",
                "AMAZONBW",
                "AMAZONRGB"
                )
        if not hasattr( self, 'amazon_cam' ):
            if self.dataset in (
                "amazonBW",
                "amazonRGB",
                "AMAZONBW",
                "AMAZONRGB"
                ):
                self.amazon_cam = 5                     # default camera in amazon dataset
            else:
                self.amazon_cam = None                  # not in use for other datasets
        if not hasattr( self, 'dset_prefetch' ):
            self.dset_prefetch  = False
        if not hasattr( self, 'dset_cache' ):
            self.dset_cache     = False

        # -------- ARCHITECTURE -------- 
        assert hasattr( self, 'arch_kwargs' )               and isinstance( self.arch_kwargs, dict )


        assert 'arch_layout' in self.arch_kwargs            and isinstance( self.arch_kwargs[ 'arch_layout' ], str )
        assert 'optimiz' in self.arch_kwargs                and self.arch_kwargs[ 'optimiz' ] in optimiz_code
        assert 'loss' in self.arch_kwargs                   and self.arch_kwargs[ 'loss' ] in loss_code
        assert 'lrate' in self.arch_kwargs                  and isinstance( self.arch_kwargs[ 'lrate' ], float )

        assert hasattr( self, 'enc_kwargs' )
        if 'baseline' in self.enc_kwargs:                   # the baseline architecture don't use arch_layout
                                                            # could use just one Dense, the last in enc_dnse_size
            self.dec_kwargs     = {}
            return False

        if hasattr( self, 'enc_kwargs' ):
            assert isinstance( self.arch_kwargs, dict )    

            n_conv      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ 0 ].count( layer_code[ 'CONV' ] )
            n_pool      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ 0 ].count( layer_code[ 'POOL' ] )
            n_dnse      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ 0 ].count( layer_code[ 'DNSE' ] )

            assert 'conv_kernel_num' in self.enc_kwargs         and len( self.enc_kwargs[ 'conv_kernel_num' ] ) == n_conv
            assert 'conv_kernel_size' in self.enc_kwargs        and len( self.enc_kwargs[ 'conv_kernel_size' ] ) == n_conv
            assert 'conv_strides' in self.enc_kwargs            and len( self.enc_kwargs[ 'conv_strides' ] ) == n_conv
            assert 'enc_dnse_size' in self.enc_kwargs           and len( self.enc_kwargs[ 'enc_dnse_size' ] ) == n_dnse

            assert self.enc_kwargs[ 'enc_dnse_size' ][ -1 ] == self.n_rot, "Number of rotations don't match with latent size"
            if n_pool > 0:
                assert 'pool_size' in self.enc_kwargs           and len( self.enc_kwargs[ 'pool_size' ] ) == n_pool

            if 'conv_padding' not in self.enc_kwargs:
                self.enc_kwargs[ 'conv_padding' ]               = n_conv * [ 'same' ]
            elif isinstance( self.enc_kwargs[ 'conv_padding' ], str ):
                self.enc_kwargs[ 'conv_padding' ]               = n_conv * [ self.enc_kwargs[ 'conv_padding' ] ]

            if 'conv_activation' not in self.enc_kwargs:
                self.enc_kwargs[ 'conv_activation' ]            = n_conv * [ 'relu' ]
            elif isinstance( self.enc_kwargs[ 'conv_activation' ], str ):
                self.enc_kwargs[ 'conv_activation' ]            = n_conv * [ self.enc_kwargs[ 'conv_activation' ] ]

            if 'conv_train' not in self.enc_kwargs:
                self.enc_kwargs[ 'conv_train' ]                 = n_conv * [ True ]
            elif isinstance( self.enc_kwargs[ 'conv_train' ], bool ):
                self.enc_kwargs[ 'conv_train' ]                 = n_conv * [ self.enc_kwargs[ 'conv_train' ] ]

            if 'enc_dnse_dropout' not in self.enc_kwargs:
                self.enc_kwargs[ 'enc_dnse_dropout' ]           = n_dnse * [ 0 ]
            else:
                assert len( self.enc_kwargs[ 'enc_dnse_dropout' ] ) == n_dnse

            if 'enc_dnse_activation' not in self.enc_kwargs:
                self.enc_kwargs[ 'enc_dnse_activation' ]        = n_dnse * [ 'relu' ]
            elif isinstance( self.enc_kwargs[ 'enc_dnse_activation' ], str ):
                self.enc_kwargs[ 'enc_dnse_activation' ]        = n_dnse * [ self.enc_kwargs[ 'enc_dnse_activation' ] ]

            if 'enc_dnse_train' not in self.enc_kwargs:
                self.enc_kwargs[ 'enc_dnse_train' ]             = n_dnse * [ True ]
            elif isinstance( self.enc_kwargs[ 'enc_dnse_train' ], bool ):
                self.enc_kwargs[ 'enc_dnse_train' ]             = n_dnse * [ self.enc_kwargs[ 'enc_dnse_train' ] ]

            if sum( self.enc_kwargs[ 'enc_dnse_train' ] ) != n_dnse:        # some layer is non-trainable
                if sum( self.enc_kwargs[ 'enc_dnse_dropout' ] ) > 0:        # some layer has dropout
                    print_wrn( "You have dropout on a frozen layer, are you sure you want this??" )


        # -------- DECODER -------- 
        assert hasattr( self, 'dec_kwargs' )
        if hasattr( self, 'dec_kwargs' ):
            assert isinstance( self.arch_kwargs, dict )

            n_dcnv      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ -1 ].count( layer_code[ 'DCNV' ] )
            n_dnse      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ -1 ].count( layer_code[ 'DNSE' ] )

            assert 'dcnv_kernel_num' in self.dec_kwargs         and len( self.dec_kwargs[ 'dcnv_kernel_num' ] ) == n_dcnv
            assert 'dcnv_kernel_size' in self.dec_kwargs        and len( self.dec_kwargs[ 'dcnv_kernel_size' ] ) == n_dcnv
            assert 'dcnv_strides' in self.dec_kwargs            and len( self.dec_kwargs[ 'dcnv_strides' ] ) == n_dcnv
            assert 'dec_dnse_size' in self.dec_kwargs           and len( self.dec_kwargs[ 'dec_dnse_size' ] ) == n_dnse
            assert 'first_3d_shape' in self.dec_kwargs          and len( self.dec_kwargs[ 'first_3d_shape' ] ) == 3

            if 'dcnv_padding' not in self.dec_kwargs:
                self.dec_kwargs[ 'dcnv_padding' ]               = n_dcnv * [ 'same' ]
            elif isinstance( self.dec_kwargs[ 'dcnv_padding' ], str ):
                self.dec_kwargs[ 'dcnv_padding' ]               = n_dcnv * [ self.dec_kwargs[ 'dcnv_padding' ] ]

            if 'dcnv_activation' not in self.dec_kwargs:
                self.dec_kwargs[ 'dcnv_activation' ]            = n_dcnv * [ 'relu' ]
            elif isinstance( self.dec_kwargs[ 'dcnv_activation' ], str ):
                self.dec_kwargs[ 'dcnv_activation' ]            = n_dcnv * [ self.dec_kwargs[ 'dcnv_activation' ] ]

            if 'dcnv_train' not in self.dec_kwargs:
                self.dec_kwargs[ 'dcnv_train' ]                 = n_dcnv * [ True ]
            elif isinstance( self.dec_kwargs[ 'dcnv_train' ], bool ):
                self.dec_kwargs[ 'dcnv_train' ]                 = n_dcnv * [ self.dec_kwargs[ 'dcnv_train' ] ]

            if 'dec_dnse_dropout' not in self.dec_kwargs:
                self.dec_kwargs[ 'dec_dnse_dropout' ]           = n_dnse * [ 0 ]
            else:
                assert len( self.dec_kwargs[ 'dec_dnse_dropout' ] ) == n_dnse

            if 'dec_dnse_activation' not in self.dec_kwargs:
                self.dec_kwargs[ 'dec_dnse_activation' ]        = n_dnse * [ 'relu' ]
            elif isinstance( self.dec_kwargs[ 'dec_dnse_activation' ], str ):
                self.dec_kwargs[ 'dec_dnse_activation' ]        = n_dnse * [ self.dec_kwargs[ 'dec_dnse_activation' ] ]

            if 'dec_dnse_train' not in self.dec_kwargs:
                self.dec_kwargs[ 'dec_dnse_train' ]             = n_dnse * [ True ]
            elif isinstance( self.dec_kwargs[ 'dec_dnse_train' ], bool ):
                self.dec_kwargs[ 'dec_dnse_train' ]             = n_dnse * [ self.dec_kwargs[ 'dec_dnse_train' ] ]

            if sum( self.dec_kwargs[ 'dec_dnse_train' ] ) != n_dnse:        # some layer is non-trainable
                if sum( self.dec_kwargs[ 'dec_dnse_dropout' ] ) > 0:        # some layer has dropout
                    print_wrn( "You have dropout on a frozen layer, are you sure you want this??" )

        return True


    def __str__( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Visualize the list of all parameters
        --------------------------------------------------------------------------------------------------------- """
        s   = ''
        d   = self.__dict__

        for k in d:
            if isinstance( d[ k ], dict ):
                s   += "{}:\n".format( k )
                for j in d[ k ]:
                    s   += "{:5}{:<30}{}\n".format( '', j, d[ k ][ j ] )
            else:
                s   += "{:<35}{}\n".format( k, d[ k ] )

        return s


# ===================================================================================================================


def read_args():
    """ -------------------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags
    
    return:         [dict] key = name of parameter, value = value of parameter
    ------------------------------------------------------------------------------------------------------------- """
    parser      = ArgumentParser()

    parser.add_argument(
            '-a',
            '--angle',
            action          = 'store_true',
            dest            = 'INVERTED',
            help            = "Use a trained model in inverted mode to predict the angle"
    )

    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            default         = None,
            help            = "Name of configuration file (without path nor extension)"
    )
    parser.add_argument(
            '-g',
            '--gpu',
            action          = 'store',
            dest            = 'GPU',
            required        = True,
            type            = int,
            help            = "Number of GPUs to use (-1 for CPU)"
    )
    parser.add_argument(
            '-i',
            '--index',
            nargs           = '+',
            dest            = 'IDX',
            default         = None,
            type            = int,
            help            = "indexes of samples to be tested with graphic output"
    )
    parser.add_argument(
            '-l',
            '--load',
            action          = 'store',
            dest            = 'LOAD',
            type            = str,
            default         = None,
            help            = "HDF5 file to load as weights or entire model"
    )
    parser.add_argument(
            '-L',
            '--latent',
            action          = 'store_true',
            dest            = 'LATENT',
            help            = "Execute latent analysis of the model"
    )
    parser.add_argument(
            '-o',
            '--object',
            action          = 'store',
            dest            = 'OBJECT',
            type            = str,
            default         = None,
            help            = "name of an object for which all shifted latents are shown"
    )
    parser.add_argument(
            '-r',
            '--redir',
            action          = 'store_true',
            dest            = 'REDIRECT',
            help            = "Redirect stderr and stdout to log files"
    )
    parser.add_argument(
            '-s',
            '--save',
            action          = 'store_true',
            dest            = 'ARCHIVE',
            help            = "Archive python scripts"
    )
    parser.add_argument(
            '-T',
            '--train',
            action          = 'store_true',
            dest            = 'TRAIN',
            help            = "Execute training of the model"
    )
    parser.add_argument(
            '-t',
            '--test',
            action          = 'store_true',
            dest            = 'TEST',
            help            = "Execute testing of the model"
    )

    return vars( parser.parse_args() )
