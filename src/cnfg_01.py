"""
#####################################################################################################################

    Rotation by latent shift
    Alice   2022

    Configuration file


#####################################################################################################################
"""

kwargs      = {
        # -------- GENERAL ---------------------------------------- #
        'n_epochs':         2,                                      # [int] number of epochs
        'batch_size':       64,                                     # [int] size of batch
        'save_best':        True,                                   # [bool] save best model during training
        'every_rot':        2,                                      # [int] how many rotations to skip during training
        'n_rot':            120,                                    # [int] how many rotations in 2Pi
        'dataset':          "AMAZONRGB",                            # [str] code fo dataset
        'amazon_cam':       5,                                      # [int] # of camera in the Amazon dataset
        'dset_prefetch':    True,                                   # [bool] prefetch samples in ft.data.Dataset
        'dset_cache':       True,                                   # [bool] sample caching in ft.data.Dataset
        'train_val':        None,                                   # [int] ratio training set validation set
                                                                    # or None for a validation set made with
                                                                    # different objects

        # -------- ARCHITECTURE ----------------------------------- #
        'arch_kwargs':      {
                'arch_layout':          'CPCPFDDD-DDDRTTT',         # [str] code describing the order of layers in the model
                'lrate':                1e-5,                       # [float] learning rate
                'optimiz':              'ADAM',                     # [str] code of the optimizer
                'loss':                 'MSE'                       # [str] code of the loss function
        },
        # -------- ENCODER ---------------------------------------- #
        'enc_kwargs':       {
                'conv_kernel_num':      [ 64, 128 ],                # [list of int] number of kernels for each convolution
                'conv_kernel_size':     [ 5, 3 ],                   # [list of int] (square) size of kernels for each convolution
                'conv_strides':         [ 2, 1 ],                   # [list of int] stride for each convolution
                'pool_size':            [ 2, 2 ],                   # [list of int] pooling size for each MaxPooling
                'enc_dnse_size':        [ 256, 256, 120 ],          # [list of int] size of each dense layer
                'enc_dnse_dropout':     [ 0.1, 0.1, 0.1 ]           # [list of int] dropout of each dense layer
        },
        # -------- DECODER ---------------------------------------- #
        'dec_kwargs':       {
                'first_3d_shape':       ( 16, 16, 4 ),              # [tuple] the first 3D shape after reshaping
                'dcnv_kernel_num':      [ 64, 32, 3 ],              # [list of int] number of kernels for each deconvolution
                'dcnv_kernel_size':     [ 3, 3, 5 ],                # [list of int] (square) size of kernels for each deconv
                'dcnv_strides':         [ 2, 2, 2 ],                # [list of int] stride for each deconvolution
                'dec_dnse_size':        [ 516, 1024, 1024 ],        # [list of int] size of each dense layer
                'dec_dnse_dropout':     [ 0.1, 0.1, 0.1 ]           # [list of int] dropout of each dense layer
        }
}
