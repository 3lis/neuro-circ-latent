"""
#####################################################################################################################

    Rotation by latent shift
    Alice   2022

    Configuration file


#####################################################################################################################
"""

kwargs      = {
        # -------- GENERAL ---------------------------------------- #
        'n_epochs':         100,                                    # [int] number of epochs
        'batch_size':       64,                                     # [int] size of batch
        'save_best':        True,                                   # [bool] save best model during training
        'every_rot':        1,                                      # [int] how many rotations to skip during training
        'n_rot':            72,                                     # [int] how many rotations in 2Pi
        'dataset':          "COILRGB",                              # [str] code fo dataset
        'amazon_cam':       5,                                      # [int] # of camera in the Amazon dataset
        'dset_prefetch':    True,                                   # [bool] prefetch samples in ft.data.Dataset
        'dset_cache':       True,                                   # [bool] sample caching in ft.data.Dataset
        'train_val':        None,                                   # [int] ratio training set validation set
                                                                    # or None for a validation set made with
                                                                    # different objects

        # -------- ARCHITECTURE ----------------------------------- #
        'arch_kwargs':      {
                'arch_layout':          'CPCPFDD-DDDRTTT',          # [str] code describing the order of layers in the model
                'lrate':                1e-4,                       # [float] learning rate
                'optimiz':              'ADAM',                     # [str] code of the optimizer
                'loss':                 'MSE'                       # [str] code of the loss function
        },
        # -------- ENCODER ---------------------------------------- #
        'enc_kwargs':       {
                'baseline':             'InceptionV3',              # use a baseline architecture, all the remaning is ignored
                                                                    # currently available models are:
                                                                    #   ResNet50
                                                                    #   InceptionV3
                                                                    #   EfficientNetV2B0
                                                                    # more can be added, but it is necessary to modify arch_net.preprocess_code
                'enc_dnse_size':        [ 256 ]                     # [list of int] size of each dense layer
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
