"""
#####################################################################################################################

    Alice   2022

#####################################################################################################################
"""

SEED = 1
import      os
import      numpy           as np
import      random          as rn
import      tensorflow      as tf
os.environ[ 'PYTHONHASHSEED' ] = str( SEED )
np.random.seed( SEED )
rn.seed( SEED )
# -------------------------------------------------------------------------------------------------------------------

import      sys
import      platform
import      time
import      datetime
import      pickle

import      arch_net
import      load_cnfg
import      load_model
import      data_gen
import      test_net

from        matplotlib          import pyplot
from        tensorflow.keras    import callbacks
from        print_msg           import print_err, print_wrn, print_flush


FRMT                    = "%y-%m-%d_%H-%M-%S"   # datetime format for folder names


# folders and files inside the main execution folder - NOTE the variables will be updated in init_dirs()
dir_current         	= None
dir_res             	= '../res'
dir_log             	= 'log'
dir_src             	= 'src'
dir_plot            	= 'plot'
dir_test            	= 'test'
log_train           	= "train.log"
log_hist            	= "hist.pickle"
log_err             	= "err.log"
log_time            	= "time.log"
nn_best             	= "nn_best.h5"
nn_final            	= "nn_final.h5"
file_cnfg               = "cnfg.py"

shuffle_buffer      	= 128                   # size of the shuffle buffer, not clear how to set it,
                                                # seems that the larger the better, but in hp machine larger
                                                # values require too much memory

nn                      = None                  # network object
cnfg                    = None                  # [Config] object keeping together all parameters
bsline                  = None                  # ture in case of a baseline model
train_set, valid_set    = None, None            # [list] training and validation sets
train_desc, valid_desc  = None, None            # [list] training and validation descriptions

DO_NOTHING              = False                 # for debugging

def init_cnfg():
    """ -------------------------------------------------------------------------------------------------------------
    Set global parameters from command line and python file
    ------------------------------------------------------------------------------------------------------------- """
    global cnfg, bsline

    cnfg            = load_cnfg.Config()

    # load parameters from command line
    line_kwargs     = load_cnfg.read_args()
    cnfg.load_from_line( line_kwargs )


    # load parameters from file
    exec( "import " + cnfg.CONFIG )                                 # exec the import statement
    file_kwargs     = eval( cnfg.CONFIG + ".kwargs" )               # assign the content to a variable
    bsline          = not cnfg.load_from_file( file_kwargs )        # read the configuration file,

    # add needed info in architecture dict
    cnfg.arch_kwargs[ 'batch_size' ]    = cnfg.batch_size
    
    if cnfg.GPU >= 0:                   # if it is not indicated to use the CPU, choose a GPU
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( cnfg.GPU )

    # inform other modules
    data_gen.n_rot  = cnfg.n_rot
    data_gen.bsline = bsline

    if cnfg.dataset in ( "coilRGB", "COILRGB", "amazonRGB", "AMAZONRGB" ):
        data_gen.color  = True          # needs to set color soon, because arch_net rely on it
    if cnfg.amazon_cam is not None:
        data_gen.amazon_cam = cnfg.amazon_cam


def init_dirs():
    """ -------------------------------------------------------------------------------------------------------------
    Set paths to directories where to save the execution
    ------------------------------------------------------------------------------------------------------------- """
    global dir_current, dir_log, dir_plot, dir_src, dir_test                                        # dirs
    global log_train, log_time, log_err, log_hist, nn_best, nn_final, file_cnfg                     # files

    if cnfg.LOAD is not None:
        dir_current     = os.path.dirname( cnfg.LOAD )
    else:
        dir_current     = os.path.join( dir_res, time.strftime( FRMT ) )
    dir_log         = os.path.join( dir_current, dir_log )
    dir_src         = os.path.join( dir_current, dir_src )
    dir_plot        = os.path.join( dir_current, dir_plot )
    dir_test        = os.path.join( dir_current, dir_test )

    if cnfg.TRAIN:
        os.makedirs( dir_current )
        os.makedirs( dir_log )
        os.makedirs( dir_src )
        os.makedirs( dir_plot )
        os.makedirs( dir_test )

    log_train       = os.path.join( dir_log, log_train )
    log_hist        = os.path.join( dir_log, log_hist )
    log_err         = os.path.join( dir_log, log_err )
    log_time        = os.path.join( dir_log, log_time )
    nn_best         = os.path.join( dir_current, nn_best )
    nn_final        = os.path.join( dir_current, nn_final )
    file_cnfg       = os.path.join( dir_current, file_cnfg )


def archive():
    """ -------------------------------------------------------------------------------------------------------------
    Archive python source code and configuration file
    ------------------------------------------------------------------------------------------------------------- """
    os.system( "cp {}.py {}".format( '*', dir_src ) )               # save python sources
    os.system( "cp {}.py {}".format( cnfg.CONFIG, file_cnfg ) )     # save config file in "main" folder


def create_model( summ=True ):
    """ -------------------------------------------------------------------------------------------------------------
    Create the model object
    ------------------------------------------------------------------------------------------------------------- """
    other_kwargs    = ( cnfg.enc_kwargs, cnfg.dec_kwargs )

    nn              = arch_net.create_model(
            cnfg.arch_kwargs,
            other_kwargs,
            summary     = dir_plot if summ else False
    )

    sys.stdout.flush()          # flush to have the current stdout in the log

    return nn


def create_dset():
    """ -------------------------------------------------------------------------------------------------------------
    create the datasets for training and validation
    ------------------------------------------------------------------------------------------------------------- """
    global train_set, valid_set, train_desc, valid_desc
    print_flush( "Now creating the datasets...\n" )
    train_set, valid_set, train_desc, valid_desc    = data_gen.gen_dset(
            cnfg.every_rot,
            cnfg.train_val,
            cnfg.dataset
    )


def create_testset():
    """ -------------------------------------------------------------------------------------------------------------
    create the datasets for test
    note that the dataset and its description is written in the same globals used for validation
    ------------------------------------------------------------------------------------------------------------- """
    global valid_set, valid_desc
    print_flush( "Now creating the test dataset...\n" )
    if isinstance( valid_set, dict ):
        return True

    valid_set, valid_desc    = data_gen.gen_test_set(
            cnfg.every_rot,
            cnfg.train_val,
            cnfg.dataset
    )

    return False



def plot_history( history, fname ):
    """ -------------------------------------------------------------------------------------------------------------
    Plot the loss performance, on training and validation sets

    history:        [keras.callbacks.History]
    fname:          [str] path+name of output file without extension
    ------------------------------------------------------------------------------------------------------------- """
    train_loss  = history.history[ 'loss' ]
    valid_loss  = history.history[ 'val_loss' ]
    epochs      = range( 1, len( train_loss ) + 1 )

    pyplot.plot( epochs, train_loss, 'r--' )
    pyplot.plot( epochs, valid_loss, 'b-' )

    pyplot.legend( [ 'Training Loss', 'Validation Loss' ] )
    pyplot.xlabel( 'Epoch' )
    pyplot.ylabel( 'Loss' )

    pyplot.grid( True )
    pyplot.rc( 'grid', linestyle='--', color='lightgrey' )
    pyplot.savefig( "{}.pdf".format( fname ) )

    if len( train_loss ) > 5:
        m   = np.mean( train_loss )
        s   = np.std( train_loss )
        pyplot.ylim( [ m - s, m + s ] )
        pyplot.grid( True )
        pyplot.savefig( "{}_zoom.pdf".format( fname ) )

    pyplot.close()



def train_model():
    """ -------------------------------------------------------------------------------------------------------------
    Train the model
    ------------------------------------------------------------------------------------------------------------- """
    global train_set, valid_set
    print_flush( "Now starting training...\n" )
    t_start     = datetime.datetime.now()

    if cnfg.save_best:
        callback    = [ callbacks.ModelCheckpoint( filepath=nn_best, save_weights_only=True, save_best_only=True ) ]
    else:
        callback    = None

    if cnfg.dataset in ( "COILBW", "COILRGB", "AMAZONBW", "AMAZONRGB" ):
# in this case train_set and valid_set are of type tf.data.Dataset, for which a variety of tf functions
# are available, here the most essential for training have been used, is might be useful to check in more
# detail the tf.data.Dataset documentation for further improvements
        train_set   = train_set.batch( cnfg.batch_size )            # organize the dataset in batches
        if cnfg.dset_cache:
# it might speed up training, but may occupy too much memory NOTE: should be called BEFORE shuffling
            train_set   = train_set.cache()                         
        train_set   = train_set.shuffle( shuffle_buffer, reshuffle_each_iteration=True )
        if cnfg.dset_prefetch:
            train_set   = train_set.prefetch( tf.data.AUTOTUNE )    # it might speed up training
        valid_set   = valid_set.batch( cnfg.batch_size )
        hist        = nn.model.fit(
                x                   = train_set,
                validation_data     = valid_set,
                epochs              = cnfg.n_epochs,
                callbacks           = callback,
                verbose             = 2 if cnfg.ARCHIVE else 1
        )
    else:
        hist        = nn.model.fit(
                x                   = train_set[ 'x' ],
                y                   = train_set[ 'y' ],
                validation_data     = ( valid_set[ 'x' ], valid_set[ 'y' ] ),
                epochs              = cnfg.n_epochs,
                batch_size          = cnfg.batch_size,
                callbacks           = callback,
                verbose             = 2 if cnfg.ARCHIVE else 1,
                shuffle             = True
        )

    t_end       = datetime.datetime.now()

    # save model
    nn.model.save_weights( nn_final )       # NOTE even in case of n_gpus>1, nn.model is the one to be saved

    # save and plot history
    with open( log_hist, 'wb') as f:
        pickle.dump( hist.history, f )
    plot_history( hist, os.path.join( dir_plot, 'loss' ) )

    # save duration of training
    with open( log_time, 'a' ) as f:
        f.write( "Training duration:\t{}\n".format( str( t_end - t_start ) ) )

    print_flush( "\nTraining completed" )


def test_model():
    """ -------------------------------------------------------------------------------------------------------------
    Test the model
    ------------------------------------------------------------------------------------------------------------- """
    print_flush( "Now starting testing...\n" )
    t_start             = datetime.datetime.now()

    test_net.save_dir   = dir_test
    if cnfg.dataset in ( "coilRGB", "COILRGB", "AMAZONRGB", "AMAZONRGB" ):
        test_net.metric_tset_color( nn.model, valid_set, valid_desc )
    else:
        test_net.metric_tset_bw( nn.model, valid_set, valid_desc )

    # save duration of tests
    t_end               = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Testing duration:\t{}\n".format( str( t_end - t_start ) ) )


def test_inverted():
    """ -------------------------------------------------------------------------------------------------------------
    Test the model for predicting and angle
    in the case of the baseline the full model is used, otherwise the "inverted" model with the encoder only
    ------------------------------------------------------------------------------------------------------------- """
    print_flush( "Now starting testing the inverted model...\n" )
    t_start             = datetime.datetime.now()

    test_net.save_dir   = dir_test
    model               = nn.model  if bsline   else nn.encoder.model
    test_net.metric_inverted( model, valid_set, valid_desc )

    # save duration of tests
    t_end               = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Testing duration:\t{}\n".format( str( t_end - t_start ) ) )


def test_samples():
    """ -------------------------------------------------------------------------------------------------------------
    Test the model on a short list of samples, producing graphic output
    ------------------------------------------------------------------------------------------------------------- """
    print_flush( "Now starting testing list of samples...\n" )
    t_start             = datetime.datetime.now()

    test_net.save_dir   = dir_test
    for i in cnfg.IDX:
        if cnfg.dataset in ( "coilRGB", "COILRGB", "AMAZONRGB", "AMAZONRGB" ):
            test_net.show_tset_one_color( nn.model, valid_set, i )
        else:
            test_net.show_tset_one_bw( nn.model, valid_set, i )

    # save duration of tests
    t_end               = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Samples testing duration:\t{}\n".format( str( t_end - t_start ) ) )



def test_inv_samples():
    """ -------------------------------------------------------------------------------------------------------------
    Test the simple inverted model on a short list of samples, producing textual output
    see test_net.pred_invert_one() for a descriptipon of how this model works
    ------------------------------------------------------------------------------------------------------------- """
    print_flush( "Now starting testing list of samples on the inverted model...\n" )
    t_start             = datetime.datetime.now()

    test_net.save_dir   = dir_test
    internal            = cnfg.dataset == "int"
    for i in cnfg.IDX:
        test_net.show_invert_one( nn.encoder.model, valid_set, valid_desc, i, internal )


    # save duration of tests
    t_end               = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Samples testing duration:\t{}\n".format( str( t_end - t_start ) ) )


def latent_analysis():
    """ -------------------------------------------------------------------------------------------------------------
    Perform an analysis of the latent space, using single images as stimuli, and the encoder only
    results are stored as images in the test folder of the result directory of the used model
    ------------------------------------------------------------------------------------------------------------- """
    print_flush( "Now starting latent space analysis...\n" )
    t_start             = datetime.datetime.now()

    test_net.save_dir   = dir_test
    model               = nn.model  if bsline   else nn.encoder.model
    vl_set, vl_desc     = data_gen.gen_onerot_tset( dataset=cnfg.dataset )
    latent              = test_net.pred_onerot_tset( model, vl_set )
    lat_code            = test_net.latent_coding( latent, vl_desc )
    lat_mean_code       = test_net.latent_mean_coding( latent, vl_desc )

    # save duration of tests
    t_end               = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "latent analysis duration:\t{}\n".format( str( t_end - t_start ) ) )

    

# ===================================================================================================================
#
#   MAIN
#
#
# ===================================================================================================================
if __name__ == '__main__' and not DO_NOTHING:

    init_cnfg()
    init_dirs()

    t_start             = datetime.datetime.now()

    # NOTE to restore use sys.stdout = sys.__stdout__
    if cnfg.REDIRECT:
        sys.stdout      = open( log_train, 'w' )
        sys.stderr      = open( log_err, 'w' )
        command         = sys.executable + " " + " ".join( sys.argv )
        host            = platform.node()
        print( "executing:\n" + command )
        print( "on host " + host + "\n\n" )


    if cnfg.ARCHIVE:
        archive()

    nn                  = create_model( summ=cnfg.TRAIN )

    if cnfg.LOAD is not None:
#       if cnfg.INVERTED:
#           load_model.load_encoder( nn.model, cnfg.LOAD )
#       else:
#           nn.model.load_weights( cnfg.LOAD )
        nn.model.load_weights( cnfg.LOAD )


    if cnfg.TRAIN:
        create_dset()
        train_model()

    if cnfg.TEST:
        create_testset()
        if cnfg.INVERTED:
            test_inverted()
        else:
            test_model()

    if cnfg.IDX is not None:
        create_testset()
        if cnfg.INVERTED:
            test_inv_samples()
        else:
            test_samples()

    if cnfg.OBJECT:
        test_net.save_dir   = dir_test
        test_net.show_latents( nn, cnfg.OBJECT, cnfg.dataset )

    if cnfg.LATENT:
        latent_analysis()

    # save total duration
    t_end       = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Total duration:\t\t{}\n".format( str( t_end - t_start ) ) )

    print_flush( '~ End of execution! ~\n' )
