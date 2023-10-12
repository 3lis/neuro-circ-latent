"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Utilities for printing messages

#####################################################################################################################
"""

import      os
import      sys
import      inspect


def print_flush( msg ):
    """ ---------------------------------------------------------------------------------------------------------
    Print an message and immediately flush the buffer to have the current stdout in the log

    msg:        [str] message to print
    --------------------------------------------------------------------------------------------------------- """
    print( msg )
    sys.stdout.flush()
    
    

def print_err( msg ):
    """ ---------------------------------------------------------------------------------------------------------
    Print an error message, including the file and line number where the print is called

    msg:        [str] message to print
    --------------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    print( "ERROR [{}:{}] --> {}\n".format( FILE, LINE, msg ) )

    # sys.exit( 1 )
    os._exit( 1 )   # even more brutal, useful inside the TF computation graph



def print_wrn( msg ):
    """ ---------------------------------------------------------------------------------------------------------
    Print a warning message, including the file and line number where the print is called

    msg:        [str] message to print
    --------------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    print( "WARNING [{}:{}] --> {}\n".format( FILE, LINE, msg ) )
