"""
#############################################################################################################

general image preprocessing utilities

    Alice   2022

#############################################################################################################
"""

import  os
import  glob
import  numpy               as np
import  argparse


from    PIL                 import Image, ImageDraw
from    math                import pi, sin, cos

# ===========================================================================================================
#
#   Support functions for image handling
#
#   - array_to_image
#   - image_to_array
#   - save_image
#   - save_collage
#   - save_animation
#
# ===========================================================================================================

def array_to_image( array, remove_channel=False, rgb=False ):
    """ -----------------------------------------------------------------------------------------------------
    Convert numpy.ndarray to PIL.Image

    array:          [numpy.ndarray] pixel values (between 0..1)
    remove_channel: [bool] in case of numpy array with extra dimension for channel, discard it
    rgb:            [bool] treat the array as image with RGB channels

    return:         [PIL.Image.Image]
    ----------------------------------------------------------------------------------------------------- """
    if len( array.shape ) == 3 and not rgb:
        if remove_channel:
            array   = array.squeeze()
        else:
            array   = array[ 0, ... ]                             # remove batch axis

    ptp     = array.ptp()
    if ptp:
        array  = ( array - array.min() ) / ptp
    array  = 255. * array                                         # normalization
    array  = np.uint8( array )
    img     = Image.fromarray( array )

    return img



def image_to_array( img ):
    """ -----------------------------------------------------------------------------------------------------
    Convert PIL.Image to numpy.ndarray

    img:            [PIL.Image] path to image file

    return:         [numpy.ndarray] pixel values (between 0..1)
    ----------------------------------------------------------------------------------------------------- """
    i       = np.asarray( img, dtype=float )
    i       /= 255.                                         # NOTE normalization between 0..1

    return i



def file_to_array( fname, remove_channel=True, resize=None ):
    """ -----------------------------------------------------------------------------------------------------
    Convert PIL.Image to numpy.ndarray

    fname:          [str] path of input file
    remove_channel: [bool] in case of image with color channels, collapse to graylevel
    resize:         [tuple] final size of the output array, None to keep the original size

    return:         [numpy.ndarray] pixel values (between 0..1)
    ----------------------------------------------------------------------------------------------------- """
    assert os.path.isfile( fname ), f"image file {fname} does not exist"
    img     = Image.open( fname )
    if remove_channel:
        img     = img.convert( 'L' )
    if resize is not None:
        assert isinstance( resize, tuple ), f"Invalid resize argument {resize}"
        img = img.resize( resize )

    return image_to_array( img )



def save_image( img, fname, resize=None, rgb=False, transpose=False ):
    """ -----------------------------------------------------------------------------------------------------
    Save an image to file

    img:            [numpy.ndarray or PIL.Image.Image] image
    fname:          [str] path of output file
    resize:         [tuple] final size of the saved image, None to keep the original size
    transpose:      [bool] transpose the image
    ----------------------------------------------------------------------------------------------------- """
    if isinstance( img, np.ndarray ):
        img = array_to_image( img, rgb=rgb )

    if resize is not None:
        assert isinstance( resize, tuple ), "Invalid resize argument {}".format( resize )
        img = img.resize( resize )

    if transpose:
        img = img.transpose( method=Image.TRANSPOSE )       # NOTE that the argument depends on the PIL version

    img.save( fname )



def draw_angle( img, angle1, angle2=None, angle3=None, resize=None ):
    """ -----------------------------------------------------------------------------------------------------
    draw an angle, or optionally two, or three, into a given image
    the convention is to start at the Y axis and turn counterclockwise

    img:            [numpy.ndarray or PIL.Image.Image] image
    angle1:         [float] first angle in degrees
    angle2:         [float] second angle in degrees
    ----------------------------------------------------------------------------------------------------- """
    if isinstance( img, np.ndarray ):
        img = array_to_image( img, remove_channel=True )

    if resize is not None:
        assert isinstance( resize, tuple ), "Invalid resize argument {}".format( resize )
        img = img.resize( resize )

    img     = img.convert( mode="RGB" )
    angle1  = angle1 % 360

    w, h    = img.size
    w0      = w // 2
    h0      = h // 2
    dw      = w0 // 2
    dh      = h0 // 2
    box     = w0 - dw, h0 - dh, w0 + dw, h0 + dh        # the arc funcion requires a bounding box...
    box1    = w0 - dw + 4, h0 - dh + 4, w0 + dw - 4, h0 + dh - 4
    box2    = w0 - dw - 4, h0 - dh - 4, w0 + dw + 4, h0 + dh + 4
    arc0    = 270                                       # the starting angle for drawing arcs
    drw     = ImageDraw.Draw( img )

    ax      = [ ( 0, h0 ), ( w - 1, h0 ) ]
    ay      = [ ( w0, 0 ), ( w0, h0 - 1 ) ]
    drw.line( ax, fill="blue", width = 2 )
    drw.line( ay, fill="blue", width = 2 )

    y       = int( w0 * sin ( pi * ( angle1 / 180 - 0.5 ) ) )
    x       = int( w0 * cos ( pi * ( angle1 / 180 - 0.5 ) ) )
#   line    = [ ( w0 - x, h0 - y ), ( w0 + x, h0 + y ) ]
    line    = [ ( w0, h0 ), ( w0 + x, h0 + y ) ]
    drw.line( line, fill="green", width=3 )
    if angle2 is None:
        drw.arc( box, arc0, 270 + angle1, fill="green", width=3 )

    if angle2 is not None:
        angle2  = angle2 % 360
        y       = int( 0.9 * w0 * sin ( pi * ( angle2 / 180 - 0.5 ) ) )
        x       = int( 0.9 * w0 * cos ( pi * ( angle2 / 180 - 0.5 ) ) )
#       line    = [ ( w0 - x, h0 - y ), ( w0 + x, h0 + y ) ]
        line    = [ ( w0, h0 ), ( w0 + x, h0 + y ) ]
        drw.line( line, fill="red", width=3 )
        drw.arc( box1, arc0, 270 + angle2, fill="red", width=3 )

    if angle3 is not None:
        angle3  = angle3 % 360
        y       = int( 0.8 * w0 * sin ( pi * ( angle3 / 180 - 0.5 ) ) )
        x       = int( 0.8 * w0 * cos ( pi * ( angle3 / 180 - 0.5 ) ) )
#       line    = [ ( w0 - x, h0 - y ), ( w0 + x, h0 + y ) ]
        line    = [ ( w0, h0 ), ( w0 + x, h0 + y ) ]
        drw.line( line, fill="yellow", width = 4 )
        drw.arc( box2, arc0, 270 + angle3, fill="yellow", width=3 )

    return img



def save_collage( imgs, w, h, fname, pad_size=5, pad_color="#a08060" ):
    """ -----------------------------------------------------------------------------------------------------
    Combine multiple sequences of images into a collage
    typically the upper row is the target seqnece, the lower rows are predictions of the sequence

    imgs:           [tuple] of lists (at least 2) of PIL.Image.Image, first list goes in the upper row
    w:              [int] desired width of single image tile inside the collage
    h:              [int] desired height of single image tile inside the collage
    fname:          [str] path of output file
    pad_size:       [int] pixels between image tiles
    pad_color:      [str] padding color
    ----------------------------------------------------------------------------------------------------- """
    n_rows  = len( imgs )
    n_cols  = len( imgs[ 0 ] )

    width   = n_cols * w + ( n_cols - 1 ) * pad_size
    height  = n_rows * h + ( n_rows - 1 ) * pad_size

    i       = 0
    img     = Image.new( 'RGB', ( width, height ), color=pad_color )
    
    for r in range( n_rows ):
        y   = r * ( h + pad_size )
        for c in range( n_cols ):
            x   = c * ( w + pad_size )
            img.paste( imgs[ r ][ c ].resize( ( w, h ) ), ( x, y ) )

    img.save( fname )



def save_animation( imgs, w, h, fname, pad_size=5, pad_col="#aa0000", pad_col_pred="#00aa00", split_col=None ):
    """ -----------------------------------------------------------------------------------------------------
    Combine two sequences of images into an animated gif
    typically the upper image is the target, the lower image is the predicted one

    imgs:           [tuple] of two lists of PIL.Image.Image, first list goes in the upper row
    w:              [int] desired width of single image tile inside the collage
    h:              [int] desired height of single image tile inside the collage
    fname:          [str] path of output file
    pad_size:       [int] pixels between image tiles
    pad_col:        [str] padding color
    pad_col_pred:   [str] padding color for predicted frames
    split_col:      [int] number of frames in which pad_color is used, then
    ----------------------------------------------------------------------------------------------------- """
    n_rows      = len( imgs )
    n_frames    = len( imgs[ 0 ] )

    width       = w + 2 * pad_size
    height      = n_rows * h + ( n_rows + 1 ) * pad_size

    frames      = []

    
    color   = pad_col
    for f in range( n_frames ):
        if split_col is not None:
            if f > split_col:
                color   = pad_col_pred
        img     = Image.new( 'RGB', ( width, height ), color=color )
        for r in range( n_rows ):
            img.paste( imgs[ r ][ f ].resize( ( w, h ) ), ( pad_size, ( r * h ) + ( r + 1 ) * pad_size ) )
        frames.append( img )

    frames[ 0 ].save( fname, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0 )
