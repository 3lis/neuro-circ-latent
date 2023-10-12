SEED = 1
"""
#####################################################################################################################

    Generation of images and dataset for rotation model

    Alice   2022

    this script can be executed standalone or imported as module
    the execution as main typically produces all the single images

#####################################################################################################################
"""
import  os
import  numpy               as np
import  tensorflow          as tf

from    PIL                 import Image, ImageDraw
from    math                import sin, cos, sqrt, pi
from    tensorflow.keras    import utils
from    tensorflow.keras.preprocessing  import image

from    img_utils           import file_to_array, image_to_array, draw_angle

#####################################################################################################################
# common globals
#####################################################################################################################
isize       = ( 128, 128 )                          # image size
n_rot       = 72                                    # number of rotations (should be even), set by the main
n_tra       =  9                                    # number of translations (should be square)
bsline      = False                                 # generate data for baseline model, set by the main
shift_mat   = []                                    # list of all shift matrices, will be validated by shifts()
DO_NOTHING  = True                                  # for debugging


#####################################################################################################################
# globals for internal generated shapes
#####################################################################################################################
o_size      = 0.25                                  # object size in scale [0..1]
o_offset    = np.array( [ 0.1, 0.1 ] )              # range of possible object translations
background  = 20                                    # background gray level
foreground  = 250                                   # object filling gray level
edgecolor   = 250                                   # object edge gray level
img_dir     = "../imgs"                             # directory with images stored in files
o_classes   = (                                     # possible object classes [NOTE: care memory!]
        "tri",
        "square",
        "arrow",
        "ell",
        "jay",
        "zig",
#       "zag",
#       "seven",
#       "iseven",
#       "bump",
        "hole"
)
v_classes   = (                                     # object classes reserved for validation
        "jay",
        "zag"
)


#####################################################################################################################
# globals for coil images
#####################################################################################################################
coil_dir    = "../data/coil/coil-24"                # directory with coil images
color       = False                                 # image type, should be set by main_exec
coil_dict   = {
        "rolaids":	"obj05-000",
        "cup":		"obj10-165",
        "kitty":	"obj14-000",
        "cupflower":	"obj16-035",
        "car":		"obj23-320",
        "frog":		"obj28-245",
        "fancyfeast":	"obj32-000",
        "wood":		"obj41-135",
        "cupgreen":	"obj43-140",
        "spoon":	"obj44-060",
        "pig":		"obj48-000",
        "house":	"obj51-035",
        "toast":	"obj53-140",
        "advil":	"obj54-035",
        "cuporange":	"obj59-000",
        "telephone":	"obj60-020",
        "justforcopies":"obj61-000",
        "philadelphia":	"obj72-000",
        "burger":	"obj73-110",
        "duck":		"obj74-355",
        "anacin":	"obj79-055",
        "cupsmall":	"obj89-000",
        "jag":		"obj97-180",
        "nuclearwaste":	"obj99-000"
}

# possible object classes
# NOTE: should be in alphabetic order
# best practice is to comment out excluded objects
# number of possible objects are constrained by memory, in hp computer:
#   - 64x64, graylevel, every_rot=1:   10 objects
#   - 128x128, graylevel, every_rot=4: 10 objects
#   - 128x128, RGB, every_rot=4:        7 objects
coil_o_classes   = (
#       "advil",
#       "anacin",
#       "burger",
        "car",
#       "cupflower",
#       "cupgreen",
#       "cup",
#       "cuporange",
#       "cupsmall",
#       "duck",
#       "fancyfeast",
#       "frog",
        "house",
#       "jag",
#       "justforcopies",
#       "kitty",
#       "nuclearwaste",
#       "philadelphia",
#       "pig",
#       "rolaids",
#       "spoon",
#       "telephone",
#       "toast",
#       "wood"
)
coil_v_classes   = (                                # object classes reserved for validation
#       "car",
#       "duck",
#       "frog",
        "house",
#       "jag",
#       "telephone",
)
coil_t_classes  = (                                 # object for testing - take all for latent analysis
        "advil",
        "anacin",
        "burger",
        "car",
        "cupflower",
        "cupgreen",
        "cup",
        "cuporange",
        "cupsmall",
        "duck",
        "fancyfeast",
        "frog",
        "house",
        "jag",
        "justforcopies",
        "kitty",
        "nuclearwaste",
        "philadelphia",
        "pig",
        "rolaids",
        "spoon",
        "telephone",
        "toast",
        "wood"
)




#####################################################################################################################
# globals for amazon images
#####################################################################################################################
amazon_cam      = None                                      # code of camera [1-5] set in config
amazon_fmt      = "../data/amazon/amazon-N{:01d}"           # fromat of the directory with amazon images
color           = False                                     # image type, should be set by main_exec
amazon_dict     = (
        "champion_copper_plus_spark_plug",
        "cheezit_big_original",
        "crayola_64_ct",
        "dove_beauty_bar",
        "dr_browns_bottle_brush",
        "elmers_washable_no_run_school_glue",
        "expo_dry_erase_board_eraser",
        "feline_greenies_dental_treats",
        "first_years_take_and_toss_straw_cups",
        "genuine_joe_plastic_stir_sticks",
        "highland_6539_self_stick_notes",
        "kong_air_dog_squeakair_tennis_ball",
        "kong_duck_dog_toy",
        "kong_sitting_frog_dog_toy",
        "kygen_squeakin_eggs_plush_puppies",
        "laugh_out_loud_joke_book",
        "mark_twain_huckleberry_finn",
        "mead_index_cards",
        "mommys_helper_outlet_plugs",
        "munchkin_white_hot_duck_bath_toy",
        "one_with_nature_soap_dead_sea_mud",
        "oreo_mega_stuf",
        "paper_mate_12_count_mirado_black_warrior",
        "rollodex_mesh_collection_jumbo_pencil_cup",
        "safety_works_safety_glasses",
        "sharpie_accent_tank_style_highlighters",
        "stanley_66_052"
)

# possible object classes
# NOTE: should be in alphabetic order
# best practice is to comment out excluded objects
amazon_o_classes    = (
        "champion_copper_plus_spark_plug",
        "cheezit_big_original",
        "crayola_64_ct",
        "dove_beauty_bar",
        "dr_browns_bottle_brush",
        "elmers_washable_no_run_school_glue",
        "expo_dry_erase_board_eraser",
        "feline_greenies_dental_treats",
        "first_years_take_and_toss_straw_cups",
        "genuine_joe_plastic_stir_sticks",
        "highland_6539_self_stick_notes",
        "kong_air_dog_squeakair_tennis_ball",
        "kong_duck_dog_toy",
        "kong_sitting_frog_dog_toy",
        "kygen_squeakin_eggs_plush_puppies",
        "laugh_out_loud_joke_book",
        "mark_twain_huckleberry_finn",
        "mead_index_cards",
        "mommys_helper_outlet_plugs",
        "munchkin_white_hot_duck_bath_toy",
        "one_with_nature_soap_dead_sea_mud",
        "oreo_mega_stuf",
        "paper_mate_12_count_mirado_black_warrior",
        "rollodex_mesh_collection_jumbo_pencil_cup",
        "safety_works_safety_glasses",
        "sharpie_accent_tank_style_highlighters",
        "stanley_66_052"
)
amazon_v_classes    = (                                # object classes reserved for validation
        "champion_copper_plus_spark_plug",
        "cheezit_big_original",
        "kong_air_dog_squeakair_tennis_ball",
        "oreo_mega_stuf",
        "paper_mate_12_count_mirado_black_warrior",
        "rollodex_mesh_collection_jumbo_pencil_cup",
        "stanley_66_052"
)
amazon_t_classes    = amazon_v_classes                 # object classes reserved for testing



#####################################################################################################################
# functions for generating shapes
#####################################################################################################################

def draw_object( xy, fname=None ):
    """ -------------------------------------------------------------------------------------------------------------
    draw one object, and optionally write it on file

    xy:             [list] [ ( p1x, p1y ), ( p2x, p2y ), ( p3x, p3y ) ]
    fname:          [str] filename or None for no output

    return:         [np.array] with shape isize

    ------------------------------------------------------------------------------------------------------------- """

    img     = Image.new( 'L', isize, color=background )
    drw     = ImageDraw.Draw( img )
    drw.polygon( xy, fill=foreground, outline=edgecolor, width=2 )

    if fname is not None:
        img.save( fname )

    return image_to_array( img )


def scale( points, off_center=np.zeros( 2 ) ):
    """ -------------------------------------------------------------------------------------------------------------
    scale points from real to image coordinates and optionally translate from the center

    points:         [np.array] with shape ( N, 2 )
    off_center:     [np.array] with shape (2, )

    return:         [list] of np.array with shape ( 2, )

    ------------------------------------------------------------------------------------------------------------- """

    fact    = np.array( isize )
    scaled  = []
    center  = ( 0.5, 0.5 ) + off_center

    for p in points:
        p   = fact * ( p + center )                    # scale up to pixels
        scaled.append( p.astype( int ) )

    return scaled


def translate( i ):
    """ -------------------------------------------------------------------------------------------------------------
    compute the amount of translation from the index, that should span from 0 to n_tra

    i:              [int] index of the translation

    return:         [np.array] with shape ( 2 )

    ------------------------------------------------------------------------------------------------------------- """

    assert i < n_tra, "Error, translation index {} too large".format( i )
    tra_0   = int( sqrt( n_tra ) )                  # number of translation intervals per dimension
    x       = i % tra_0                             # translation index in the X dimension
    y       = i // tra_0                            # translation index in the Y dimension
    d_off   = o_offset / ( tra_0 - 1 )              # minimum translation span
    c_off   = o_offset / 2                          # center of translation
    tra     = ( x, y ) * d_off - c_off              # translation for the current index
    return tra


def rotate( points, a ):
    """ -------------------------------------------------------------------------------------------------------------
    rotate an array of 2D points coordinated of given angle a

    points:         [np.array] with shape ( X, 2 )
    a:              [float] angle in radiants

    return:         [np.array] with shape ( X, 2 )

    ------------------------------------------------------------------------------------------------------------- """

    assert points.shape[ -1 ] == 2, "Error: inconsistent shape of input points"
    c       = cos( a )
    s       = sin( a )
    rot     = np.array( [ [ c, -s ], [ s, c ] ] )
    return np.dot( points, rot )


def img_rot_transl( points, i_rot, i_trans ):
    """ -------------------------------------------------------------------------------------------------------------
    return an array with rotation and translation of the drawn object, with additional channel dimension

    points:         [np.array] with shape ( X, 2 )
    i_rot:          [int] index of rotation
    i_trans         [int] index of translation

    return:         [np.array] with shape ( *isize, 1 )

    ------------------------------------------------------------------------------------------------------------- """

    angle   = 2 * pi / n_rot
    off_center  = translate( i_trans )
    pts     = rotate( points, i_rot * angle )
    pts     = scale( pts, off_center=off_center )
    xy      = [ tuple( p ) for p in pts ]
    img     = draw_object( xy )
    img     = img[ ..., np.newaxis ]
    return img


def gen_square():
    """ -------------------------------------------------------------------------------------------------------------
    generate a square

    return:         [np.array] with shape ( 4, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    p1          = np.array( [  o_size,  o_size ] )
    p2          = np.array( [  o_size, -o_size ] )
    p3          = np.array( [ -o_size, -o_size ] )
    p4          = np.array( [ -o_size,  o_size ] )

    return np.array( [ p1, p2, p3, p4 ] )


def gen_tri():
    """ -------------------------------------------------------------------------------------------------------------
    generate equilateral triangle

    return:         [np.array] with shape ( 3, 2 )
    ------------------------------------------------------------------------------------------------------------- """

    a120        = 2 * pi / 3
    p1          = np.array( [ 0., o_size ] )
    p2          = rotate( p1, a120 )
    p3          = rotate( p1, - a120 )

    return np.array( [ p1, p2, p3 ] )


def gen_arrow():
    """ -------------------------------------------------------------------------------------------------------------
    generate an arrow shape

    return:         [np.array] with shape ( 4, 2 )
    ------------------------------------------------------------------------------------------------------------- """

    a120        = 2 * pi / 3
    p1          = np.array( [ 0., o_size ] )
    p2          = rotate( p1, a120 )
    p3          = np.zeros( 2 )
    p4          = rotate( p1, - a120 )

    return np.array( [ p1, p2, p3, p4 ] )


def gen_ell():
    """ -------------------------------------------------------------------------------------------------------------
    generate a "L" shaped object

    return:         [np.array] with shape ( 6, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    h_size      = 0.5 * o_size
    p1          = np.array( [ -h_size, -o_size ] )
    p2          = np.array( [ -h_size,  h_size ] )
    p3          = np.array( [  h_size,  h_size ] )
    p4          = np.array( [  h_size,  0      ] )
    p5          = np.array( [       0,  0      ] )
    p6          = np.array( [       0, -o_size ] )

    return np.array( [ p1, p2, p3, p4, p5, p6 ] )


def gen_jay():
    """ -------------------------------------------------------------------------------------------------------------
    generate a "J" shaped object

    return:         [np.array] with shape ( 6, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    ell         = gen_ell()
    jay         = np.array( [ -1., 1 ] ) * ell

    return jay


def gen_hole():
    """ -------------------------------------------------------------------------------------------------------------
    generate a square with hole shaped object

    return:         [np.array] with shape ( 8, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    h_size      = 0.5 * o_size
    p1          = np.array( [ -h_size,  o_size ] )
    p2          = np.array( [ -h_size, -h_size ] )
    p3          = np.array( [  h_size, -h_size ] )
    p4          = np.array( [  h_size,  0      ] )
    p5          = np.array( [       0,  0      ] )
    p6          = np.array( [       0,  h_size ] )
    p7          = np.array( [  h_size,  h_size ] )
    p8          = np.array( [  h_size,  o_size ] )

    return np.array( [ p1, p2, p3, p4, p5, p6, p7, p8 ] )


def gen_bump():
    """ -------------------------------------------------------------------------------------------------------------
    generate a square with bump shaped object

    return:         [np.array] with shape ( 8, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    h_size      = 0.5 * o_size
    o_h         = h_size + o_size
    p1          = np.array( [ -h_size,  o_size ] )
    p2          = np.array( [ -h_size, -h_size ] )
    p3          = np.array( [  h_size, -h_size ] )
    p4          = np.array( [  h_size,  0      ] )
    p5          = np.array( [     o_h,  0      ] )
    p6          = np.array( [     o_h,  h_size ] )
    p7          = np.array( [  h_size,  h_size ] )
    p8          = np.array( [  h_size,  o_size ] )

    return np.array( [ p1, p2, p3, p4, p5, p6, p7, p8 ] )


def gen_zig():
    """ -------------------------------------------------------------------------------------------------------------
    generate a zig shaped object

    return:         [np.array] with shape ( 9, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    h_size      = 0.5 * o_size
    p1          = np.array( [       0,  o_size ] )
    p2          = np.array( [       0,  h_size ] )
    p3          = np.array( [ -h_size,       0 ] )
    p4          = np.array( [ -h_size, -h_size ] )
    p5          = np.array( [  o_size, -h_size ] )
    p6          = np.array( [  o_size,       0 ] )
    p7          = np.array( [       0,       0 ] )
    p8          = np.array( [  h_size,  h_size ] )
    p9          = np.array( [  h_size,  o_size ] )

    return np.array( [ p1, p2, p3, p4, p5, p6, p7, p8, p9 ] )


def gen_zag():
    """ -------------------------------------------------------------------------------------------------------------
    generate a zag shaped object

    return:         [np.array] with shape ( 9, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    zig         = gen_zig()
    zag         = np.array( [ -1., 1 ] ) * zig

    return zag


def gen_seven():
    """ -------------------------------------------------------------------------------------------------------------
    generate a "7" shaped object

    return:         [np.array] with shape ( 6, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    h_size      = 0.5 * o_size
    o_h         = h_size + o_size
    p1          = np.array( [ -h_size,  h_size ] )
    p2          = np.array( [       0,  h_size ] )
    p3          = np.array( [     o_h, -o_size ] )
    p4          = np.array( [ -h_size, -o_size ] )
    p5          = np.array( [ -h_size, -h_size ] )
    p6          = np.array( [  h_size, -h_size ] )

    return np.array( [ p1, p2, p3, p4, p5, p6 ] )


def gen_iseven():
    """ -------------------------------------------------------------------------------------------------------------
    generate an inverted "7" shaped object

    return:         [np.array] with shape ( 6, 2 )
    ------------------------------------------------------------------------------------------------------------- """
    seven      = gen_seven()
    iseven     = np.array( [ -1., 1 ] ) * seven

    return iseven


def gen_imgs( obj ):
    """ -------------------------------------------------------------------------------------------------------------
    generate files with object images
    obj:            [str] the class of object
    ------------------------------------------------------------------------------------------------------------- """

    assert obj in o_classes, "Error: no drawing function available for object {}".format( obj )
    generator   = "gen_{}()".format( obj )
    points      = eval( generator )
    img_fmt     = "{}_{:03d}_{:02d}.jpg"             # format of a filename
    angle       = 2 * pi / n_rot

    for t in range( n_tra ):
        off_center  = translate( t )
        for r in range( n_rot ):
            pts     = rotate( points, r * angle )
            pts     = scale( pts, off_center=off_center )
            xy      = [ tuple( p ) for p in pts ]
            fname   = img_fmt.format( obj, r, t )
            fname   = os.path.join( img_dir, fname )
            draw_object( xy, fname=fname )


def gen_imgs_rot( obj ):
    """ -------------------------------------------------------------------------------------------------------------
    generate files with object images without translation, and with axes and angle line
    obj:            [str] the class of object
    ------------------------------------------------------------------------------------------------------------- """

    assert obj in o_classes, "Error: no drawing function available for object {}".format( obj )
    generator   = "gen_{}()".format( obj )
    points      = eval( generator )
    img_fmt     = "{}_{:03d}.jpg"             # format of a filename
    angle       = 2 * pi / n_rot
    degree      = 360 / n_rot

    for r in range( n_rot ):
        pts     = rotate( points, r * angle )
        pts     = scale( pts )
        xy      = [ tuple( p ) for p in pts ]
        fname   = img_fmt.format( obj, r )
        fname   = os.path.join( img_dir, fname )
        img     = draw_object( xy )
        img     = draw_angle( img, r * degree, resize=( 256, 256 ) )
        img.save( fname )

def gen_all_imgs():
    """ -------------------------------------------------------------------------------------------------------------
    generate files with object images for all classes
    ------------------------------------------------------------------------------------------------------------- """
    for obj in o_classes:
        print( "doing object " + obj )
        gen_imgs_rot( obj )
        gen_imgs( obj )



#####################################################################################################################
# functions for reading coil images
#####################################################################################################################

def read_coil( obj, rot ):
    """ -------------------------------------------------------------------------------------------------------------
    read a coil file
    obj:            [str] the class of object
    rot:            [int] rotation index
    ------------------------------------------------------------------------------------------------------------- """
    gray    = not color
    angle   = int( 360 * rot / n_rot )
    fn      = "{}__{:03d}.png".format( coil_dict[ obj ], angle )
    fname   = os.path.join( coil_dir, fn )
    img     = file_to_array( fname, remove_channel=gray, resize=isize )
    if gray:
        img     = img[ ..., np.newaxis ]
    return img


def read_coils( training=True ):
    """ -------------------------------------------------------------------------------------------------------------
    read all necessary coil files
    ------------------------------------------------------------------------------------------------------------- """
    imgs    = []
    objects = coil_o_classes if training else coil_t_classes

    for rot in range( n_rot ):
        imgs.append( {} )

    for obj in objects:
        for rot in range( n_rot ):
            imgs[ rot ][ obj ]  = read_coil( obj, rot )

    return imgs


def read_keras_coil( obj, rot ):
    """ -------------------------------------------------------------------------------------------------------------
    read a coil file using keras functions
    obj:            [str] the class of object
    rot:            [int] rotation index
    ------------------------------------------------------------------------------------------------------------- """
    color_mode  = "rgb" if color  else "grayscale"
    angle       = int( 360 * rot / n_rot )
    fn          = "{}__{:03d}.png".format( coil_dict[ obj ], angle )
    fname       = os.path.join( coil_dir, fn )
    img         = utils.load_img( fname, color_mode=color_mode, target_size=isize )
    img         = utils.img_to_array( img )
    img         = img / 255.
    return img



#####################################################################################################################
# functions for reading amazon images
#####################################################################################################################

def read_amazon( obj, rot ):
    """ -------------------------------------------------------------------------------------------------------------
    read a amazon file
    obj:            [str] the class of object
    rot:            [int] rotation index
    ------------------------------------------------------------------------------------------------------------- """
    fmt     = "{:03d}_{:d}_{:03d}.jpg"
    gray    = not color
    angle   = int( 360 * rot / n_rot )
    obj_idx = amazon_dict.index( obj ) + 1
    fn      = fmt.format( obj_idx, amazon_cam, angle )
    fname   = os.path.join( amazon_fmt.format( amazon_cam ), fn )
    img     = file_to_array( fname, remove_channel=gray, resize=isize )
    if gray:
        img     = img[ ..., np.newaxis ]
    return img


def read_amazons( training=True ):
    """ -------------------------------------------------------------------------------------------------------------
    read all necessary amazon files
    ------------------------------------------------------------------------------------------------------------- """
    imgs    = []
    objects = amazon_o_classes if training else amazon_t_classes

    for rot in range( n_rot ):
        imgs.append( {} )

    for obj in objects:
        for rot in range( n_rot ):
            imgs[ rot ][ obj ]  = read_amazon( obj, rot )

    return imgs


def read_keras_amazon( obj, rot ):
    """ -------------------------------------------------------------------------------------------------------------
    read a amazon file using keras functions
    obj:            [str] the class of object
    rot:            [int] rotation index
    ------------------------------------------------------------------------------------------------------------- """
    color_mode  = "rgb" if color  else "grayscale"
    fmt         = "{:03d}_{:d}_{:03d}.jpg"
    angle       = int( 360 * rot / n_rot )
#   print( "object: " + obj )
    obj_idx     = amazon_dict.index( obj ) + 1
    fn          = fmt.format( obj_idx, amazon_cam, angle )
    fname       = os.path.join( amazon_fmt.format( amazon_cam ), fn )
    img         = utils.load_img( fname, color_mode=color_mode, target_size=isize )
    img         = utils.img_to_array( img )
    img         = img / 255.
    return img


def read_baseline_amazon( obj, rot ):
    """ -------------------------------------------------------------------------------------------------------------
    read a amazon file using keras functions, clean and simple for baseline CNN
    obj:            [str] the class of object
    rot:            [int] rotation index
    ------------------------------------------------------------------------------------------------------------- """
    color_mode  = "rgb" if color  else "grayscale"
    fmt         = "{:03d}_{:d}_{:03d}.jpg"
    angle       = int( 360 * rot / n_rot )
#   print( "object: " + obj )
    obj_idx     = amazon_dict.index( obj ) + 1
    fn          = fmt.format( obj_idx, amazon_cam, angle )
    fname       = os.path.join( amazon_fmt.format( amazon_cam ), fn )
    img         = image.load_img( fname, target_size=isize )
    img         = image.img_to_array( img )
#   img         = np.expand_dims( img, axis=0 )
    return img



#####################################################################################################################
# functions for reading generic images
#####################################################################################################################

def read_generic( obj, rot, dataset ):
    """ -------------------------------------------------------------------------------------------------------------
    read a generic file in the given dataset
    obj:            [str] the class of object
    rot:            [int] rotation index
    dataset         [str] dataset code, see load_cnfg.py for details of the currently available dataset codes
    ------------------------------------------------------------------------------------------------------------- """
    if 'coil' in dataset or 'COIL' in dataset:
        dset    = 'coil'
    if 'amazon' in dataset or 'AMAZON' in dataset:
        dset    = 'amazon'
    img         = eval( f"read_{dset}( '{obj}', {rot} )" )
    return img


#####################################################################################################################
# functions for dataset generation
#####################################################################################################################


def shifts():
    """ -------------------------------------------------------------------------------------------------------------
    generate all the shift operation matrices to be applied to a latent space, for all possible rotations
    see Bouchacourt et al. 2021 p.7 equations (4) and (5)

    NOTE: ensure dtype='float32' for compatibility with Keras tensor operations
    ------------------------------------------------------------------------------------------------------------- """

    ones            = np.ones( n_rot, dtype='float32' )
    shift0          = np.diag( ones )           # generate a diagonal matrix as 0 rotation
    shift_mat.append( shift0 )
    ones            = np.ones( n_rot -1, dtype='float32' )
    shift1          = np.diag( ones, k=-1 )     # generate a matrix with a diagonal one place lower that the main one
    shift1[ 0, -1 ] = 1.                        # complete the vector shift with the first element that become last
    shift           = shift1                    # this is the matrix for the first rotation
    shift_mat.append( shift1 )
    for r in range( n_rot - 2 ):                # progressive powers of the basic matrix
        shift       = np.dot( shift, shift1 )
        shift_mat.append( shift )


def shift( i_rot1, i_rot2 ):
    """ -------------------------------------------------------------------------------------------------------------
    get the shift operation matrix that align from i_rot1 to i_rot2
    requires a valid shift_mat list (as generated by shifts() )
    i_rot1          [int] first rotation index
    i_rot2          [int] second rotation index
    ------------------------------------------------------------------------------------------------------------- """

    assert len( shift_mat ), "Error, shift_mat is not validated"
    r   = i_rot2 - i_rot1
    if r < 0:
        r   = n_rot + r             # the operation is cyclic
    assert r < n_rot, "cannot shift for rotation index {}".format( r )
    return shift_mat[ r ]


def rot_1hot( i_rot1, i_rot2 ):
    """ -------------------------------------------------------------------------------------------------------------
    return a one-hot vector with 1 at the ratation between the two given angles
    i_rot1          [int] first rotation index
    i_rot2          [int] second rotation index
    ------------------------------------------------------------------------------------------------------------- """

    r           = i_rot2 - i_rot1
    if r < 0:
        r       = n_rot + r             # the operation is cyclic
    assert r < n_rot, "cannot shift for rotation index {}".format( r )
    rot         = np.zeros( n_rot, dtype='float32' )
    rot[ r ]    = 1.0
    return rot


def rot_target( i_rot1, i_rot2 ):
    """ -------------------------------------------------------------------------------------------------------------
    return a target for baseline CNN as sin and cos of the angle
    i_rot1          [int] first rotation index
    i_rot2          [int] second rotation index
    ------------------------------------------------------------------------------------------------------------- """

    r           = i_rot2 - i_rot1
    if r < 0:
        r       = n_rot + r             # the operation is cyclic
    assert r < n_rot, "cannot shift for rotation index {}".format( r )
    a       = 2 * pi * r / n_rot
    c       = cos( a )
    s       = sin( a )
    rot     = np.array( [ s, c ], dtype='float32' )
    return rot


def gen_inner_dset( every_rot=4, train_val=2 ):
    """ -------------------------------------------------------------------------------------------------------------
    generate the full dataset using internal generated images
    every_rot       [int] how many rotations to skip as second target, with 1 the dataset is very large
    train_val       [int] the ratio between training and validation sizes, with 2 the training set is twice larger
                    if None then a v_classes variable should eixist, with the objects to be used for validation

    return:         [tuple] train, val, tr_desc, vl_desc, where train/val are [dict] with keys 'x', 'y'
                    'x' value is a list [ x1, x2 ], and 'y' value is a list [ y1, y2 ]
            x1      [np.array] of shape ( n_samples, *isize )
            x2      [np.array] of shape ( n_samples, n_rot, n_rot )
            y1      [np.array] of shape ( n_samples, *isize )
            y2      [np.array] same as x1
                    and tr_desc, vl_desc are lists with tuple ( obj, rot1, rot2, transl )

    ------------------------------------------------------------------------------------------------------------- """
    tr_x1   = []
    tr_x2   = []
    tr_y    = []
    tr_desc = []
    vl_x1   = []
    vl_x2   = []
    vl_y    = []
    vl_desc = []
    tr_set  = {}
    vl_set  = {}

    print( "generating dataset with internal shapes\n" )

    for obj in o_classes:
        generator   = "gen_{}()".format( obj )
        points      = eval( generator )
        train       = True
        tr_count    = 1
        for rot1 in range( n_rot ):
            if train_val is None:
                train       = obj not in v_classes
            elif tr_count > train_val:            # reserve this starting rotation for validation
                train       = False
                tr_count    = 0
            for transl in range( n_tra ):
                img1    = img_rot_transl( points, rot1, transl )
                for rot2 in range( 0, n_rot, every_rot ):
                    img2    = img_rot_transl( points, rot2, transl )
                    mat_sh  = shift( rot1, rot2 )
                    if train:
                        tr_x1.append( img1 )
                        tr_x2.append( mat_sh )
                        tr_y.append( img2 )
                        tr_desc.append( ( obj, rot1, rot2, transl ) )
                    else:
                        vl_x1.append( img1 )
                        vl_x2.append( mat_sh )
                        vl_y.append( img2 )
                        vl_desc.append( ( obj, rot1, rot2, transl ) )
            if train_val is not None:
                train       = True
                tr_count    += 1

    tr_set[ 'x' ] = [ np.array( tr_x1 ), np.array( tr_x2 ) ]
    tr_set[ 'y' ] = [ np.array( tr_x1 ), np.array( tr_y ) ]
    vl_set[ 'x' ] = [ np.array( vl_x1 ), np.array( vl_x2 ) ]
    vl_set[ 'y' ] = [ np.array( vl_x1 ), np.array( vl_y ) ]

    return tr_set, vl_set, tr_desc, vl_desc


def gen_extern_dset( every_rot=4, train_val=2, dset="coil" ):
    """ -------------------------------------------------------------------------------------------------------------
    generate the full dataset using external images, laded in internal memory
    every_rot       [int] how many rotations to skip as second target, with 1 the dataset is very large
    train_val       [int] the ratio between training and validation sizes, with 2 the training set is twice larger
                    if None then a v_classes variable should exist, with the objects to be used for validation
    dset            [str] code name for the daset, possible values are "coil", "amazon"
    return:         [tuple] train, val, tr_desc, vl_desc, where train/val are [dict] with keys 'x', 'y'
                    'x' value is a list [ x1, x2 ], and 'y' value is a list [ y1, y2 ]
            x1      [np.array] of shape ( n_samples, *isize )
            x2      [np.array] of shape ( n_samples, n_rot, n_rot )
            y1      [np.array] of shape ( n_samples, *isize )
            y2      [np.array] same as x1
                    and tr_desc, vl_desc are lists with tuple ( obj, rot1, rot2, transl )

    ------------------------------------------------------------------------------------------------------------- """
    tr_x1   = []
    tr_x2   = []
    tr_y    = []
    tr_desc = []
    vl_x1   = []
    vl_x2   = []
    vl_y    = []
    vl_desc = []
    tr_set  = {}
    vl_set  = {}
    transl  = 0     # translation is not a variable taken into accout for coil images, so far


    if color:
        print( f"generating dataset with {dset} RGB external images\n" )
    else:
        print( f"generating dataset with {dset} graylevel external images\n" )

    imgs        = eval( f"read_{dset}s()" )
    o_classes   = eval( f"{dset}_o_classes" )
    v_classes   = eval( f"{dset}_v_classes" )

    for obj in o_classes:
        train       = True
        tr_count    = 1
        for rot1 in range( n_rot ):
            if train_val is None:
                train       = obj not in v_classes
            elif tr_count > train_val:            # reserve this starting rotation for validation
                train       = False
                tr_count    = 0
            img1    = imgs[ rot1 ][ obj ]
            for rot2 in range( 0, n_rot, every_rot ):
                img2    = imgs[ rot2 ][ obj ]
                mat_sh  = shift( rot1, rot2 )
                if train:
                    tr_x1.append( img1 )
                    tr_x2.append( mat_sh )
                    tr_y.append( img2 )
                    tr_desc.append( ( obj, rot1, rot2, transl ) )
                else:
                    vl_x1.append( img1 )
                    vl_x2.append( mat_sh )
                    vl_y.append( img2 )
                    vl_desc.append( ( obj, rot1, rot2, transl ) )
            if train_val is not None:
                train       = True
                tr_count    += 1

    tr_set[ 'x' ] = [ np.array( tr_x1 ), np.array( tr_x2 ) ]
    tr_set[ 'y' ] = [ np.array( tr_x1 ), np.array( tr_y ) ]
    vl_set[ 'x' ] = [ np.array( vl_x1 ), np.array( vl_x2 ) ]
    vl_set[ 'y' ] = [ np.array( vl_x1 ), np.array( vl_y ) ]

    return tr_set, vl_set, tr_desc, vl_desc


def gen_test_set( every_rot=4, train_val=2, dataset="int" ):
    """ -------------------------------------------------------------------------------------------------------------
    generate the test dataset in internal memory
    every_rot       [int] how many rotations to skip as second target, with 1 the dataset is very large
    train_val       [int] the ratio between training and validation sizes, with 2 the training set is twice larger
                    if None then a v_classes variable should eixist, with the objects to be used for validation
    dataset         [str] dataset code

    return:         [tuple] vl_set, vl_desc, vl_set is a [dict] with keys 'x', 'y'
                    'x' value is a list [ x1, x2 ], and 'y' value is a list [ y1, y2 ]
            x1      [np.array] of shape ( n_samples, *isize )
            x2      [np.array] of shape ( n_samples, n_rot, n_rot )
            y1      [np.array] of shape ( n_samples, *isize )
            y2      [np.array] same as x1
            vl_desc is a list with tuple ( obj, rot1, rot2, transl )

    ------------------------------------------------------------------------------------------------------------- """
    if len( shift_mat ) == 0:                   # ensure shift_mat has been validated, otherwise do it
        shifts()

    if dataset == "int":
        _, vl_set, _, vl_desc   = gen_inner_dset( every_rot=every_rot, train_val=train_val )
        return vl_set, vl_desc

    vl_x1   = []
    vl_x2   = []
    vl_y    = []
    vl_desc = []
    vl_set  = {}
    transl  = 0     # translation is not a variable taken into accout for coil images, so far

    if 'coil' in dataset or 'COIL' in dataset:
        dset    = 'coil'
    if 'amazon' in dataset or 'AMAZON' in dataset:
        dset    = 'amazon'
    imgs        = eval( f"read_{dset}s( training=False )" )
    t_classes   = eval( f"{dset}_t_classes" )

    for obj in t_classes:
        for rot1 in range( n_rot ):
            for rot2 in range( 0, n_rot, every_rot ):
                img1    = imgs[ rot1 ][ obj ]
                img2    = imgs[ rot2 ][ obj ]
                mat_sh  = shift( rot1, rot2 )
                vl_x1.append( img1 )
                vl_x2.append( mat_sh )
                vl_y.append( img2 )
                vl_desc.append( ( obj, rot1, rot2, transl ) )

    vl_set[ 'x' ] = [ np.array( vl_x1 ), np.array( vl_x2 ) ]
    vl_set[ 'y' ] = [ np.array( vl_x1 ), np.array( vl_y ) ]

    return vl_set, vl_desc


def gen_onerot_tset( dataset="int" ):
    """ -------------------------------------------------------------------------------------------------------------
    generate a test dataset in internal memory used for latenta nalysis only, with just one image
    dataset         [str] dataset code

    return:         [tuple] vl_set, vl_desc, vl_set is a [dict] with keys 'x', 'y'
                    'x' value is a list [ x1, x2 ], and 'y' value is a list [ y1, y2 ]
            vl_set  [list] of np.arrays of shape ( n_rot, *isize )
            vl_desc is a list with [ obj ]

    ------------------------------------------------------------------------------------------------------------- """
    if dataset == "int":
        print( "Warining: no gen_onerot_tset implementation for internal samples" )
        return None, None

    vl_x1   = []
    vl_x2   = []
    vl_y    = []
    vl_desc = []
    vl_set  = []
    transl  = 0     # translation is not a variable taken into accout for coil images, so far

    dset    = None
    if 'coil' in dataset or 'COIL' in dataset:
        dset    = 'coil'
    if 'amazon' in dataset or 'AMAZON' in dataset:
        dset    = 'amazon'
    if dset is None:
        print( f"Warining: no gen_onerot_tset implementation for {dataset}" )
        return None, None
    imgs        = eval( f"read_{dset}s( training=False )" )
    t_classes   = eval( f"{dset}_t_classes" )

    for obj in t_classes:
        im      = []    
        for rot in range( n_rot ):
            im.append( imgs[ rot ][ obj ] )
        im      = np.array( im )
        vl_set.append( im  )
        vl_desc.append( obj )

    return vl_set, vl_desc



def gen_extern_large_tr_vl( every_rot=4, train_val=2, dset="coil" ):
    """ -------------------------------------------------------------------------------------------------------------
    generate the full dataset using coil external images without internal memory
    with internal split between training and validation, based on rotations
    every_rot       [int] how many rotations to skip as second target, with 1 the dataset is very large
    train_val       [int] the ratio between training and validation sizes, with 2 the training set is twice larger
    dset            [str] code name for the daset, possible values are "coil", "amazon"

    return:         [tuple] train, val, tr_desc, vl_desc, where train/val are tf.data.Dataset
                    and tr_desc, vl_desc are lists with tuple ( obj, rot1, rot2, transl )

    ------------------------------------------------------------------------------------------------------------- """
    tr_desc     = []
    vl_desc     = []
    n_channels  = 3 if color    else 1
    i_shape     = ( *isize, n_channels )
    s_shape     = ( n_rot, n_rot )
    transl      = 0     # translation is not a variable taken into accout for coil images, so far


    if color:
        print( f"generating large dataset with {dset} RGB external images and train/val split by rotations\n" )
    else:
        print( f"generating large dataset with {dset} graylevel external images and train/val split by rotations\n" )

    # strings with functions for reading images appropriate for the given dataset
    reader_1    = f"read_keras_{dset}( obj, rot1 )"
    reader_2    = f"read_keras_{dset}( obj, rot2 )"
    o_classes   = eval( f"{dset}_o_classes" )   # object names for the give dataset

    for obj in o_classes:
        train       = True
        tr_count    = 1
        for rot1 in range( n_rot ):
            if tr_count > train_val:            # reserve this starting rotation for validation
                train       = False
                tr_count    = 0
            for rot2 in range( 0, n_rot, every_rot ):
                if train:
                    tr_desc.append( ( obj, rot1, rot2, transl ) )
                else:
                    vl_desc.append( ( obj, rot1, rot2, transl ) )
                train       = True
                tr_count    += 1

    def gen_train():
        for obj in o_classes:
            train       = True
            tr_count    = 1
            for rot1 in range( n_rot ):
                img1    = eval( reader_1 )
                if tr_count > train_val:
                    train       = False
                    tr_count    = 0
                for rot2 in range( 0, n_rot, every_rot ):
                    if train:
                        img2    = eval( reader_2 )
                        mat_sh  = shift( rot1, rot2 )
                        yield ( img1, mat_sh ), ( img1, img2 )
                    train       = True
                    tr_count    += 1

    def gen_valid():
        for obj in o_classes:
            train       = True
            tr_count    = 1
            for rot1 in range( n_rot ):
                img1    = eval( reader_1 )
                if tr_count > train_val:
                    train       = False
                    tr_count    = 0
                for rot2 in range( 0, n_rot, every_rot ):
                    if not train:
                        img2    = eval( reader_2 )
                        mat_sh  = shift( rot1, rot2 )
                        yield ( img1, mat_sh ), ( img1, img2 )
                    train       = True
                    tr_count    += 1

    tr_set  = tf.data.Dataset.from_generator( gen_train,
            output_signature=(
                (
                    tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                    tf.TensorSpec( shape=s_shape, dtype=tf.float32 )
                ),
                (
                    tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                    tf.TensorSpec( shape=i_shape, dtype=tf.float32 )
                )
            ) )

    vl_set  = tf.data.Dataset.from_generator( gen_valid,
            output_signature=(
                (
                    tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                    tf.TensorSpec( shape=s_shape, dtype=tf.float32 )
                ),
                (
                    tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                    tf.TensorSpec( shape=i_shape, dtype=tf.float32 )
                )
            ) )


    return tr_set, vl_set, tr_desc, vl_desc


def gen_extern_large( objects, every_rot=4, dset="coil" ):
    """ -------------------------------------------------------------------------------------------------------------
    generate the full dataset using external images without internal memory
    objects         [list] list with names of the objects to be included in the dataset
    every_rot       [int] how many rotations to skip as second target, with 1 the dataset is very large
    dset            [str] code name for the daset, possible values are "coil", "amazon"

    return:         [tuple] tf.data.Dataset, list with tuple ( obj, rot1, rot2, transl )

    ------------------------------------------------------------------------------------------------------------- """

    if bsline:
        print( f"generating large dataset with {dset} RGB external images, for a baseline model\n" )
    if color:
        print( f"generating large dataset with {dset} RGB external images\n" )
    else:
        print( f"generating large dataset with {dset} graylevel external imagess\n" )

    # strings with functions for reading images appropriate for the given dataset
    reader_1    = f"read_keras_{dset}( obj, rot1 )"
    reader_2    = f"read_keras_{dset}( obj, rot2 )"

    n_channels  = 3 if color    else 1
    i_shape     = ( *isize, n_channels )
    s_shape     = ( n_rot, n_rot )
    dset_desc   = []
    for obj in objects:
        for rot1 in range( n_rot ):
            for rot2 in range( 0, n_rot, every_rot ):
                dset_desc.append( ( obj, rot1, rot2, 0 ) )

    def gen():
        for obj in objects:
            for rot1 in range( n_rot ):
                img1    = eval( reader_1 )
                for rot2 in range( 0, n_rot, every_rot ):
                    img2    = eval( reader_2 )
                    if bsline:
# for extra-safety, a personalized reading function...
#                       image1  = read_baseline_amazon( obj, rot1 )
#                       image2  = read_baseline_amazon( obj, rot2 )
# NOTE: do not use lists for multiple inputs, always tuples
#                       yield ( image1, image2 ), rot_1hot( rot1, rot2 )
#                       yield ( img1, img2 ), rot_1hot( rot1, rot2 )
# attempt with [sin, cos ]
                        yield ( img1, img2 ), rot_target( rot1, rot2 )
                    else:
                        mat_sh  = shift( rot1, rot2 )
                        yield ( img1, mat_sh ), ( img1, img2 )

    if bsline:
# NOTE: output_signature can be any nested ddcombination of tf.TensorSpec functions, but with tuples only
        output_signature    = (
            (
                tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                tf.TensorSpec( shape=i_shape, dtype=tf.float32 )
            ),
#           tf.TensorSpec( shape=( n_rot, ), dtype=tf.float32 ),
# attempt with [sin, cos ]
            tf.TensorSpec( shape=( 2, ), dtype=tf.float32 ),
        )
    else:
        output_signature    = (
            (
                tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                tf.TensorSpec( shape=s_shape, dtype=tf.float32 )
            ),
            (
                tf.TensorSpec( shape=i_shape, dtype=tf.float32 ),
                tf.TensorSpec( shape=i_shape, dtype=tf.float32 )
            )
        )

    dset    = tf.data.Dataset.from_generator( gen, output_signature=output_signature )

    return dset, dset_desc


def gen_dset( every_rot=4, train_val=2, dataset="int" ):
    """ -------------------------------------------------------------------------------------------------------------
    generate the full training and validation sets, for all possible image datasets
    every_rot       [int] how many rotations to skip as second target, with 1 the dataset is very large
    train_val       [int] the ratio between training and validation sizes, with 2 the training set is twice larger
                    if None then a v_classes variable should exist, with the objects to be used for validation
    dataset         [str] dataset code, that includes information about the source images, use of color, use of memory
                    see load_cnfg.py for details of the currently available dataset codes

    return:         [tuple] train, val, tr_desc, vl_desc,
            where train/val differs depending on whether sets are strored in memory or using tf.data
            in the first case are [dict] with keys 'x', 'y'
                    'x' value is a list [ x1, x2 ], and 'y' value is a list [ y1, y2 ]
            x1      [np.array] of shape ( n_samples, *isize )
            x2      [np.array] of shape ( n_samples, n_rot, n_rot )
            y1      [np.array] of shape ( n_samples, *isize )
            y2      [np.array] same as x1
            tr_desc, vl_desc are lists with tuple ( obj, rot1, rot2, transl )

    ------------------------------------------------------------------------------------------------------------- """
    if len( shift_mat ) == 0:                   # ensure shift_mat has been validated, otherwise do it
        shifts()

    # extract from the dataset code the information for the image sources, and store it in dset
    if 'coil' in dataset or 'COIL' in dataset:
        dset    = 'coil'
    if 'amazon' in dataset or 'AMAZON' in dataset:
        dset    = 'amazon'
    # case of images stored in internal memory
    if dataset in ( "coilBW", "coilRGB", "amazonBW", "amazonRGB" ):
        return gen_extern_dset( every_rot=every_rot, train_val=train_val, dset=dset )

    # case of images accessed by tf.data.Dataset
    if dataset in ( "COILBW", "COILRGB", "AMAZONBW", "AMAZONRGB" ):
        # case of training and validation sets using different objects
        if train_val is None:
            o_classes   = eval( f"{dset}_o_classes" )
            v_classes   = eval( f"{dset}_v_classes" )
            train_objects       = [ o for o in o_classes if o not in v_classes ]
            tr_dset, tr_desc    = gen_extern_large( train_objects, every_rot=every_rot, dset=dset )
            vl_dset, vl_desc    = gen_extern_large( v_classes, every_rot=every_rot, dset=dset )
            return tr_dset, vl_dset, tr_desc, vl_desc
        # case of training and validation sets using different rotations on same objects
        return gen_extern_large_tr_vl( every_rot=every_rot, train_val=train_val, dset=dset )

    # case of shapes generated internally
    return gen_inner_dset( every_rot=every_rot, train_val=train_val )


# ===================================================================================================================
#
#   MAIN
#
#
# ===================================================================================================================
if __name__ == '__main__':
    if not DO_NOTHING:
        gen_all_imgs()
