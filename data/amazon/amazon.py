import  os
from    PIL     import Image


bg_color    = 'white'       # background color of segmented image
cam_list    = ( 5, )        # which camera poses to extract from the dataset [1-5]
n_size      = 256           # square size of new images


def mask( f_img, f_mask ):
    img     = Image.open( f_img )
    mask    = Image.open( f_mask )
    assert img.size == mask.size

    bg      = Image.new( 'RGB', img.size, bg_color )
    res     = Image.composite( bg, img, mask )
    return res


def crop_scale( o_img ):
    ow, oh  = o_img.size

    nw, nh  = ( 1500, 1500 )        # parameters tuned for CAM N5
    t_delta = 1000                  # parameters tuned for CAM N5
    l_delta = 0                     # parameters tuned for CAM N5

    # nw, nh  = ( 2000, 2000 )        # parameters tuned for CAM N4
    # t_delta = 0                     # parameters tuned for CAM N4
    # l_delta = 0                     # parameters tuned for CAM N4

    # nw, nh  = ( 2000, 2000 )        # parameters tuned for CAM N3
    # t_delta = 0                     # parameters tuned for CAM N3
    # l_delta = -500                  # parameters tuned for CAM N3

    l       = ( ow - nw + l_delta ) // 2
    t       = ( oh - nh + t_delta ) // 2
    r       = ( ow + nw + l_delta ) // 2
    b       = ( oh + nh + t_delta ) // 2

    n_img   = o_img.crop( ( l, t, r, b ) )
    n_img   = n_img.resize( ( n_size, n_size ) )
    return n_img


def make_dataset( old_dir, new_dir, cam_list ):
    if not os.path.exists( new_dir ):
        os.makedirs( new_dir )

    o_cnt   = 0

    for obj_dir in sorted( os.listdir( old_dir ) ):
        if obj_dir == '.DS_Store':
            continue

        o_cnt += 1

        cond_img    = lambda x: x.endswith( '.jpg' ) and int( x[ 1 ] ) in cam_list
        l_img       = os.listdir( os.path.join( old_dir, obj_dir ) )
        l_img       = sorted( [ x for x in l_img if cond_img( x ) ] )

        for f in l_img:
            fm      = f.split( '.' )[ 0 ] + '_mask.pbm'
            f_img   = os.path.join( old_dir, obj_dir, f )
            f_mask  = os.path.join( old_dir, obj_dir, 'masks', fm )

            img     = mask( f_img, f_mask )
            img     = crop_scale( img )

            s_cam   = f[ 1 ]
            s_ang   = int( f.split( '_' )[ 1 ].split( '.' )[ 0 ] )
            s       = f"{o_cnt:03d}_{s_cam}_{s_ang:03d}.jpg"
            s       = os.path.join( new_dir, s )

            print( "saving", s )
            img.save( s )


# folder with original images, new folder with processed images
make_dataset( "amazon-orig", "amazon-N5", cam_list )
