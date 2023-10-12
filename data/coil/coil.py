import  os
from    PIL     import Image

ang_step    = 5             # rotation delta
rot_scale   = 2             # padding to enlarge images before rotation
bg_color    = ( 0, 0, 0 )   # background color


def rot_pad( im, angle ):
    size    = im.size[ 0 ]
    s1      = size * rot_scale
    s2      = size // rot_scale

    # padding
    im_pa   = Image.new( 'RGB', ( s1, s1 ), bg_color )
    im_pa.paste( im, ( s2, s2 ) )

    # rotation
    im_ro   = im_pa.rotate( -angle, resample=Image.BILINEAR )
    im_ro   = im_ro.convert( 'RGB' )

    # im_ro.show()
    return im_ro


def all_rot( f_im, o_dir, n_dir ):
    ls_rot  = range( 0, 360, ang_step )

    for ang in ls_rot:
        im      = Image.open( os.path.join( o_dir, f_im ) )
        im      = rot_pad( im, ang )
        f1, f2  = f_im.split( '.' )
        f1      = os.path.join( n_dir, f1 )
        fn      = f"{f1}__{ang:03d}.{f2}"

        print( "saving", fn )

        im.save( fn )


def make_dataset( o_dir, n_dir ):
    for f_im in os.listdir( o_dir ):
        if not f_im.endswith( ( ".png", ".jpg" ) ):
            continue
        all_rot( f_im, o_dir, n_dir )


# im      = Image.open( "coil-24/obj99-000.png" )

# folder with original images, new folder with all rotations
make_dataset( "coil-24-orig", "coil-24" )
