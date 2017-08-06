""" pure utilitaries(other)
 
 all of other usefull functions 
 
See Also
------------
https://docs.python.org/2/library/urllib.html
 
"""
#\package Caiman/utils
#\version   1.0
#\bug       
#\warning   
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2015
#\author: andrea giovannucci
#\namespace utils
#\pre none






from __future__ import print_function


import numpy as np
import os
try:
    from urllib2 import urlopen as urlopen
except:
    from urllib.request import urlopen as urlopen








def download_demo(name='Sue_2x_3000_40_-46.tif'):
    """download a file from the file list with the url of its location
 
 
    using urllib, you can add you own name and location in this global parameter
 
        Parameters:
        -----------
 
        name: str
            the path of the file correspondong to a file in the filelist (''Sue_2x_3000_40_-46.tif' or 'demoMovieJ.tif')
 
    Raise:
    ---------
        WrongFolder Exception
 

    """
   
    
    #\bug       
    #\warning  
    
    file_dict = {'Sue_2x_3000_40_-46.tif':'https://www.dropbox.com/s/09z974vkeg3t5gn/Sue_2x_3000_40_-46.tif?dl=1',
                 'demoMovieJ.tif':'https://www.dropbox.com/s/8j1cnqubye3asmu/demoMovieJ.tif?dl=1',
                 'demo_behavior.h5':'https://www.dropbox.com/s/53jmhc9sok35o82/movie_behavior.h5?dl=1'}
    #          ,['./example_movies/demoMovie.tif','https://www.dropbox.com/s/obmtq7305ug4dh7/demoMovie.tif?dl=1']]
    base_folder = './example_movies'
    if os.path.exists(base_folder):
         path_movie = os.path.join(base_folder,name)
         if not os.path.exists(path_movie):        
                url = file_dict[name]
                print( "downloading "+ name +"with urllib" )
                f = urlopen(url)
                data = f.read()
                with open(path_movie, "wb") as code:
                    code.write(data)
         else:
             
             print("File already downloaded")
    else:

         raise Exception('You must be in caiman folder')
#    print("downloading with requests")
#    r = requests.get(url)
#    with open("code3.tif", "wb") as code:
#        code.write(r.content)



def val_parse(v):
    """parse values from si tags into python objects if possible from si parse

     Parameters:
     -----------
     
     v: si tags

     returns:
     -------

    v: python object 

    """


    try:

        return eval(v)

    except:

        if v == 'true':
            return True
        elif v == 'false':
            return False
        elif v == 'NaN':
            return np.nan
        elif v == 'inf' or v == 'Inf':
            return np.inf

        else:
            return v




def si_parse(imd):

    """parse image_description field embedded by scanimage from get iamge description

     Parameters:
     -----------
     
     imd: image description

     returns:
     -------

    imd: the parsed description

    """

    imd = imd.split('\n')
    imd = [i for i in imd if '=' in i]
    imd = [i.split('=') for i in imd]
    imd = [[ii.strip(' \r') for ii in i] for i in imd]
    imd = {i[0]:val_parse(i[1]) for i in imd}
    return imd


def get_image_description_SI(fname):
    
    """Given a tif file acquired with Scanimage it returns a dictionary containing the information in the image description field
    
     Parameters:
     -----------
     
     fname: name of the file

     returns:
     -------

        image_description: information of the image

    Raise:
    -----
        ('tifffile package not found, using skimage.external.tifffile')


    """

    image_descriptions=[]
    
    try:
        #todo check this unresolved reference
        from tifffile import TiffFile
    
    except:

        print('tifffile package not found, using skimage.external.tifffile')
        from skimage.external.tifffile import TiffFile 
        
    tf=TiffFile(fname)
    
    for idx,pag in enumerate(tf.pages):
        if idx%1000==0:
            print(idx)
    #        i2cd=si_parse(pag.tags['image_description'].value)['I2CData']
        field=pag.tags['image_description'].value

        image_descriptions.append(si_parse(field))

    return image_descriptions


