##
#@file utils 
#@brief pure utilitaries(other)
#  all of other usefull functions -*- coding: utf-8 -*-.
#
#@namespace utils
#@version   1.0
#@pre       EXample.First initialize the system.
#@bug       
#@warning   
#@copyright GNU General Public License v2.0
#@date Created on Tue Jun 30 21:01:17 2015
#@author agiovann

from __future__ import print_function


import numpy as np
import os
try:
    from urllib2 import urlopen as urlopen
except:
    from urllib.request import urlopen as urlopen

##\brief      downloading the demo from a dropbox folder
#\details using urllib, os.path
#\version   1.0
#\throws an exception if not in the Caiman folder
#\author  andrea giovannucci
def download_demo():
    if os.path.exists('./example_movies'):
        if not(os.path.exists('./example_movies/demoSue2x.tif')):        
            url = 'https://www.dropbox.com/s/09z974vkeg3t5gn/Sue_2x_3000_40_-46.tif?dl=1'
            print("downloading demo Sue2x with urllib")
            f = urlopen(url)
            data = f.read()
            with open("./example_movies/demoSue2x.tif", "wb") as code:
                code.write(data)
        else:
            print('File already existing')
    else:
         raise Exception('You must be in caiman folder')
#    print("downloading with requests")
#    r = requests.get(url)
#    with open("code3.tif", "wb") as code:
#        code.write(r.content)



def val_parse(v):
    """parse values from si tags into python objects if possible from si parse

     Parameters
     -----------
     
     v: si tags

     returns

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

     Parameters
     -----------
     
     imd: image description

     returns

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
    
     Parameters
     -----------
     
     fname: name of the file

     returns

        image_description: information of the image

    """

    image_descriptions=[]
    
    try:
        
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


