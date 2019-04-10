'''light_curve_class.py - Joshua Wallace - Feb 2019

This contains a class useful for organizing an object's light curve and 
other useful data, just to keep things clean going forward.
'''

import numpy as np


class light_curve():
    ''' A base class for storing the basic light curve data: times,
    magnitudes, and errors.

    The initialization takes three arguments:
     times - the times of the observations
     mags  - the measured brightnesses, in magnitudes
     errs  - the measurement errors
    '''
    def __init__(self,times_,mags_,errs_):
        # First, make sure that times_, mags_, and errs_ all have
        # a len attribute and are the same length
        if not hasattr(times_,'__len__'):
            raise AttributeError("times does not have a len attribute")
        if not hasattr(mags_,'__len__'):
            raise AttributeError("mags does not have a len attribute")
        if not hasattr(errs_,'__len__'):
            raise AttributeError("errs does not have a len attribute")
        if len(times_) != len(mags_):
            raise ValueError("The lengths of times and mags are not the same")
        if len(mags_) != len(errs_):
            raise ValueError("The lengths of mags and errs are not the same")

        # Make sure these are numpy arrays, since that is what the astrobase
        # period search functions will take
        if isinstance(times_,np.ndarray):
            self.times = times_
        else:
            self.times = np.array(times_)
        if isinstance(mags_,np.ndarray):
            self.mags = mags_
        else:
            self.mags = np.array(mags_)
        if isinstance(errs_,np.ndarray):
            self.errs = errs_
        else:
            self.errs = np.array(errs_)


class single_lc_object(light_curve):
    ''' A class for storing light curve info (time, mags, errs) coupled
    with an object's position and ID.  Arbitrary extra information can
    be stored as well in a dictionary.

    Subclasses the light_curve class.

    The initialization takes six arguments, with a seventh optional:
     times - the times of the observations
     mags  - the measured brightnesses, in magnitudes
     errs  - the measurement errors
     x     - the x-position of the object in pixel coordinates
     y     - the y-position of the object in pixel coordinates
     ID    - A unique ID for the object
     extra_info - (optional) a dictionary containing extra information
                  for the object
    '''
    def __init__(self,times_,mags_,errs_,x_,y_,ID_,extra_info={}):
        light_curve.__init__(self,times_,mags_,errs_)
        self.x = x_
        self.y = y_
        self.ID = ID_
        self.neighbors = [] # Empty list of nearby stars
        self.extra_info = extra_info


class lc_objects():
    '''A class for storing single_lc_object instances for a whole suite
    of objects.  This class also determines which objects are neighbors 
    when they get added to the collection, as well as ensures uniqueness
    of object IDs.

    The initialization takes one argument:
     radius - the circular radius, in pixels, for objects to be in 
              sufficient proximity to be regarded as neighbors
    '''
    def __init__(self,radius_):
        self.neighbor_radius_squared = radius_**2 # Storing radius squared
        self.objects = [] # To store all the objects
        self.index_dict = {} # Dictionary to map IDs to self.objects indices
        self.results = {} # Dictionary to save results

    # Method to add a single_lc_object
    def add_object(self,times,mags,errs,x,y,ID,extra_info={}):#,object):
        # Make sure the object's ID does not match any other object
        if ID in self.index_dict.keys():
            raise ValueError("Not a unique ID, " + str(ID))

        # Add the object to the list
        self.objects.append(single_lc_object(times,mags,errs,x,y,ID,
                                                 extra_info=extra_info))
        # Map the object's ID to its position in self.objects
        self.index_dict[ID] = len(self.objects)-1

        # Give it a place in self.results
        self.results[ID] = {}

        # Check if the new object is neighbor to any other objects
        for o in self.objects[:-1]:
            if (x - o.x)**2 + (y - o.y)**2 < self.neighbor_radius_squared:
                self.objects[-1].neighbors.append(o.ID)
                o.neighbors.append(ID)



        
