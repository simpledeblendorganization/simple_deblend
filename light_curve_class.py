'''lightcurve_class.py - Joshua Wallace - Jan 2019

This contains a class useful for organizing an object's light curve and 
other useful data, just to keep things clean going forward.
'''


class light_curve():
    def __init__(self,times_,mags_,errs_):
        if not hasattr(times_,'__len__'):
            raise RuntimeError("times does not have a len attribute")
        if not hasattr(mags_,'__len__'):
            raise RuntimeError("mags does not have a len attribute")
        if not hasattr(errs_,'__len__'):
            raise RuntimeError("errs does not have a len attribute")
        if len(times_) != lens(mags_):
            raise RuntimeError("The lengths of times and mags are not the same")
        if len(mags_) != len(errs_):
            raise RuntimeError("The lengths of mags and errs are not the same")
        self.times = times_
        self.mags = mags_
        self.errs = errs_


class single_lc_object(light_curve):
    def __init__(self,times_,mags_,errs_,x_,y_,ID_,extra_info_={}):
        light_curve.init(self,times_,mags_,errs_)
        self.x = x_
        self.y = y_
        self.ID = ID_
        self.neighbors = []
        self.extra_info = extra_info_


class lc_objects():
    def __init__(self,radius_):
        self.neighbor_radius = radius_
        self.objects = []

    def add_object(self,object):
        if not isinstance(object,single_lc_object):
            raise RuntimeError("Was not given an instance of single_lc_object")

        # Check if the new object is neighbor to any other objects
        for o in self.objects:
            if (object.x - o.x)**2 + (object.y - o.y)**2 < self.neighbor_radius:
                object.neighbors.append(o.ID)
                o.neighbors.append(object.ID)

        self.objects.append(object)

        
