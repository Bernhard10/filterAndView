from collections import OrderedDict
from collections.abc import Mapping
import copy
import warnings

__all__ = [ "CONFIG"]

class ConfigValues(Mapping):
    """
    A dictionary holding all configuration variables known to the program with their documentation.
    """
    def __init__(self):
        self.values = OrderedDict()
        self.type = {}
        self.doc = {}
        self.none_allowed = {}
        self.restricted = {}
        #If a copy of this instance is made, the original instance will be stored as parent 
        # and used as fallback for lookup.
        self.parent = None
    def add_item(self, name, default, type_, doc, none_allowed = False, restricted = None):
        """
        Add a new configuration variable
        
        ..note::
        
            This should be called when a module that relies on a config variable is loaded,
            before the user updates some config variables (using __setitem__) 
            or a config file is read.
                   
        :param name: The variable name
        :param default: The initial value of this variable
        :param type_: the type of the variable. Calling this should 
                      return a new object of the correct type.
        :param doc: Documenatation for this configuration variable
        :param none_allowed: Bool. Wheter this variable can be set to None
        :param restricted: None or a list. 
                           If a list is given, the config variable can only take 
                           values from this list.
        """
        if name in self.values:
            warnings.warn("Cannot add item {} twice".format(name))
            return
        if none_allowed and default is None:
            self.values[name] = None
        else:
            self.values[name] = type_(default)
        self.type[name] = type_
        self.doc[name] = doc
        self.none_allowed[name] = none_allowed
        self.restricted[name] = restricted
    def __getitem__(self, key):
        try:
            return self.values[key]
        except KeyError: 
            #If we are using a copy, try falling back to the original for reading (not for writing)
            return self.parent[key]
        
    def __setitem__(self, key, val):
        if key not in self:
            raise ValueError("Can only update values")
        if (val == "None" or val is None) and self.none_allowed[key]:
            self.values[key] = None
        elif val is None and not self.none_allowed[key]:
            raise TypeError("None is not allowed for key {}".format(key))
        else:
            val = self.type[key](val)
            if self.restricted[key] and val not in self.restricted[key]:
                raise ValueError("Value for configuration variable {} must be "
                                 "one of {}".format(name, self.restricted[name]))
            self.values[key] = val
    def __len__(self):
        return len(self.values)
    def __iter__(self):
        return iter(self.values)
    def __copy__(self):
        sett = ConfigValues()
        sett.parent = self
        sett.values = copy.copy(self.values)
        # A reference to the following dictsis enough, because they may only acquire new entries 
        # but no entries can be deleted or modified.
        sett.type = self.type
        sett.doc = self.doc
        sett.none_allowed = self.none_allowed
        sett.restricted = self.restricted
        return sett
    def get_doc(self, key):
        """
        Get the documentation for this config variable
        
        :param key: The name of the variable
        """
        typestr = "{}".format(self.type[key])
        if self.restricted[key]:
            typestr += ", one of {}".format(self.restricted[key])
        elif self.none_allowed[key]:
            typestr += " or None"
        return  ("{}\n"
                "\t\t[{}]\n"
                "\t\t{}".format(repr(self.values[key]), typestr, self.doc[key]))
                
CONFIG = ConfigValues()