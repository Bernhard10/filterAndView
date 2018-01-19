# coding: utf-8
                      
from .configValues import CONFIG
import functools
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import copy
import json
import warnings
import os
import re
import logging
import itertools
import ast
import operator as op
log = logging.getLogger(__name__)

ast_operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg, ast.Mod: op.mod}
ast_functions = { "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt }

def _adv_getitem_rec(data, node, allow_string):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Str) and allow_string:
        log.debug("String node %s", node.s)
        return node.s
    elif isinstance(node, ast.BinOp):
        return ast_operators[type(node.op)](_adv_getitem_rec(data, node.left), _adv_getitem_rec(data, node.right))
    elif isinstance(node, ast.UnaryOp):
        return ast_operators[type(node.op)](_adv_getitem_rec(data, node.operand))
    elif isinstance(node, ast.Call):
        f = ast_functions[node.func.id]
        print(f, type(f))
        args = [ _adv_getitem_rec(data, arg) for arg in node.args ]
        return f(*args) 
    elif isinstance(node, ast.Name):
        try:
            return data[node.id]
        except KeyError as e:
            if allow_string:
                warnings.warn("Treating {0} as string literal, since no column with this name exists."
                              "Consider using '{0}' instead".format(node.id))
                return node.id
            else:
                raise InvalidInput("{} is not a valid column name "
                                   "(and in the current context, "
                                   "strings are nt supported). ".format(node.id)) from e
    else:
        raise InvalidInput("Nodes of type {} not allowed in key".format(type(node)))

def adv_getitem(data, key, allow_string=False):
    try:
        tree = ast.parse(key, mode="eval").body
        ret = _adv_getitem_rec(data, tree, allow_string)
        log.info("Advanced_getitem for %s (allow_string = %s) returns %r", key, allow_string, ret)
        return ret
    except InvalidInput:
        raise
    except Exception as e:
        raise InvalidInput("The key '{}' was not understood".format(key)) from e
    
    """
    pattern = "[/*+%-]"
    hits = re.findall(pattern, key)
    try:
        if len(hits)==0:
            return data[key]
        elif len(hits)==1:
            try:
                key1, op, key2 = re.split("({})".format(pattern), key)
            except ValueError as e:
                log.error("Splitting {}:".format(key), re.split(pattern, key))
                raise  
            try:
                rs=int(key2)
            except:
                rs=data[key2]
            if op=="*":
                return data[key1]*rs
            elif op=="%":
                return data[key1]%rs
            elif op=="/":
                #log.info("divide {} by {}".format(data[key1],rs))
                return data[key1]/rs
            elif op=="+":
                return data[key1]+rs
            elif op=="-":
                return data[key1]-rs
        else:
            raise InvalidInput("At most one operator such as '*', '+', '-', '/' is allowed")
    except KeyError as e:
        raise InvalidInput("No column named '{}' for key '{}'".format(e, key))
    """
def datafilter(f):
    """
    Datafilters select a subsection of a pandas dataframe and store the selection criterion.
    
    Whether the subsection is a copy or a view of the original data is not specified.
    
    The selection criterion is stored in the dataframe's `_fav_history` attribute 
    (which was assigned by fav and is not from pandas)
    """
    @functools.wraps(f)
    def wrapped(data, key, value):
        d_type = adv_getitem(data,key).dtype
        log.info("Dtype of %s is %r", key, d_type)
        try:
            if (d_type == np.bool_ and value in ["0", "False", "false"] ):
                value2 = False
            else:
                try:
                    value2 = adv_getitem(data, value, allow_string=(d_type==object))
                except:
                    value2 = d_type.type(value)
        except ValueError as e:
            raise InvalidInput("Cannot convert value {} to type {} of "
                               "column {}.".format(value, d_type, key)) from e
        newdata = f(data, key, value2)
        newdata._fav_history = data._fav_history + ["{} {} {}".format(key, f.__doc__, value)]
        newdata._fav_datasetname = data._fav_datasetname
        return newdata
    return wrapped

# Now come the concrete datafilter implementations
@datafilter
def eq(data, key, value):
    """=="""
    try:
        if np.isscalar(value) and np.isnan(value):
            return data[np.isnan(adv_getitem(data,key))]
    except TypeError: #Datatype does not support nan (int/ string)
        pass 
    return data[adv_getitem(data,key)==value]

@datafilter
def ne(data, key, value):
    """!="""
    try:
        if np.isscalar(value) and np.isnan(value):
            return data[np.logical_not(np.isnan(adv_getitem(data,key)))]
    except TypeError: #Datatype does not support nan (int/ string)
        pass
    return data[adv_getitem(data,key)!=value]

@datafilter
def gt(data, key, value):
    """>"""
    return data[adv_getitem(data,key)>value]

@datafilter
def lt(data, key, value):
    """<"""
    return data[adv_getitem(data,key)<value]

@datafilter
def ge(data, key, value):
    """>="""
    return data[adv_getitem(data,key)>=value]

@datafilter
def le(data, key, value):
    """<="""
    return data[adv_getitem(data,key)<=value]

@datafilter
def in_(data, key, value):
    """contains"""
    return data[adv_getitem(data,key).str.contains(value)]

@datafilter
def notin(data, key, value):
    """doesnot_contain"""
    return data[np.invert(adv_getitem(data,key).str.contains(value))]

def hist_to_title(history):
    if history:
        title = "<"+";".join(history)+">"
    else:
        title = "<no filters>"
    return title  

OPS = {
      "==": eq,
      "=": eq,
      ">": gt,
      "<": lt,
      "<=": le,
      ">=": ge,
      "!=": ne,
      "contains": in_, 
      "doesnot_contain": notin
      }
    
def apply_filter(data, key, operator, value):
    if operator not in OPS:
        raise InvalidInput("Operator {} not understood. Must be one of "
                           "{}".format(operator, ",".join(map(repr, OPS.keys()))))
    #print(repr(value))
    return OPS[operator](data, key, value)

def subdataset_from_history(data, history):
    for hist_entry in history:
        key, op, val = hist_entry.split()
        data=apply_filter(data, key, op, val)
    return data

FAIL = '\033[91m'
ENDC = '\033[0m'
OUTPUT = '\033[92m' #expected output of commands
OKBLUE = '\033[94m' 
BOLD = '\033[1m'



class InvalidInput(ValueError):
    """
    Raised during interactive analysis, whenever the user input is invalis
    """

class UnknownCommand(InvalidInput):
    """
    Raised for unknown commands by the user.
    """
    pass

class DataAnalysis(object):
    """
    A main class that holds all the data loaded and has member functions for all data analysis
    """
    def __init__(self, data = pd.DataFrame(), dataset_name = "main", column_metadata = {}):
        """
        :param data: A pandas dataframe
        :param dataset_name: Assign a name to this dataset
        :param column_metadata: None or a dict of dicts.
        """
        data._fav_history = []
        data._fav_datasetname = dataset_name
        self.data={dataset_name: data}
        self.column_metadata = defaultdict(dict)
        self.column_metadata[dataset_name] = column_metadata
        self.filtered_data = data
        self.stored = {}
        

        self.allowed_commands = OrderedDict([
            ("HELP", self.show_help),
            ("R", self.reset),
            ("CONFIG", self.set_config),
            ("SAVE", self.save),
            ("LOAD", self.load),
            ("DEL", self.del_saved),
            ("SHOW_SAVED", self.show_saved),
            ("PRINT", self.print_data),
            ("IMPORT_DATASET", self.import_dataset),
            ("SELECT_DATASET", self.select_dataset),
            ("WRITE_DATASET", self.write_dataset),
            ("JOIN", self.join_datasets)
        ])
        #Keep our local instance of configuration variables.
        self.settings = copy.copy(CONFIG)
    
    @property
    def current_title(self):
        return hist_to_title(self.filtered_data._fav_history)
    
    def p(self, command):
        """Shortcut for use in jupyter,..."""
        return self.perform(command)
    def __lshift__(self, shift):
        """Shortcut for use in jupyter,..."""
        return self.perform(shift)
    def perform(self, command):
        """
        Perform one command out of self.allowed_commands or apply a filter.
        """
        parts = command.strip().split()
        if not parts:
            return #Empty command
        if parts[0] in self.allowed_commands:
            try:
                self.allowed_commands[parts[0]](*parts[1:])
            except TypeError as e:
                if "argument" in str(e):
                    raise InvalidInput("Wrong number of arguments.") from e
                else:
                    raise

        else:
            try:
                self.apply_filter(*parts)
            except TypeError as e:
                if "argument" in str(e):
                    raise InvalidInput("Filters need to be specified as space-seperated triples: "
                               "key, operator, value.") from e
                else:
                    raise
            except KeyError:
                raise UnknownCommand(parts[0])

    def apply_filter(self, key, operator, value):
        self.filtered_data = apply_filter(self.filtered_data, key, operator, value)

    def _get_range(self, key, from_, to_, step=None):
        use_intervals = False
        if "." in from_ or "." in to_ or (step and "." in step):
            use_intervals = True
        if self.filtered_data[key].dtype == np.float_:
            use_intervals = True
        if use_intervals:
            from_, to_ = sorted([float(to_),float(from_)])
            if not step:
                step =(from_-to_)/10
                if step<0:
                    step*=-1
            else:
                step=abs(float(step))
            target_range = []
            for i in itertools.count():
                target_range.append((step*i, step*(i+1)))
                if step*(i+1)>max(to_, from_):
                    break
        else:
            from_, to_ = sorted([int(to_),int(from_)])
            if step:
                step = abs(int(step))
                target_range=list(range(int(from_), int(to_), int(step)))
            else:
                target_range=list(range(int(from_), int(to_)))
        return target_range

    def _filter_from_r(self, key, r):
        """
        Further filter the filtered data by the given range object.
        
        param r: an integer (for '==') or a tuple (start, stop)
        """
        if isinstance(r, int):
            data = OPS["=="](self.filtered_data, key, r)
        else:
            data = OPS["<"](OPS[">="](self.filtered_data, key, r[0]), key, r[1])
        return data

    def show_help(self, arg=None):
        """
        Show this help message
        
        Use `HELP [ USAGE | KEYS | OPERATORS | COMMANDS ]`
        """
        if arg is None or arg=="USAGE":
            print("Usage of the interactive prompt:\n"
                     "###  Input filers like 'ml_length == 3' or 'segment_length < 4'\n"
                     "     Note that the space is important!\n")
        if arg is None or arg == "KEYS":
            print("***  Valid keys depend on the loaded dataset. Currently they are:\n")
            for header in self.filtered_data.columns.values:
                print(   "          {}".format(header))        
        if arg is None or arg == "OPERATORS":
            print(       "***  Valid operators are:")
            for op in OPS.keys():
                print(   "          {}".format(op))
        if arg is None or arg == "COMMANDS":
            print     ("###  In addition, the following commands are available:")
            max_commandlength = max(len(c) for c in self.allowed_commands.keys())
            template = "     * {: <"+str(max_commandlength)+"}\t{}"
            for command, function in self.allowed_commands.items():
                print (template.format(command, function.__doc__))
            print     ("###  The following configuration values are available:")
            template = "     * {: <"+str(max(len(c) for c in self.settings.keys()))+"}\t{})"
            for name in self.settings:
                print (template.format(name, self.settings.get_doc(name)))

    def reset(self):
        """
        Reset all filters (restore original dataset).
        """
        self.filtered_data = self.data[self.filtered_data._fav_datasetname]

    def set_config(self, name, val):
        """
        Set configuration values (see below)

        Use 'CONFIG NAME VALUE' to set the configuration variable with name NAME to the value VALUE.
        """
        try:
            self.settings[name] = val
        except Exception as e:
            raise InvalidInput from e
        
    def save(self, *args):
        """
        Save current dataset under the given name.

        Use 'SAVE NAME' to save the current dataset under the name NAME.
        Use 'SAVE NAME for KEY FROM TO [STEP]' to save subsets of the current dataset according
            to the range specified with FROM and TO. 
            They will have '_NUMBER' appended to their name.        
        """
        if len(args)>1 and args[1]=="for":
            range_ = self._get_range(*args[2:])
            for r in range_:
                data = self._filter_from_r(args[2], r)
                if isinstance(r, int):
                    name = "{}_{}".format(args[0], r) 
                else:
                    name = "{}_[{},{})".format(args[0], r[0], r[1]) 
                print("SAVING {}".format(name))
                self.stored[name]=data
        elif len(args)==0:
            raise InvalidInput("Need a name to save data")
        elif len(args)>1:
            raise InvalidInput("Name for data save must not contain any spaces! "
                               " (Use the 'SAVE NAME for'-syntax for saving several sub-datasets")
        else:
            if args[0]=="for":
                raise InvalidInput("Name 'for' is not allowed as a name for saving data, because it is a keyword")
            self.stored[args[0]]=self.filtered_data
            
    def load(self, name):
        """ 
        Load a previousely saved dataset.
        """
        self.filtered_data = self.stored[name]
        
    def del_saved(self, *args):
        """
        Delete a dataset that was previousely saved under this name(s)

        Use 'DEL NAME1 [NAME2...] to delete saves NAME1,... 
        """
        if len(args)==0:
            raise InvalidInput("Need at least one name of a save which should be deleted.")
        for name in args:
            del self.stored[name]
            
    def show_saved(self):
        """
        Show all saved datasets
        """
        if self.stored:
            for k, v in sorted(self.stored.items()):
                title = hist_to_title(v._fav_history)
                print ( k, "\t", title, "\t", "Dataset: {}".format(v._fav_datasetname))
        else:
            print("-- nothing saved --")
                
    def print_data(self, column_name=None, verbose = False):
        """
        Print some values and a summary for a column of data.

        Use 'PRINT key' to print information about the column with name key
        Use 'PRINT key TRUE' to print verbose information about the column with name key

        """            
        if verbose:
            print (self.filtered_data[column_name])
        print(self.filtered_data[column_name].describe())
    
    def import_dataset(self, path, name):
        """
        Import another dataset into the application.
        
        Use 'IMPORT_DATASET path/to/csv/file/csv name' to import the file and assign the name 'name' to it.
        """        
        if name in self.data:
            raise InvalidInput("Name {} exists already. Please choose another name.")
        saved_dict = {}
        
        with open(os.path.expanduser(path)) as f:
            for line in f:
                if line[:2]!="#_":
                    break #We only look at comments at the beginning of the file
                elif line.startswith("#_fav_saved"):
                    saved_dict = json.loads(line.partition(" ")[2])
                elif line.startswith("#fav_colmeta"):
                    self.column_metadata[name]=json.loads(line.partition(" ")[2])
        data = pd.read_csv(os.path.expanduser(path), comment="#")
        data._fav_history = []
        data._fav_datasetname = name
        self.data[name] = data
        log.info("The file contains the following stored entries: {}".format(saved_dict))
        for savename, hist in saved_dict.items():
            if savename in self.stored:
                warnings.warn("Saved subdataset '{}' from file ignored."
                              "There is already a save with this name present.")
            else:
                self.stored[savename] = subdataset_from_history(data, hist)
        print ("Dataset has been imported. Use SELECT_DATASET {} to select it.".format(name))
    
    def select_dataset(self, name):
        """
        Switch to another dataset.
        
        Use `SELECT_DATASET name` to switch to the dataset `name`
        """
        self.filtered_data = self.data[name]
        print("Dataset {} has been selected".format(name))

    def write_dataset(self, name, path):
        histories = {}
        for savename, save in self.stored.items():
            if save._fav_datasetname == name:
                histories[savename] = save._fav_history
            else:
                print(save._fav_datasetname)
        print(json.dumps(histories))
        with open(os.path.expanduser(path), "w") as f:
            print("#_fav_saved " + json.dumps(histories), file=f)
            print("#_fav_colmeta " + json.dumps(self.column_metadata[name]), file=f)
        
        self.data[name].to_csv(path, mode="a")
    
    def join_datasets(self, name1, name2, _on, key1, *args ):
        """
        Join two datasets on the columns specified by key to producte a new dataset.
        
        Use `JOIN name1 name2 ON key1 = key2 AS target_name` or
            `JOIN name1 name2 ON key1 == key2 AS target_name`
            to join dataset `name1` with dataset `name2` using `key1` of dataset `name1` 
            and `key2` of dataset `name2` and save it as dataset `target_name`
        Use `JOIN name1, name2 ON key1 AS target_name` if tboth datasets have the same key.
        
        Performs an outer join. See `http://pandas.pydata.org/pandas-docs/stable/merging.html`
        for documentation.
        
        WARNING: If a many-to-one or many-to-many relationship exists, rows will be duplicated,
                 which can influence statistics calculated on the resultng dataframe.
        """
        if args[0]=="AS":
            if len(args)!=2:
                raise InvalidInput("Wrong Syntax for JOIN. Expected exactly one argument after 'AS', found {}: {}".format(len(args)-1, args[1:]))
            key2 = key1
            target_name = args[1]
        elif args[0] in ["=", "=="]:
            if len(args)!=4:
                raise InvalidInput("Wrong Syntax for JOIN. Expected 8 arguments, found {}".format(4+len(args)))
            key2 = args[1]
            if args[2]!="AS":
                raise InvalidInput("Wrong Syntax for JOIN. Expected 'AS', found {}".format(args[2]))
            target_name = args[3]
        if target_name in self.data:
            raise InvalidInput("Target_name {} exists already. Please choose another name.".format(target_name))
        try:
            left = self.data[name1]
            right = self.data[name2]
        except KeyError as e:
            raise InvalidInput("Dataset {} does not exist".format(e))
        try:
            merged = pd.merge(left, right, how='outer', left_on=key1, right_on = key2)
        except KeyError:
            error = ""
            if key1 not in left:
                error += "Key {} not in dataset {}".format(key1, name1)
            if key2 not in right:
                error += "Key {} not in dataset {}".format(key2, name2)
            if error:
                raise InvalidInput(error) from None
            else:
                raise
        merged._fav_history = [ "{} merged_with {}".format(name1, name2) ]
        merged._fav_datasetname = target_name
        self.data[target_name] = merged

        print("Dataset {} has been created".format(target_name))
        print("Use `SELECT_DATASET {}` to select it.".format(target_name))
    
    def replay_history(self, dataset, history):
        new_dataset = dataset.copy()
        new_dataset._fav_history=[]
        new_dataset._fav_datasetname = dataset._fav_datasetname
        for hist_entry in history:
            hist_entry = hist_entry.split()
            if len(hist_entry) == 3:
                if hist_entry[1] in OPS:
                    log.info("Replaying {}".format(hist_entry))
                    new_dataset = apply_filter(new_dataset, *hist_entry)
                elif hist_entry[1]=="merged_with":
                    new_dataset._fav_history+=hist_entry
            else:
                raise RuntimeError("During replaying history: Entry {} could not be replayed.".format(hist_entry))
        return new_dataset
