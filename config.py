from __future__ import annotations
from typing import List, Dict
from abc import abstractmethod
import argparse
from distutils.util import strtobool

from filter_simulator.common import Limits, Position


class Config:
    def __init__(self):
        self._parser_groups = dict()
        self._parser = argparse.ArgumentParser(add_help=False, formatter_class=self._ArgumentDefaultsRawDescriptionHelpFormatter)
    # end def

    class _ArgumentDefaultsRawDescriptionHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        def _split_lines(self, text, width):
            return super()._split_lines(text, width) + ['']  # Add empty line between the entries
        # end def
    # end class

    class _EvalAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, comptype=None, user_eval=None):
            argparse.Action.__init__(self, option_strings, dest, nargs, const, default, type, choices, required, help, metavar)
            self.comptype = comptype
            self.user_eval = user_eval
        # end def

        def __call__(self, parser, namespace, values, option_string=None, comptype=None):
            x = self.silent_eval(values)

            if isinstance(x, self.comptype):
                setattr(namespace, self.dest, x)
            else:
                raise TypeError(f"'{str(x)}' is not of type '{self.comptype}', but of '{type(x)}'.")
            # end if
        # end def

        def silent_eval(self, expr: str):
            res = None

            try:
                if self.user_eval:
                    res = self.user_eval(expr)
                else:
                    res = eval(expr)
                # end if
            except SyntaxError:
                pass
            except AttributeError:
                pass
            # end try

            return res
        # end def
    # end class

    class _EvalListAction(_EvalAction):
        def __call__(self, parser, namespace, values, option_string=None, comptype=None, user_eval=None):
            x = self.silent_eval(values)

            if isinstance(x, list) and all(isinstance(item, self.comptype) for item in x):
                setattr(namespace, self.dest, x)
            # end if
        # end def
    # end class

    class _EvalListToTypeAction(_EvalAction):
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, comptype=None, restype=None, user_eval=None):
            Config._EvalAction.__init__(self, option_strings, dest, nargs, const, default, type, choices, required, help, metavar, comptype, user_eval)
            self.restype = restype
        # end def

        def __call__(self, parser, namespace, values, option_string=None, comptype=None, restype=None):
            x = self.silent_eval(values)

            if isinstance(x, list) and all(isinstance(item, self.comptype) for item in x):
                setattr(namespace, self.dest, self.restype(x))
            # end if
        # end def
    # end class

    class _LimitsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            x = Limits(*[float(x) for x in values])
            setattr(namespace, self.dest, x)
        # end def
    # end class

    class _PositionAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            x = Position(*[float(x) for x in values])
            setattr(namespace, self.dest, x)
        # end def
    # end class

    class _IntOrWhiteSpaceStringAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            x = ' '.join(values)
            setattr(namespace, self.dest, self.int_or_str(x))
        # end def

        @staticmethod
        def int_or_str(arg: str):
            try:
                res = int(arg)
            except ValueError:
                res = arg
            # end if

            return res
        # end def
    # end class

    class InRange:
        def __init__(self, dtype, min_val=None, max_val=None, min_ex_val=None, max_ex_val=None, excl=False):
            self.__dtype = dtype
            self.__min_val = min_val
            self.__max_val = max_val
            self.__min_ex_val = min_ex_val
            self.__max_ex_val = max_ex_val
            self.__excl = excl

            # The min-ex and max-ax values outplay the non-ex-values which just will be ignored
            if min_ex_val is not None:
                self.__min_val = None
            # end if

            if max_ex_val is not None:
                self.__max_val = None
            # end if
        # end def

        def __str__(self):
            elements = list()
            elements.append(f"dtype={type(self.__dtype()).__name__}")
            elements.append(f"min_val={self.__min_val}" if self.__min_val is not None else None)
            elements.append(f"min_ex_val={self.__min_ex_val}" if self.__min_ex_val is not None else None)
            elements.append(f"max_val={self.__max_val}" if self.__max_val is not None else None)
            elements.append(f"max_ex_val={self.__max_ex_val}" if self.__max_ex_val is not None else None)
            elements.append(f"excl={self.__excl}" if self.__excl else None)

            elements = [e for e in elements if e is not None]

            return f"InRange({', '.join(elements)})"
        # end def

        def __repr__(self):
            return str(self)
        # end def

        def __call__(self, x):
            x = self.__dtype(x)

            err = False

            if self.__min_val is not None and x < self.__min_val or \
                    self.__min_ex_val is not None and x <= self.__min_ex_val or \
                    self.__max_val is not None and x > self.__max_val or \
                    self.__max_ex_val is not None and x >= self.__max_ex_val:
                err = True
            # end if

            # If the defined range is ment to be exclusive, the previously determinded result gets inverted.
            if self.__excl:
                err = not err
            # end if

            if err:
                raise ValueError
            # end if

            return x
        # end def
    # end class

    class IsBool:
        def __new__(cls, x) -> bool:
            try:
                x = bool(strtobool(x))
                return x
            except ValueError:
                raise ValueError
            # end try
        # end def
    # end class

    @staticmethod
    @abstractmethod
    def _user_eval(s: str):  # This needs to be overridden, since eval() needs the indidivual globals and locals, which are not available here.
        pass
    # end def

    def help(self) -> str:
        return self._parser.format_help()
    # end def

    def read(self, argv: List[str]):
        args, unknown_args = self._parser.parse_known_args(argv)

        if len(unknown_args) > 0:
            print("Unknown argument(s) found:")
            for arg in unknown_args:
                print(arg)
            # end for
        # end if

        return args
    # end def
# end class


class ConfigAttribute:
    def __init__(self, attr_type, nullable=False, islist=False):
        self.types = attr_type if isinstance(attr_type, list) else [attr_type]
        self.nullable = nullable
        self.islist = islist
    # end def
# end class


# An alternative to this class might be to just use the Namespace pbject returned from argparse
class ConfigSettings:
    def __init__(self):
        self._attr_props: Dict[str, ConfigAttribute] = dict()
        self._mapping: Dict = dict()
    # end def

    def __getattr__(self, item):
        if item in self._mapping:
            return self._mapping[item]
        # end if
        raise KeyError(f"Object '{self}' has no attribute '{item}'.")
    # end def

    def __dir__(self):
        return self._mapping.keys()
    # end def

    def _add_attribute(self, attr_name, attr_type, nullable=False, islist=False):
        if attr_name not in self._get_attributes():
            setattr(self, attr_name, None)
            self._attr_props[attr_name] = ConfigAttribute(attr_type, nullable, islist)
            self._mapping[attr_name] = len(self._mapping)
        else:
            raise AttributeError(f"Attribute '{attr_name}' already set.")
    # end def

    def _get_attributes(self):
        return [a for a in dir(self) if not (a.startswith('__') or a.startswith('_') or a == "from_obj")]
    # end def

    @classmethod
    def from_obj(cls, obj):
        # Create new object
        new = cls()

        # Check if obj contains all elements we need for our class
        if not all(attr in dir(obj) for attr in new._get_attributes()):
            raise AttributeError(f"Missing attribute(s) in object of type {type(obj).__name__}: {[a for a in new._get_attributes() if a not in dir(obj)]}")
        # end if

        # Check and copy attributes
        for attr_name in new._get_attributes():
            # Check for correct type
            attr = getattr(obj, attr_name)
            attr_props = new._attr_props[attr_name]

            # Error checking
            if attr is None:
                if not attr_props.nullable:
                    raise TypeError(f"Attribute '{attr_name}' is of type 'None'.")
            else:
                if type(attr) is list:
                    if not attr_props.islist:
                        raise TypeError(f"Attribute '{attr_name}' is a list.")
                    else:
                        for i, item in enumerate(attr):
                            if not any(isinstance(item, _type) for _type in attr_props.types):
                                raise TypeError(f"Attribute '{attr_name}[{i}]' is of type '{type(item)}', but need's be of one of types '{attr_props.types}'.")
                            # end if
                        # end for
                    # end if
                else:
                    if attr_props.islist:
                        raise TypeError(f"Attribute '{attr_name}' is not a list.")
                    else:
                        if not any(isinstance(attr, _type) for _type in attr_props.types):
                            raise TypeError(f"Attribute '{attr_name}' is of type '{type(attr)}', but need's be of type '{attr_props.types[0]}'.")
                        # end if
                    # end if
                # end if
            # end if

            # Copy value
            setattr(new, attr_name, getattr(obj, attr_name))
        # end def

        return new
    # end def
# end class
