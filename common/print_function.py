"""Overloads the built-in print function to log into a file as well."""

import builtins as __builtin__


class Print(object):
  def __init__(self, file_path):
    self.file_path = file_path
    self.fileout = open(self.file_path, "a+")
    self.builtin_print = __builtin__.print
    print("Logging to " + self.file_path)
  
  def __call__(self, *args, **kwargs):
    """Writing to file works only if there is a single string input."""
    self.builtin_print(*args, **kwargs)
    kwargs["file"] = self.fileout
    self.builtin_print(*args, **kwargs)
    self.fileout.flush()
