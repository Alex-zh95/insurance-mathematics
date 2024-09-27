'''
Context file for importing parent module directories. Useful for writing tests and keeping test suite scripts organized.

Source: https://stackoverflow.com/questions/66961262/python-submodules-importerror-attempted-relative-import-with-no-known-parent
'''
import os
import sys


def access_root_dir(depth=1):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    args: list = [parent_dir]

    for _ in range(depth):
        args.append('..')

    rel_path = os.path.join(*args)
    sys.path.append(rel_path)
