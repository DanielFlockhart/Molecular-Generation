import os,sys
sys.path.insert(0, os.path.abspath('..'))
from CONSTANTS import *


def format_title(title):
    '''
    Formats the title to be centered and have a line either side in this format:
    ----------------- Title -----------------
    '''
    return str(UI_TERMINAL_WIDTH*"-" + f" {title} " + UI_TERMINAL_WIDTH*"-")


def console_grid(strings,outs):
    '''
    Creates a grid like output for the console
    '''
    output = ""
    longest_length = max(len(string) for string in strings)
    for (i, string) in enumerate(strings):
        output+=string + str(" "*(longest_length - len(string))) + " : "
        output+= str(outs[i])
        output+="\n"
    return output
