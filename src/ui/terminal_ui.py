import os,sys
sys.path.insert(0, os.path.abspath('..'))
from CONSTANTS import *


def format_title(title):
    '''
    Formats the title to be centered and have a line either side in this format:
    ----------------- Title -----------------
    '''
    return str(UI_TERMINAL_WIDTH*"-" + f" {title} " + UI_TERMINAL_WIDTH*"-")