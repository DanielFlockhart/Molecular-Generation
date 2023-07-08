import os,sys
sys.path.insert(0, os.path.abspath('..'))
from CONSTANTS import *


def format_title(title):
    return str(UI_TERMINAL_WIDTH*"-" + title + UI_TERMINAL_WIDTH*"-")