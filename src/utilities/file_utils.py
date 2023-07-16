import os,sys,glob

def get_files(folder,file_type='png'):
    '''
    Returns a list of images in a folder
    '''
    return glob.glob(os.path.join(folder, f'*.{file_type}'))

def clear_folder(folder):
    ''' 
    Clears Contents of a Specified folder
    '''
    files = get_files(folder)
    for file in files:
        os.remove(file)


def get_directory():
    '''
    Gets the directory of the data folder, relative to position of this file
    '''
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))