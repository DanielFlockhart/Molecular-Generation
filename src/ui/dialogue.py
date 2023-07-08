def get_user_choice():
    '''
    Get the user's choice of whether to train a new model or generate new molecules with a previously trained model
    '''
    while True:
        # Hard Coded at the moment - will be changed later
        choice = input("Please choose an option:\n1. Train a new model\n2. Generate new molecules with previously trained model\n")
        if choice in ['1', '2']:
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")

def confirm_choice(choice):
    '''
    Confirm the user's choice
    '''
    while True:
        confirm = input(f"You chose option {choice}. Is this correct? (Y/N) ")
        if confirm.upper() == 'Y':
            return True
        elif confirm.upper() == 'N':
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

