# IMPORT SECTION
#################

# LOCAL IMPORT
#################
from src.welcome_text import print_welcome
from src.check_compatibility import check_package, do_install
from src.utils import *

# MAIN PROGRAM
#################
def main():

    ##### WELCOME MESSAGE #####
    print_welcome()

    datapath = 'data/'
    if not os.path.exists(datapath):
        os.mkdir(datapath)

    ##### USER INPUT #####
    main_menu_user_input = input("Please select an option: ")
    if main_menu_user_input == "0":
        print("Exiting...")
        exit()

    elif main_menu_user_input == "9":
        check_package()
        # back to main question
        print(f"=====================")
        print(f"Do you want to back to main menu? (y/n)")
        back_to_main_menu = input("Please select an option: ")
        if back_to_main_menu == "y":
            clear_terminal()
            main()
        else:
            print("Exiting...")
            exit()

    elif main_menu_user_input == "10":
        do_install()
        print(f"=====================")
        print(f"Do you want to back to main menu? (y/n)")
        back_to_main_menu = input("Please select an option: ")
        if back_to_main_menu == "y":
            clear_terminal()
            main()
        else:
            print("Exiting...")
            exit()

    elif main_menu_user_input == "1":
        from src.batch_process import run_batch_process
        run_batch_process()

    elif main_menu_user_input == "2":
        from src.custom_process import run_custom_process
        run_custom_process()

    elif main_menu_user_input == "3":
        from src.create_topomap import run_create_topomap
        run_create_topomap()
    
    else:
        print("Invalid input, please try again")
        clear_terminal()
        main()

    
if __name__ == "__main__":
    main()