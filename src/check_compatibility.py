import subprocess


def do_check(package_name):
    try:
        pkg = __import__(package_name)
        print(f'Package "{package_name}" is installed.')
        print(f'Installed version of "{package_name}" is {pkg.__version__}.')
        return True
    except ImportError:
        print(f'Package "{package_name}" is NOT installed.')
        return False
    except AttributeError:
        print(f'Package "{package_name}" does not have a __version__ attribute.')
        return False

def check_package():
    packages = ['mne', 'numpy', 'pandas', 'scipy', 'tabulate', 'matplotlib']
    check_results = []
    for package in packages:
        result = do_check(package)
        check_results.append(result)

    if all(check_results):
        print("\nAll required packages are installed. You are good to go!")
    else:
        print("\nSome required packages are missing. Please check the messages above.")

# install_packages.py
def install_package(package_name):
    try:
        subprocess.check_call(['pip', 'install', package_name])
        print(f'Package "{package_name}" has been installed successfully.')
    except subprocess.CalledProcessError:
        print(f'An error occurred while attempting to install "{package_name}".')

def do_install():
    packages = ['mne', 'numpy', 'pandas', 'scipy', 'tabulate', 'matplotlib']
    for package in packages:
        install_package(package)
