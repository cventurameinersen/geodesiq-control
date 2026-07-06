# About geodesiq-control

import platform
import sys
from importlib.metadata import version, PackageNotFoundError

def about():
    """
    Prints version information for geodesiq and its core dependencies,
    along with system architecture details.
    """
    
    try:
        pkg_version = version("geodesiq")
    except PackageNotFoundError:
        pkg_version = "Development / Uninstalled"

   
    dependencies = ["numpy", "scipy"] 
    dep_versions = {}
    for dep in dependencies:
        try:
            dep_versions[dep] = version(dep)
        except PackageNotFoundError:
            dep_versions[dep] = "Not Installed"

    print("-" * 50)
    print(f"geodesiq Information")
    print("-" * 50)
    print(f"geodesiq version: {pkg_version}")
    print(f"Python version:    {platform.python_version()} ({sys.implementation.name})")
    print(f"Operating System:  {platform.system()} ({platform.release()}, {platform.machine()})")
    
    print("\nCore Dependencies:")
    for dep, ver in dep_versions.items():
        print(f"  {dep:<15}: {ver}")
    print("-" * 50)

    print("\n")
    print("=" * 50)
    print("Please cite geodesiq in your publication:")
    print("Your Citation Information Here")
    print("=" * 50)
