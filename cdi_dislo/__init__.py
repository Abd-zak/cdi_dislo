import importlib
import pkgutil
import sys
# Do not import everything by default. Handlers should import only the
# specific packages they need, e.g. `from cdi_dislo import common_imports as ci`.
# This avoids polluting the package namespace and reduces costly startup imports.

# Get the package name dynamically
package_name = __name__

# List all submodules dynamically
__all__ = []

def recursive_import(package_path, package_name):
    """
    Recursively import all submodules and add them to __all__.
    """
    submodules = []
    for finder, module_name, ispkg in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        try:
            module = importlib.import_module(module_name)
            submodules.append(module_name.split(".")[-1])  # Add only the last part of module name
        except ModuleNotFoundError as e:
            print(f"⚠️ ModuleNotFoundError: Could not import {module_name}: {e}")
        except ImportError as e:
            print(f"⚠️ ImportError: Could not import {module_name}: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected Error: Failed to import {module_name}: {e}")

    return submodules

# Run recursive import and populate __all__
__all__ = recursive_import(__path__, package_name)

# Ensure no duplicates in __all__
__all__ = list(set(__all__))

# Print confirmation (for debugging)
print(f"✅ Fully loaded modules in {package_name}: {__all__}")

