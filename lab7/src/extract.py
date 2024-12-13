#!/usr/bin/env python3
import os
import ast
import importlib.util
import pkgutil

def find_python_files(directory):
    """
    Recursively find all Python files in the given directory.
    
    Args:
        directory (str): Path to the directory to search
    
    Returns:
        list: List of full paths to Python files
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path):
    """
    Extract imports from a Python file.
    
    Args:
        file_path (str): Path to the Python file
    
    Returns:
        set: Set of imported module names
    """
    imports = set()
    try:
        with open(file_path, 'r') as file:
            tree = ast.parse(file.read())
            
            # Collect import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.add(n.name.split('.')[0])
                
                # Collect from ... import statements
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return imports

def filter_rospy_dependencies(imports):
    """
    Filter out ROS and system-related dependencies.
    
    Args:
        imports (set): Set of imported module names
    
    Returns:
        set: Filtered set of potential package dependencies
    """
    # List of standard library and known system modules to filter out
    std_libs = {
        'os', 'sys', 'math', 're', 'json', 'yaml', 'xml', 
        'datetime', 'time', 'logging', 'traceback', 'random',
        'subprocess', 'argparse', 'collections', 'enum', 
        'typing', 'functools', 'itertools', 'pathlib'
    }
    
    # ROS-related modules to filter out
    ros_modules = {
        'rospy', 'std_msgs', 'geometry_msgs', 'sensor_msgs', 
        'nav_msgs', 'actionlib', 'tf', 'message_filters'
    }
    
    # Filter out standard libraries, ROS modules, and built-in modules
    filtered_imports = set()
    for imp in imports:
        # Check if it's not a standard library or ROS module
        if (imp not in std_libs and 
            imp not in ros_modules and 
            not imp.startswith('_') and 
            not hasattr(pkgutil.get_loader(imp), 'get_filename')):
            filtered_imports.add(imp)
    
    return filtered_imports

def extract_package_dependencies(directory):
    """
    Extract dependencies from all Python files in a ROS package directory.
    
    Args:
        directory (str): Path to the ROS package directory
    
    Returns:
        set: Set of package dependencies
    """
    # Find all Python files
    python_files = find_python_files(directory)
    
    # Collect all imports
    all_imports = set()
    for file in python_files:
        all_imports.update(extract_imports(file))
    
    # Filter dependencies
    dependencies = filter_rospy_dependencies(all_imports)
    
    return dependencies

def main():
    # Use the current directory or specify a path
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # Extract dependencies
    package_deps = extract_package_dependencies(directory)
    
    # Print results
    print("Package Dependencies:")
    for dep in sorted(package_deps):
        print(f"- {dep}")
    
    print(f"\nTotal unique dependencies found: {len(package_deps)}")

if __name__ == '__main__':
    main()