#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory of the script's directory
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Set the directory where yaml-cpp should be cloned and built, relative to the parent directory
DEPENDENCY_DIR="$WORKSPACE_DIR/dependencies"
YAML_CPP_DIR="$DEPENDENCY_DIR/yaml-cpp"
YAML_CPP_VERSION="yaml-cpp-0.8.0"  # Specify the version you want to checkout

# Create the directory if it doesn't exist
mkdir -p $DEPENDENCY_DIR

# Check if yaml-cpp is already cloned
if [ ! -d "$YAML_CPP_DIR" ]; then
    echo "Cloning yaml-cpp..."
    git clone https://github.com/jbeder/yaml-cpp.git $YAML_CPP_DIR
    cd $YAML_CPP_DIR
    git checkout $YAML_CPP_VERSION
else
    echo "yaml-cpp already cloned, checking out the specified version..."
    cd $YAML_CPP_DIR
    git fetch --tags
    git checkout $YAML_CPP_VERSION
    cd -
fi

# Build yaml-cpp
echo "Building yaml-cpp..."
cd $YAML_CPP_DIR
mkdir -p build
cd build
cmake ..
make

# Optional: Copy the built files to a project-specific directory
INSTALL_DIR="$WORKSPACE_DIR/libs"
INCLUDE_DIR="$WORKSPACE_DIR/include"

mkdir -p $INSTALL_DIR
mkdir -p $INCLUDE_DIR

echo "Copying built files..."
cp libyaml-cpp.a $INSTALL_DIR/
cp -r ../include/yaml-cpp $INCLUDE_DIR/

# Return to the root directory
cd $WORKSPACE_DIR

echo "Removing the cloned yaml-cpp directory..."
rm -rf $YAML_CPP_DIR

echo "yaml-cpp successfully installed."
echo "Dependency installation complete."
