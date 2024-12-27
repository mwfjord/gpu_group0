#!/bin/bash
# chmod +x compile_and_run.sh

# define the CUDA source file and output executable name
CUDA_FILE="main.cu"
OUTPUT_EXE="output.exe"
OUTPUT_PPM="output.ppm"

# path to the Visual Studio cl.exe (modify as needed)
MSVC_PATH="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.35.32215/bin/Hostx64/x64/cl.exe"

# step 1: compile the CUDA source file
echo "Compiling CUDA file: $CUDA_FILE"
nvcc $CUDA_FILE -ccbin "$MSVC_PATH" -o $OUTPUT_EXE

if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

echo "Compilation successful. Executable: $OUTPUT_EXE"

# step 2: run the executable and collect output in a .ppm file
echo "Running executable and saving output to $OUTPUT_PPM"
./$OUTPUT_EXE > $OUTPUT_PPM

if [ $? -ne 0 ]; then
    echo "Execution failed."
    exit 1
fi

echo "Output saved to $OUTPUT_PPM"

# clean up the generated .exp and .lib files (if you don't need them)
rm -f *.exp *.lib

echo "Cleaned up build files"
