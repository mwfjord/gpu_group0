#!/bin/bash
# chmod +x run.sh

OUTPUT_EXE="output.exe"
OUTPUT_PPM="output.ppm"
TIME_LOG="time_cub.log"

RUNS=5

echo "Running $OUTPUT_EXE $RUNS times and saving output to $OUTPUT_PPM"

# Clear time log file if it already exists
> $TIME_LOG

for ((i=1; i<=RUNS; i++)); do
    echo "Run #$i..."
    ./$OUTPUT_EXE > $OUTPUT_PPM 2>> $TIME_LOG

    if [ $? -ne 0 ]; then
        echo "Execution failed on run #$i. See $TIME_LOG for details."
        exit 1
    fi
done

echo "Output saved to $OUTPUT_PPM"
echo "Time log saved to $TIME_LOG"