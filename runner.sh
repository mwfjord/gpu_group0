#!/bin/bash
# chmod +x runner.sh

make cudart

OUTPUT_EXE="cudart"
OUTPUT_PPM="output.ppm"
TIME_LOG="time.log"

RUNS=10
ns=(2 4 8 16 32 64 128 256)

echo "Running $OUTPUT_EXE $RUNS times and saving output to $OUTPUT_PPM"

# Clear time log file if it already exists
> $TIME_LOG

# Run for different samples
for spp in "${ns[@]}"; do
    echo "SPP = $spp" >> $TIME_LOG
    # Run for mean time
    for ((i=1; i<=RUNS; i++)); do
        echo "Run #$i... with spp = $spp"
        ./$OUTPUT_EXE $spp 0 > $OUTPUT_PPM 2>> $TIME_LOG

        if [ $? -ne 0 ]; then
            echo "Execution failed on run #$i. See $TIME_LOG for details."
            exit 1
        fi
    done
    echo "-------------------------------------------------"
    echo "-------------------------------------------------" >> $TIME_LOG
done

echo "Output saved to $OUTPUT_PPM"
echo "Time log saved to $TIME_LOG"