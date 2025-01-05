#do  10times

for i in {1..10}
do
    echo "Run $i"
    ./cudart 2 > out2.ppm
done

for i in {1..10}
do
    echo "Run $i"
    ./cudart 4 > out4.ppm
done

for i in {1..10}
do
    echo "Run $i"
    ./cudart 8 > out8.ppm
done

for i in {1..10}
do
    echo "Run $i"
    ./cudart 16 > out16.ppm
done

for i in {1..10}
do
    echo "Run $i"
    ./cudart 32 > out32.ppm
done

for i in {1..10}
do
    echo "Run $i"
    ./cudart 64 > out64.ppm
done

for i in {1..10}
do  
    echo "Run $i"
    ./cudart 128 > out128.ppm
done

for i in {1..10}
do
    echo "Run $i"
    ./cudart 256 > out256.ppm
done

