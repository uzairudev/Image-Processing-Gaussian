g++ gaussian_filter.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ gaussian_filter.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o gaussian_filter

# Use this to run 
# ./gaussian_filter

