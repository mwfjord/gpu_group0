#include "random_utils.h"

// define random device, generator, and distribution only once
std::random_device rd;
std::mt19937 gen(rd()); // initialize Mersenne Twister RNG with random seed
std::uniform_real_distribution<> dis(0, 1.0); // uniform distribution between -1 and 1

// This function replaces the Linux only drand48() function to one that works on multiple systems (like Windows)
// generates a random number between 0 and 1
double random_double() {
    return dis(gen);  
}
