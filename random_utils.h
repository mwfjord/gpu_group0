#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>

extern std::mt19937 gen; // generator
extern std::uniform_real_distribution<> dis; // distribution

double random_double(); // helper function to get a random double

#endif // RANDOM_UTILS_H
