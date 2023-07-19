//
// Created by vishnu mohan on 15/12/2022.
//


#include "Neuron.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include<cmath>
#include <random>

using namespace std;
// using namespace std::chrono;

Neuron::Neuron(char kind){

    random_device rd;     // Only used once to initialise (seed) engine
    mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
    uniform_real_distribution<float> uni(0,1); // Guaranteed unbiased
    rnd = uni(rng);

    // Excitatory Neurons
    if (kind == 'e'){
        a = 0.02;
        b = 0.2;
        c = -65 + (15 * pow(rnd, 2));
        d = 8 - (6 * pow(rnd, 2));
        v = -65;
    }

    // Inhibitory Neurons
    if (kind == 'i'){
        a = 0.02 + (0.08 * rnd);
        b = 0.25 - (0.05 * rnd);
        c = -65;
        d = 2;
        v = -65;
    }
}
