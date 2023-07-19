//
// Created by vishnu mohan on 15/12/2022.
//

#ifndef WORKIN_METAL_NEURON_H
#define WORKIN_METAL_NEURON_H

/* Some notes mentioned in the original Izikevich paper :
 * We have used this model to simulate a sparse network of 10000 spiking cortical neurons with 1000000 synaptic connections in real time
 * (resolution 1 ms) using a 1 GHz desktop PC and C++ programming language. The following MATLAB program (also available on author’s webpage)
 * simulates a network of randomly connected 1000 neurons in real time. Motivated by the anatomy of a mammalian cortex, we choose the ratio of excitatory
 * to inhibitory neurons to be 4 to 1, and we make inhibitory synaptic connections stronger. Besides the synaptic input, each neuron receives a noisy
 * thalamic input. In principle, one can use RS cells to model all excitatory neurons and FS cells to model all inhibitory neurons. The best way to
 * achieve heterogeneity (so that different neurons have different dynamics), is to assign each excitatory cell (ai;bi)=(0:02;0:2) and (ci;di)= (65;8)+(15;6)r2 i ,
 * where ri is a random variable uniformly distributed on the interval [0,1], and i is the neuron index. Thus, ri = 0 corresponds to regular spiking (RS) cell,
 * and ri =1 corresponds to the chattering (CH) cell. We use r2 i to bias the distribution toward RS cells. Similarly, each inhibitory cell has (ai;bi)=(0:02;0:25) + (0:08;0:05)ri and (ci;di)=(65;2).
 * The model belongs to the class of pulse-coupled neural networks (PCNN): The synaptic connection weights between the neurons are given by the matrix S =(sij),
 * so that firing of the jth neuron instantaneously changes variable vi by sij .
 * */


class Neuron {

public:
    Neuron(char kind);

    // 0 - exc; 1 - inh
    int type;

    /* a - time scale of the recovery variable u. Typical val is 0.02
     * b - describes the sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v. Typical val is 0.2
     * c - describes the after-spike reset value of the membrane potential v caused by the fast high-threshold K+ conductances. A typical value is c = 65 mV
     * d - describes after-spike reset of the recovery variable ucausedbyslowhigh-thresholdNa+ andK+ conductances. Atypical value is d = 2.
     * I - Input to the neuron
     * S - The synaptic connection weights between the neurons are given by the matrix S =(sij), so that firing of the
     *     jth neuron instantaneously changes variable vi by sij .
     * */
    float v, u, a = 0.02, b = 0.2, c, d, S, I, rnd;
    double spike_timestamps[];
};


#endif //WORKIN_METAL_NEURON_H
