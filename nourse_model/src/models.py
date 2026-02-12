"""
Copyright (c) Eon Systems, 2025.
All rights reserved. Unauthorized copying, distribution, or modification
of this file, in whole or in part, is strictly prohibited.
"""
import torch
import torch.nn as nn
import numpy as np

class PoissonSpikeGenerator(nn.Module):
    """
    A module which generates 1 simulation step of Poisson-distributed spikes given input firing rates.

    """
    def __init__(self, dt, scale, device='cpu'):
        """
        Initialize the module

        Args:
            dt (float): Simulation time step in ms
            scale (float): Scaling factor for the output spikes
            device (str, optional): Location for the resulting tensors. Defaults to 'cpu'.
        """
        super(PoissonSpikeGenerator, self).__init__()
        self.dt = dt      # in ms
        self.prob_scale = dt / 1000.0  # Convert rate to probability per time step
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None):
        """
        Call the module to generate spikes

        Args:
            rates (tensor): Tensor of firing rates in Hz
            generator (Pytorch generator, optional): Random number generator for reproduciblity. Defaults to None.

        Returns:
            tensor: Tensor of generated spikes for one step
        """
        spikes = torch.bernoulli(rates*self.prob_scale, generator=generator)*self.scale
        return spikes

class AlphaSynapse(nn.Module):
    """
    A module which simulates alpha-function synapse dynamics, with a delay on incoming spikes.
    """
    def __init__(self, batch, size, dt, params, device='cpu'):
        """
        Initialize the module

        Args:
            batch (int): Batch size (number of parallel simulations)
            size (int): Number of neurons
            dt (float): Simulation time step in ms
            params (dict): Parameter dictionary
            device (str, optional): Location for the resulting tensors. Defaults to 'cpu'.
        """
        super(AlphaSynapse, self).__init__()
        self.time_factor = dt/params['tauSyn']  # precompute dt/tau for efficiency
        self.steps_delay = int(params['tDelay'] / dt)   # compute number of steps for delay, with rounding
        self.size = size
        self.device = device
        self.batch = batch
    
    def state_init(self):
        """
        Get the initial states of the dynamic variables

        Returns:
            tensor: Synaptic conductance tensor
            tensor: Delay buffer tensor
        """
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(self.batch, self.steps_delay+1, self.size, device=self.device)
        return conductance, delay_buffer
    
    def forward(self, input_, conductance, delay_buffer, refrac):
        """
        Call the module to update synapse states

        Args:
            input_ (tensor): Current synaptic input
            conductance (tensor): Previous synaptic conductance
            delay_buffer (tensor): Storage buffer of delayed synaptic inputs
            refrac (tensor): Mask for refractory period (computed outside this module)

        Returns:
            tensor: New conductances
            tensor: Updated delay buffer
        """
        # Trace dynamics
        conductance_new = conductance * (1 - self.time_factor) + delay_buffer[:,0,:] * refrac

        # Update delay buffer
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:,-1,:] = input_
        return conductance_new, delay_buffer

class LIFNeuron(nn.Module):
    """
    A module which simulates Leaky Integrate-and-Fire neuron dynamics with surrogate gradient for spikes.
    """
    def __init__(self, batch, size, dt, params, device='cpu'):
        """
        Initialize the module

        Args:
            batch (int): Batch size (number of parallel simulations)
            size (int): Number of neurons
            dt (float): Simulation time step in ms
            params (dict): Parameter dictionary
            device (str, optional): Storage location for tensors. Defaults to 'cpu'.
        """
        super(LIFNeuron, self).__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params['tauMem'] # Membrane time constant
        self.v_reset = params['vReset'] # Reset potential
        self.v_rest = params['vRest']   # Resting potential
        self.v_threshold = params['vThreshold'] # Firing threshold
        self.v_0 = params['v0'] # Initial membrane potential
        self.time_factor = dt / self.tau_mem    # Precompute dt/tau for efficiency
        self.spike_gradient = self.ATan.apply   # Surrogate gradient function
        self.device = device
        self.batch = batch

    def state_init(self):
        """
        Get the initial states of the dynamic variables

        Returns:
            tensor: Tensor of spikes (all zeros)
            tensor: Tensor of membrane potentials
        """
        v = torch.zeros(self.batch,self.size, device=self.device) + self.v_0  # Initial membrane potential
        spikes = torch.zeros(self.batch,self.size, device=self.device)
        return spikes, v

    def forward(self, input_current, v):
        """
        Call the module to update neuron states

        Args:
            input_current (tensor): Incoming synaptic current to the neurons
            v (tensor): Previous membrane potentials

        Returns:
            tensor: Generated spikes
            tensor: New membrane potentials
        """
        # Update membrane potential
        v = v + self.time_factor * (input_current - (v - self.v_rest))

        # Generate spikes and reset
        spike = self.spike_gradient((v-self.v_threshold))  # call the Heaviside function
        reset = ((v - self.v_reset) * spike).detach()  # remove reset from computational graph
        v = v - reset
        
        return spike, v
    
    """
    Surrogate gradient function for spikes using ArcTan derivative. Based on the method from snnTorch
    """
    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float() # Heaviside on the forward pas
            ctx.save_for_backward(v)  # store the membrane for use in the backward pass
            return spike

        @staticmethod
        def backward(ctx, grad_output):
            (spike,) = ctx.saved_tensors  # retrieve the membrane potential
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad

class AlphaLIF(torch.nn.Module):
    """
    LIF neuron with alpha-function synapse dynamics and refractory period.
    """
    def __init__(self, batch, size, dt, params, device='cpu', *args, **kwargs):
        """
        Initialize the module

        Args:
            batch (int): Batch size (number of parallel simulations)
            size (int): Number of neurons
            dt (float): Simulation time step in ms
            params (dict): Parameter dictionary
            device (str, optional): Device for performing computation. Defaults to 'cpu'.
        """
        super().__init__(*args, **kwargs)
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device, *args, **kwargs)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device, *args, **kwargs)
        self.steps_refrac = int(params['tRefrac']/dt)   # calculate number of steps for refractory period
    
    def state_init(self):
        """
        Return the initial states of all dynamic variables

        Returns:
            tensor: Synaptic conductances
            tensor: Synaptic delay buffer
            tensor: Spikes
            tensor: Membrane potentials
            tensor: Refractory counters
        """
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.steps_refrac + torch.zeros_like(v)
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self, input_, conductance, delay_buffer, spikes, v, refrac):
        """
        Call the module to update all dynamic variables

        Args:
            input_ (tensor): External input (potentially from Poisson generator)
            conductance (tensor): Previous synaptic conductances
            delay_buffer (tensor): Synaptic delay buffer
            spikes (tensor): Previous spikes
            v (tensor): Previous membrane potentials
            refrac (tensor): Refractory counters

        Returns:
            tensor: Synaptic conductances
            tensor: Synaptic delay buffer
            tensor: Spikes
            tensor: Membrane potentials
            tensor: Refractory counters
        """
        refrac = refrac * (1 - spikes)  # reset refractory counter on spike
        refrac = refrac + 1
        conductance_new, delay_buffer = self.synapse(input_, conductance, delay_buffer, (refrac>self.steps_refrac).float())
        spikes, v_new = self.neuron(conductance, v)
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac

class TorchModel(torch.nn.Module):
    """
    High level module combining Poisson spike generator and AlphaLIF neurons with connectome-derived recurrence.
    """
    def __init__(self, batch, size, dt, params, weights, device='cpu', *args, **kwargs):
        """
        Initialize the module

        Args:
            batch (int): Batch size (number of parallel simulations)
            size (int): Number of neurons
            dt (float): Simulation time step in ms
            params (dict): Parameter dictionary
            weights (tensor): Tensor of recurrent weights (ideally sparse, either CSR or COO)
            device (str, optional): Location of computation. Defaults to 'cpu'.
        """
        super().__init__(*args, **kwargs)
        self.neurons = AlphaLIF(batch, size, dt, params, device=device, *args, **kwargs)
        self.weights = weights
        self.poisson = PoissonSpikeGenerator(dt, params['scalePoisson'], device=device)
        self.scale = params['wScale']   # scaling factor for recurrent weights
    
    def state_init(self):
        """
        Get initial states of all dynamic variables

        Returns:
            tensor: Synaptic conductances
            tensor: Synaptic delay buffer
            tensor: Spikes
            tensor: Membrane potentials
            tensor: Refractory counters
        """
        return self.neurons.state_init()
    
    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None):
        """
        Call the module to perform one simulation step

        Args:
            rates (tensor): Tensor of input firing rates for Poisson generator
            conductance (tensor): Previous synaptic conductances
            delay_buffer (tensor): Synaptic delay buffer
            spikes (tensor): Previous spikes
            v (tensor): Previous membrane potentials
            refrac (tensor): Refractory counters
            generator (torch.generator), optional): Random generator for reproducibility. Defaults to None.

        Returns:
            tensor: Synaptic conductances
            tensor: Synaptic delay buffer
            tensor: Spikes
            tensor: Membrane potentials
            tensor: Refractory counters
        """
        spikes_input = self.poisson(rates, generator=generator)
        weighted_spikes = torch.matmul(spikes, self.weights.transpose(0,1))
        conductance, delay_buffer, spikes, v, refrac = self.neurons(self.scale*(spikes_input+ weighted_spikes), conductance, delay_buffer, spikes, v, refrac)
        return conductance, delay_buffer, spikes, v, refrac
    
