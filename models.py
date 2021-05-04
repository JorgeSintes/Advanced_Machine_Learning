#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:23:47 2021

@author: yorch
"""

import numpy as np
import pandas as pd 
import math
import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.distributions import Distribution
from typing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############################################################################
############################### DISTRIBUTIONS ################################
##############################################################################

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return torch.add(torch.mul(self.sample_epsilon(),self.sigma),self.mu) # <- your code
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        return -torch.log(self.sigma*math.sqrt(2*math.pi)) - 0.5*((z-self.mu)/self.sigma)**2 # <- your code
        # return -k/2*math.log(2*math.pi) - 1/2*torch.log(torch.prod(self.sigma)) - 1/2 * ((z-self.mu)/self.sigma) @ (z - self.mu)


##############################################################################
########################### VARIATIONAL INFERENCE ############################
##############################################################################

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))
        #print(f'log_px = {log_px}')
        #print(f'log_pz = {log_pz}')
        #print(f'log_qz = {log_qz}')

        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl# <- your code here
        beta_elbo = log_px - self.beta*kl# <- your code here
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs


##############################################################################
#################################### PRINT ###################################
##############################################################################

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print('x: ', x)
        print('Shape: ', x.shape)
        return x


##############################################################################
################################### MODEL 1 ##################################
##############################################################################

# Clément's Magical Creation TM

class CMC(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers):
        super(CMC, self).__init__()
        
        self.input_shape = input_shape
        self.input_size = 1
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        jump = int((hidden_size - 1) / 3)
        layer1 = hidden_size - jump
        layer2 = hidden_size - 2*jump

        
        # or:
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True)
        
        # self.fc = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=layer1),
            nn.ReLU(),
            nn.Linear(in_features=layer1, out_features=layer2),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=layer2, out_features=1) # <- note the 2*latent_features
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        
        x = x.reshape(x.size(0), x.size(1), 1)
        # Forward propagate RNN
        out, _ = self.gru(x)
                
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
         
        out = self.fc(out)
        
        out = self.sig(out)

        return out


##############################################################################
################################### MODEL 2 ##################################
##############################################################################

# Vanilla VAE

class Model2(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
        
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int, num_layers:int) -> None:
        super(Model2, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        
        jump = int((self.observation_features - 2*latent_features) / 3)
        layer1 = self.observation_features - jump
        layer2 = self.observation_features- 2*jump
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.observation_features, out_features=layer1),
            nn.ReLU(),
            nn.Linear(in_features=layer1, out_features=layer2),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=layer2, out_features=2*latent_features) # <- note the 2*latent_features
        )
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=layer2),
            nn.ReLU(),
            nn.Linear(in_features=layer2, out_features=layer1),
            nn.ReLU(),
            nn.Linear(in_features=layer1, out_features=2*self.observation_features)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features]))) # It's a vector of zeros, first bunch of latent_features is
                                                                                              # for the mean, second for the variance. 0s means 1s really cause we
                                                                                              # will input them as the log probabilities when we call the gaussian.
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:        
        """return the distribution `p(x|z)`"""
        px = self.decoder(z)
        
        mu, log_sigma =  px.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = N(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}
    

##############################################################################
################################### MODEL 3 ##################################
##############################################################################

# GRU VAE Encoder only

class Model3(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int, num_layers:int) -> None:
        super(Model3, self).__init__()

        self.input_shape = input_shape
        self.input_size = 1
        self.latent_features = latent_features
        # input shape should be a list, np.prod returns the product of its elements
        self.observation_features = np.prod([self.input_size,sequence_length])
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # jump = int((hidden_size - 2*latent_features) / 3)
        # layer1 = hidden_size - jump
        # layer2 = hidden_size - 2*jump

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.gru = nn.GRU(self.input_size, 2*self.latent_features, num_layers, batch_first=True)

        # self.encoder = nn.Sequential(
        #     nn.Linear(in_features=hidden_size, out_features=layer1),
        #     nn.ReLU(),
        #     nn.Linear(in_features=layer1, out_features=layer2),
        #     nn.ReLU(),
        #     # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
        #     nn.Linear(in_features=layer2, out_features=2*self.latent_features), # <- note the 2*latent_features
        # )
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2*self.observation_features)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        # Here everything is only a vector of 0 as we store log_sigma and not sigma
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        x, _ = self.gru(x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]
        x = x.reshape(x.size(0), 2*self.latent_features)

        # h_x = self.encoder(x)
        mu, log_sigma =  x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        mu, log_sigma =  px_logits.chunk(2, dim=-1)

        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input # x.size(0) is equivalent to the batch size
        x = x.reshape(x.size(0), self.sequence_length, self.input_size)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


##############################################################################
################################### MODEL 4 ##################################
##############################################################################

# GRU-VAE Encoder(GRU+Linear) - sampling trick -  Decoder(GRU)

class Model4(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int,num_layers:int) -> None:
        super(Model4, self).__init__()

        self.input_shape = input_shape
        self.input_size = 1
        self.latent_features = latent_features
        # input shape should be a list, np.prod returns the product of its elements
        self.observation_features = np.prod([self.input_size,sequence_length])
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        jump = int((hidden_size - 2*latent_features) / 3)
        layer1 = hidden_size - jump
        layer2 = hidden_size - 2*jump

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        #self.prnt = PrintLayer()
        self.gru_enc = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=layer1),
            nn.ReLU(),
            nn.Linear(in_features=layer1, out_features=layer2),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=layer2, out_features=2*latent_features) # <- note the 2*latent_features
        )

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`

        self.gru_dec = nn.GRU(latent_features, 2*self.observation_features, num_layers, batch_first=True)
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        # Here everything is only a vector of 0 as we store log_sigma and not sigma
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x, _ = self.gru_enc(x)
        h_x = h_x[:,-1,:]
        h_x = h_x.reshape(x.size(0),-1)

        x = self.encoder(h_x)
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        mu, log_sigma =  x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        z = z.view(z.size(0),-1, self.latent_features)
        z = torch.repeat_interleave(z,repeats = self.sequence_length, dim = 1)
        z, _ = self.gru_dec(z)
        
        z = z[:,-1,:]
        z = z.reshape(z.size(0),-1)

        mu, log_sigma =  z.chunk(2, dim=-1)
        mu = mu.reshape(z.size(0),-1)
        log_sigma = log_sigma.reshape(z.size(0),-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input # x.size(0) is equivalent to the batch size
        x = x.view(-1,self.sequence_length, self.input_size)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


##############################################################################
################################### MODEL 5 ##################################
##############################################################################

# GRU-VAE both encoder and decoder
# There's two versions, one keeping hidden_size of decoder BIG and taking the last one only,
# and another one keeping the hidden_size small and taking all hidden states

# Right now it's the first one, need to change it to the other one?

class Model5(nn.Module):
#class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int,num_layers:int) -> None:
        super(Model5, self).__init__()

        self.input_shape = input_shape
        self.input_size = 1
        self.latent_features = latent_features
        # input shape should be a list, np.prod returns the product of its elements
        self.observation_features = np.prod([self.input_size,sequence_length])
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        jump1 = int((hidden_size - 2*latent_features) / 3)
        layer1_1 = hidden_size - jump1
        layer1_2 = hidden_size - 2*jump1
        
        jump2 = int((sequence_length - latent_features) / 3)
        layer2_1 = latent_features + jump2
        layer2_2 = latent_features + 2*jump2

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        #self.prnt = PrintLayer()
        self.gru = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=layer1_1),
            nn.ReLU(),
            nn.Linear(in_features=layer1_1, out_features=layer1_2),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=layer1_2, out_features=2*latent_features) # <- note the 2*latent_features
        )
        

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            #nn.GRU(latent_features, hidden_size, num_layers, batch_first=True, dropout = 0.2)
            nn.Linear(in_features=latent_features, out_features=layer2_1),
            nn.ReLU(),
            nn.Linear(in_features=layer2_1, out_features=layer2_2),
            nn.ReLU(),
            nn.Linear(in_features=layer2_2, out_features=sequence_length)
        )

        self.gru_dec = nn.GRU(self.input_size, 2*self.input_size, num_layers, batch_first=True)
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        # Here everything is only a vector of 0 as we store log_sigma and not sigma
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        x, _ = self.gru(x)
        x = x[:,-1,:]
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = x.reshape(x.size(0),-1)

        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        z = self.decoder(z)
        z = z.view(z.size(0),-1, self.input_size)
        z, _ = self.gru_dec(z)
        
        px_logits = z.reshape(z.size(0),-1)
        mu, log_sigma =  px_logits.chunk(2, dim=-1)
        # We get the probability of getting a white or black pixels
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        x = x.view(-1,self.sequence_length, self.input_size)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}



##############################################################################
################################### MODEL 6 ##################################
##############################################################################

class Model6(nn.Module):
#class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int,num_layers:int) -> None:
        super(Model6, self).__init__()

        self.input_shape = input_shape
        self.input_size = 1
        self.latent_features = latent_features
        # input shape should be a list, np.prod returns the product of its elements
        self.observation_features = np.prod([self.input_size,sequence_length])
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder_gru = nn.GRU(self.input_size, 2*self.latent_features, num_layers, batch_first=True)

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`

        self.decoder_gru = nn.GRU(self.latent_features, 2*sequence_length, num_layers, batch_first=True)
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        # Here everything is only a vector of 0 as we store log_sigma and not sigma
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        x, _ = self.encoder_gru(x)
        x = x[:,-1,:]
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = x.reshape(x.size(0),-1)

        mu, log_sigma =  x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        # We repeat the latent space observation sequence_length times to feed it to the gru
        z = z.view(z.size(0), 1, self.latent_features)
        z = torch.repeat_interleave(z, repeats = self.sequence_length, dim = 1)
        
        # * before self is to unpack tuple or list elements
        
        z, _ = self.decoder_gru(z)

        z = z[:,-1,:]
        
        mu, log_sigma =  z.chunk(2, dim=-1)
        mu = mu.reshape(z.size(0),-1)
        log_sigma = log_sigma.reshape(z.size(0),-1)
        
        # We get the probability of getting a white or black pixels
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input # x.size(0) is equivalent to the batch size
        x = x.view(-1,self.sequence_length, self.input_size)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}


##############################################################################
################################### ARTHUR ###################################
##############################################################################

class Arthur(nn.Module):
#class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int,num_layers:int, batch_size = 100) -> None:
        #super(VariationalAutoencoder, self).__init__()
        super(Arthur, self).__init__()

        self.input_shape = input_shape
        self.input_size = 1
        self.latent_features = latent_features
        # input shape should be a list, np.prod returns the product of its elements
        self.observation_features = np.prod([input_shape,sequence_length])
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        jump = int((latent_features-1-2*self.input_size)/3)
        layer1 = latent_features - 1 - jump
        layer2 = latent_features - 1 - 2*jump
        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        #self.prnt = PrintLayer()
        self.gru_enc = nn.GRU(self.input_size, 2*latent_features, num_layers, batch_first=True, dropout = 0.2)

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`

        self.gru_dec = nn.GRU(self.input_size, latent_features-1, num_layers, batch_first=True, dropout = 0.2)
        
        self.lin_dec = nn.Sequential(
            nn.Linear(in_features=latent_features-1, out_features=layer1),
            nn.ReLU(),
            nn.Linear(in_features=layer1, out_features=layer2),
            nn.ReLU(),
            # A Gaussian is fully characterised by its mean \mu and variance \sigma**2
            nn.Linear(in_features=layer2, out_features=2*self.input_size) # <- note the 2*latent_features
        )

        # define the parameters of the prior, chosen as p(z) = N(0, I)
        # Here everything is only a vector of 0 as we store log_sigma and not sigma
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        #self.prnt(x)
        x, _ = self.gru_enc(x)
        h_x = x[:,-1,:]
        #out: (n, 28, 128)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        h_x = h_x.reshape(x.size(0),-1)

        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        mus = torch.empty((self.batch_size, self.sequence_length, self.input_size), device='cuda:0')
        log_sigmas = torch.empty((self.batch_size, self.sequence_length, self.input_size), device='cuda:0')

        # Extracting first element of each sequence for each batch
        # Done before reshapping to match input dims of GRU (batch_size,seq_len,inp_size)
        
        z = z.reshape(-1, self.batch_size, self.latent_features)
        x_init = z[:,:,:self.input_size]
        x_init = x_init.reshape(self.batch_size, -1, self.input_size)
        # Extracting only starting hidden state without initial point
        # Required dimensions (seq_len,batch_size,inp_size)
        z = z[:,:,self.input_size:].contiguous()
        _ , h = self.gru_dec(x_init,z)

        # Getting copy here to avoid reshapping later
        h_copy = h.reshape(self.batch_size,-1)
        
        # Downscaling copy to obtain mu and sigma
        musig = self.lin_dec(h_copy) 
        mu, log_sigma = musig.chunk(2, dim=-1)

        mus[:,0,:] = mu
        log_sigmas[:,0,:] = log_sigma

        # reshaping mu to fit required input dimensions of GRU
        mu = mu.reshape(self.batch_size,-1,self.input_size) 

        for i in range(self.sequence_length-1):
          
          _, h = self.gru_dec(mu,h)
          h_copy = h.reshape(self.batch_size,-1)

          musig = self.lin_dec(h_copy) 
          mu, log_sigma =  musig.chunk(2, dim=-1)
          mus[:,i+1,:] = mu
          log_sigmas[:,i+1,:] = log_sigma

          # reshaping mu to fit required input dimensions of GRU
          mu = mu.reshape(self.batch_size,-1,self.input_size)
        
        mus = mus.reshape(self.batch_size,-1)
        log_sigmas = log_sigmas.reshape(self.batch_size,-1)
        #px_logits = px_logits.view(-1, *self.input_shape) # reshape the output
        #print(f'mus = {mus.shape}')
        #print(f'log_sigmas = {log_sigmas.shape}')
        # We get the probability of getting a white or black pixels
        return ReparameterizedDiagonalGaussian(mus, log_sigmas)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input # x.size(0) is equivalent to the batch size
        #x = x.view(x.size(0), -1)
        x = x.view(-1,self.sequence_length, self.input_size)
        self.batch_size = x.size(0)
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, x_init=None, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}
    
    
##############################################################################
################################### BETTY ####################################
##############################################################################

class Betty(nn.Module):
#class VariationalAutoencoder(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int, hidden_size:int, sequence_length:int,num_layers:int, batch_size = 100) -> None:
        #super(VariationalAutoencoder, self).__init__()
        super(Betty, self).__init__()

        self.input_shape = input_shape
        self.input_size = 1
        self.latent_features = latent_features
        # input shape should be a list, np.prod returns the product of its elements
        self.observation_features = np.prod([input_shape,sequence_length])
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        #self.prnt = PrintLayer()
        self.gru_enc = nn.GRU(self.input_size, 2*latent_features, num_layers, batch_first=True, dropout = 0.2)

        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`

        self.gru_dec = nn.GRU(latent_features - 2*self.input_size, 2*self.input_size, num_layers, batch_first=True, dropout = 0.2)
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        # Here everything is only a vector of 0 as we store log_sigma and not sigma
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        #self.prnt(x)
        x, _ = self.gru_enc(x)
        h_x = x[:,-1,:]
        #out: (n, 28, 128)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        h_x = h_x.reshape(x.size(0),-1)

        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""

        # Extracting first element of each sequence for each batch
        # Done before reshapping to match input dims of GRU (batch_size,seq_len,inp_size)
        
        z = z.reshape(self.batch_size, 1, self.latent_features)
        
        h_init = z[:,:,:2*self.input_size].contiguous()
        h_init = h_init.reshape(1, self.batch_size, -1)
        
        z_init = z[:,:,2*self.input_size:]
        z_init = torch.repeat_interleave(z_init,repeats = self.sequence_length, dim = 1)
        

        out , _ = self.gru_dec(z_init,h_init)

        
        mu, log_sigma = out.chunk(2, dim=-1)
        
        mu = mu.reshape(self.batch_size, self.sequence_length)
        log_sigma = log_sigma.reshape(self.batch_size, self.sequence_length)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # flatten the input # x.size(0) is equivalent to the batch size
        #x = x.view(x.size(0), -1)
        x = x.view(-1,self.sequence_length, self.input_size)
        self.batch_size = x.size(0)
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, x_init=None, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}