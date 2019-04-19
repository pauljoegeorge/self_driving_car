#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:26:09 2019

@author: pauljoegeroge
"""

import numpy as np
import random
import os #to save/load brain
import torch #to handle dynamic graphs
import torch.nn as nn #all tools to implement nueral n/w
import torch.nn.functional as F
import torch.optim as optim #optimizer
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):  #no of input and output neurons. input is 5, nb_action
        super(Network, self).__init__() #helps to use the tools within the function
        self.input_size = input_size
        self.nb_action = nb_action
        #full connection between input layer and hidden layer and then between hiddhen layer and output layer
        self.fc1 = nn.Linear(input_size, 30) #full connection between input and hidden layer (5 input neurons and 30 hidden neurons)
        self.fc2 = nn.Linear(30, nb_action)
    
        
    #activate neurons, return Q values
    def forward(self, state):
        x = F.relu(self.fc1(state)) #activate hidden neurons
        q_values = self.fc2(x) 
        return q_values
    
    
# Implementing Experience replay (instead of considering current state only it consider past states as well)
        
class ReplayMemory(object):
    
    def __init__(self, capacity): # capacity: no of previous transactions considering
        self.capacity = capacity
        self.memory = [] # previous transactions
        
    def push(self, event): #event is the transaction which will be added to memory.  event is a 4 tuple --> last state, new state, last action and last reward
        self.memory.append(event)
        #make sure no of transactions is same as capacity limit
        if len(self.memory) > self.capacity:   
            del self.memory[0] #delete the first element attached
            
    #get random samples from memory
    def sample(self, batch_size):
        #example: zip(*) ((1,2,3),(4,5,6)) ===> ((1,4),(2,5),(3,6))
        samples = zip(*random.sample(self.memory, batch_size)) #zip* -> reshape the list. self.memory (state, action, reward), doing zip to differentiate 
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #mapped samples to lambda function with parameter x which is samples #to convert it to torch
        
#implementing Deep Q learning    
class Dqn():
    def __init__(self, input_size, nb_action, gamma): #gamma: for delay
        self.gamma = gamma
        self.reward_window = []   #mean of the reward taken form sliding window
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) #capacity
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #connecting Adam optimizer to neural network, lr= learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)   #it needs to be torch tensor with one  more dimension (fake dimension)
        self.last_action = 0 #either 0 1 or 2 then using action2roation to angles of rotation
        self.last_reward = 0
   
    #which action to play
    def select_action(self, state): #output of nueral network (q values of each possible actions)
        #to select best action - we are using softmax
        #T = 0 then no AI 
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) #input = output of q values
        action = probs.multinomial() # give random draw from values
        return action.data[0,0]
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):   
       #unsqueeze(0) -> fake input_size, unsqueeze(1) -> fake action
       outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)  #actions we have decided/ already played , squeeze is to kill the fake dimension
       next_outputs = self.model(batch_next_state).detach().max(1)[0] #de attach all states and fetch the max state , 0 coz its state, 1 is action
       target = self.gamma * next_outputs + batch_reward # reward + target gamma of next outputs
       td_loss = F.smooth_l1_loss(outputs, target) #temporal difference
       self.optimizer.zero_grad() #reinstalize optimizer
       td_loss.backward(retain_variables = True)
       self.optimizer.step() #optimizing
        
    
    def update(self, reward, new_signal):
        #new signal are signals sensors detected
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #converting to torch format, with fake dimension
        #new state and latest_state is also torch sensor but last action is 0 1 or 2. so it has to be changed
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) 
        #got the action , now play the action
        action = self.select_action(new_state) #return action for newly reached state
        if len(self.memory.memory) > 100: 
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)  #first 100 actions
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)  #learn first 100
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:  #limit reward window size to 10000
            del self.reward_window[0]
        return action
        #amT
    #compute mean of all rewards in sliding window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    #save brain of car (saving model and optimizer)
    def save(self):
        #following will save parameter
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),}, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> Loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            # update existing model and parameter of optimizer
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("No checkpoint found")
    
    
        
        
        