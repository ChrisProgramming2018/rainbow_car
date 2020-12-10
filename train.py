import os
import sys 
import cv2
import gym
import time
import torch 
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime
from memory import ReplayMemory
from utils import time_format, eval_policy, mkdir
from agent import Agent
from framestack import FrameStack


def train_agent(env, args, config):
    """
    Args:
    """
    
    # create CNN convert the [1,3,84,84] to [1, 200]
    now = datetime.now()    
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    pathname = dt_string + "_seed" + str(config["seed"])
    print("save tensorboard {}".format(config["locexp"]))
    tensorboard_name = str(config["locexp"]) + '/runs/' + pathname 
    agent = Agent(args, env)
    memory = ReplayMemory(args, args.memory_capacity)
    #memory =  ReplayBuffer((3, config["size"], config["size"]), (1,), config["expert_buffer_size"], int(config["image_pad"]), config["device"])
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    writer = SummaryWriter(tensorboard_name)
    results_dir = os.path.join(str(config["locexp"]), args.id) 
    mkdir("", results_dir)
    scores_window = deque(maxlen=100)
    scores = [] 
    t0 = time.time()
    # Training loop
    agent.train()
    T, done = 0, True
    print("result dir ", results_dir)
    agent.save(results_dir, 'checkpoint-{}.pth'.format(T))
    #eval_policy(env, agent, writer, T, config)
    episode = -1
    steps = 0
    score= 0
    print("save policy ", args.checkpoint_interval)
    # eval_policy(env, agent, writer, 0, config)
    for T in range(1, args.T_max + 1):
        # print("\r {} of {}".format(T, args.T_max), end='')
        if done:
            episode += 1
            scores_window.append(score)       # save most recent scor
            scores.append(score)              # save most recent score
            print('\rTime steps {}  episode {} score {} Average Score: {:.2f} time: {}'.format(T, episode, score, np.mean(scores_window), time_format(time.time() - t0)), end="")         
            writer.add_scalar('Episode_reward ', score, T)
            average_reward = np.mean(scores_window)
            writer.add_scalar('Average_reward ', average_reward, T)
            state, done = env.reset(), False
            steps = 0
            score = 0 
        if T % args.replay_frequency == 0:
            agent.reset_noise()  # Draw a new set of noisy weights
        
        
        action = agent.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, _ = env.step(action)  # Step
        score += reward
        steps +=1
        if steps == 30:
            done =True
        memory.append(state, action, reward, done)  # Append transition to memory
        
        # Train and test
        if T >= args.learn_start:
            memory.priority_weight = min(memory.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                agent.learn(memory)  # Train with n-step distributional double-Q learning
            
            # Update target network
            if T % args.target_update == 0:
                agent.update_target_net()
            
            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                print("Eval policy")
                eval_policy(env, agent, writer, T, config)
                agent.save(results_dir, 'checkpoint-{}.pth'.format(T))
        state = next_state


