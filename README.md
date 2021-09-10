# Project 3: Collaboration and Competition

## Environment Details
![Trained agent playing tennis gif](./images/trained_agent.gif "Trained Agent playing Tennis")

### Problem Description
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

### State and Action Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solving Condition
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
   - Linux: [(click here)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Mac OSX: [(click here)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [(click here)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): [(click here)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    Place the file in the `unity_env/` folder, and unzip (or decompress) the file.

1. Install the pre-requisites using conda:
    ```
    conda env create --name drlnd_navigation --file=environment.yml
    ```

1. Create jupyter-notebook kernel
    ```
   python -m ipykernel install --user --name drlnd_navigation --display-name "drlnd_navigation"
    ```
    - Make sure to select kernel drlnd_navigation when executing `jupyter-notebook`

## Instructions
- To run the project using saved weights for the agent, execute the section "Watch Smart Agents!" in the notebook `Tennis.ipynb`
- To train the network, execute the section "Train Agents with MADDPG" in the notebook `Tennis.ipynb`

