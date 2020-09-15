# AlphaGo Zero Minimal Example


## Description
The RL designer aims to keep your RL project organized with a simple UI

## Getting Started with UI Mode

- Demonstration of UI is below : main page, training page and inference page

![alt text](images/all_pages.png)

- Training process : 
   1. Select the algorithm class
   2. Select the environment
   3. Select the agent hyperparameters yaml file under the ```Algorithms/Algorithm_Type/Algorithm_Class/configs``` folder
   4. Select trained agent folder under the ```Agents/``` (THIS FEATURE IS NOT READY YET)
      (This feature only being considered when the checkbox is activated)
   5. ```To watch training process live```, open tensorboard by the command below under the main directory:
      - ```tensorboard --logdir=runs```
      - That will open a port on the localhost, and you can compare your all trained agents here !
   6.  ```After the training is finished``` RL designer would create a folder named after algorithm class and the current date under
      the ```/Agents``` folder

- Inference process :
   1. Select the algorithm class
   2. Select the environment that selected agent trained on
   3. Select trained agent folder

## Getting Started with Classical Mode

1. Using standalone mode
   ```python3 Algorithm_Folder/algorithm.py```

That will train the agent and will create  '/Agents' folder under main directory.

Please check 'algorithm_name.yaml' for each of the algorithm.

## Installation
1. Clone the project via command below
   - ``` git clone https://github.com/frknayk/Reinforcement-Learning-Designer.git ```

2. Install the project
   - ``` pip3 install -e . ```
3. Install necessary packages of UI
   - ```cd Designer/agent_designer```
   - Follow the instructions README file there
4. Install necessary packages of the project
   - ```pip3 install requirements.txt``` 
   
## LIMITATIONS
For now the RL designer only supports DQN and DDQN algorithms from value based methods. But in the near future
I will re-integrate dynamic programming methods, cross entropy methods and policy methods.



## Prerequisites
   1. pytorch 
   2. numpy
   4. tensorboard
   5. tensorboardX
