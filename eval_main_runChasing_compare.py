import argparse
import torch
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
from visualize.visualizeMultiAgent import *

getPosFromAgentState = lambda state: np.array([state[0], state[1]])


def calcWolfTrajBiteAmount(traj, wolvesID, singleReward = 10):
    rewardIDinTraj = 2
    getWolfReward = lambda allAgentsReward: np.sum([allAgentsReward[wolfID] for wolfID in wolvesID])
    rewardList = [getWolfReward(timeStepInfo[rewardIDinTraj]) for timeStepInfo in traj]
    trajReward = np.sum(rewardList)
    biteNumber = trajReward/ singleReward

    return biteNumber

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def makeMultiAgentEnv():
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    run_num = 1

    numWolves = 3
    numSheep = 1
    numBlocks = 2
    numAgents = numWolves + numSheep
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks


    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    model = AttentionSAC.init_from_save(filename= run_dir / 'model.pt')

    biteList = []
    trajListToRender = []
    for ep_i in range(0, config.n_episodes):
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        trajectory = []

        for et_i in range(config.episode_length):#25
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(model.nagents)]
            torch_agent_actions = model.step(torch_obs, explore=False)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            state = [np.append(agent.state.p_pos, agent.state.p_vel) for agent in env.agents] + [np.append(landmark.state.p_pos, landmark.state.p_vel) for landmark in env.world.landmarks]

            state = obs[0]
            action = actions[0]
            reward = rewards[0]
            nextState = next_obs[0]
            trajectory.append((state, action, reward, nextState))

            obs = next_obs

        biteNum = calcWolfTrajBiteAmount(trajectory, wolvesID, singleReward = 10)
        biteList.append(biteNum)
        trajListToRender = trajListToRender + trajectory

        print(biteNum)

    meanTrajBite = np.mean(biteList)
    seTrajBite = np.std(biteList) / np.sqrt(len(biteList) - 1)
    print('meanTrajBite', meanTrajBite, 'seTrajBite ', seTrajBite)

    wolfColor = np.array([0.85, 0.35, 0.35])
    sheepColor = np.array([0.35, 0.85, 0.35])
    blockColor = np.array([0.25, 0.25, 0.25])
    entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheep + [blockColor] * numBlocks
    render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
    trajToRender = np.concatenate(trajListToRender)
    render(trajToRender)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_tag", type=str)
    parser.add_argument("--model_name", default="chasing", type=str)
    parser.add_argument("--n_rollout_threads", default=1, type=int)# 12
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=60, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int, help="Number of updates per update cycle")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)

    config = parser.parse_args()

    run(config)
