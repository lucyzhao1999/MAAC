import argparse
import torch
import os
import sys
dirName = os.path.dirname(__file__)
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnv import *
from gym import spaces

getPosFromAgentState = lambda state: np.array([state[0], state[1]])


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
    from multiagent.environmentForChasing import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def run(config):

    numWolves = 4
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
    sheepMaxSpeed = 1.3
    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheep + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment=collisionReward)

    individualRewardWolf = 0
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
    reshapeAction = ReshapeAction()
    costActionRatio = 0
    getActionCost = GetActionCost(costActionRatio, reshapeAction, individualCost=True)
    getWolvesAction = lambda action: [action[wolfID] for wolfID in wolvesID]
    rewardWolfWithActionCost = lambda state, action, nextState: np.array(
        rewardWolf(state, action, nextState)) - np.array(getActionCost(getWolvesAction(action)))

    rewardFunc = lambda state, action, nextState: \
        list(rewardWolfWithActionCost(state, action, nextState)) + list(rewardSheep(state, action, nextState))

    reset = ResetMultiAgentChasing(numAgents, numBlocks)
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState,
                                              getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)

    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    isTerminal = lambda state: [False]* numAgents

    initObsForParams = observe(reset())
    envObservationSpace = [initObsForParams[obsID].shape for obsID in range(len(initObsForParams))]

    worldDim = 2
    envActionSpace = [spaces.Discrete(worldDim * 2 + 1) for agentID in range(numAgents)]

    model_dir = os.path.join(dirName, 'models', config.env_id, config.model_name)
    model = AttentionSAC.init_from_env(envActionSpace, envObservationSpace,
                                       tau=config.tau, pi_lr=config.pi_lr, q_lr=config.q_lr,
                                       gamma=config.gamma, pol_hidden_dim=config.pol_hidden_dim,#128
                                       critic_hidden_dim=config.critic_hidden_dim,#128
                                       attend_heads=config.attend_heads, #4
                                       reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp[0] if isinstance(obsp, tuple) else obsp.shape[0] for obsp in envObservationSpace],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in envActionSpace])
    t = 0

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads): #12
        print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes))
        state = reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            obs = observe(state)
            obs = np.array([obs])
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(model.nagents)]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            action = actions[0]
            nextState = transit(state, action)
            next_obs = np.array([observe(nextState)])
            rewards = np.array([rewardFunc(state, action, nextState)])
            dones = np.array([isTerminal(nextState)])

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            state = nextState
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads): # 100 steps across rollouts -> 4 updates
                model.prep_training(device='cpu')

                for u_i in range(config.num_updates): #4
                    sample = replay_buffer.sample(config.batch_size)
                    model.update_critic(sample)
                    model.update_policies(sample)
                    model.update_all_targets()

                model.prep_rollouts(device='cpu')

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            pathIncremental = os.path.join(model_dir, 'incremental')
            if not os.path.exists(pathIncremental):
                os.makedirs(pathIncremental)
            model.save(os.path.join(pathIncremental, ('model_ep%i.pt' % (ep_i + 1))))

    model.save(os.path.join(model_dir, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_tag", type=str)
    parser.add_argument("--model_name", default="chasing", type=str)
    parser.add_argument("--n_rollout_threads", default=1, type=int)# 12
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=60000, type=int)
    parser.add_argument("--episode_length", default=75, type=int)
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
