import argparse
import torch
import os
import sys
dirName = os.path.dirname(__file__)
from gym.spaces import Box, Discrete
from torch.autograd import Variable
from utils.buffer import ReplayBuffer
from algorithms.attention_sac import AttentionSAC
from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnv import *
from gym import spaces
import json 

getPosFromAgentState = lambda state: np.array([state[0], state[1]])


def main():
    debug = 1
    if debug:
        numWolves = 3
        numSheep = 1
        numBlocks = 2
        sheepSpeedMultiplier = 1
        individualRewardWolf = 0
        costActionRatio = 0.0

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = int(condition['numWolves'])
        numSheep = int(condition['numSheeps'])
        numBlocks = int(condition['numBlocks'])

        sheepSpeedMultiplier = float(condition['sheepSpeedMultiplier'])
        individualRewardWolf = float(condition['individualRewardWolf'])
        costActionRatio = float(condition['costActionRatio'])


    modelName = "maac{}wolves{}sheep{}blocksSheepSpeed{}WolfActCost{}individ{}".format(
        numWolves, numSheep, numBlocks, sheepSpeedMultiplier, costActionRatio, individualRewardWolf)


    n_rollout_threads = 1
    buffer_length = int(1e6)
    n_episodes = 60000
    episode_length = 75
    steps_per_update = 100
    num_updates = 4
    batch_size = 1024
    save_interval = 1000
    pol_hidden_dim = 128
    critic_hidden_dim = 128
    attend_heads = 4
    pi_lr = 0.001
    q_lr =0.001
    tau = 0.001
    gamma =0.99
    reward_scale =100.

    numAgents = numWolves + numSheep
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.3
    sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheep + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    collisionReward = 10
    isCollision = IsCollision(getPosFromAgentState)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision,
                              punishForOutOfBound, collisionPunishment=collisionReward)

    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward, individualRewardWolf)
    reshapeAction = ReshapeAction()
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

    model_dir = os.path.join(dirName, 'models', 'chasing')
    model = AttentionSAC.init_from_env(envActionSpace, envObservationSpace,
                                       tau=tau, pi_lr=pi_lr, q_lr=q_lr,
                                       gamma=gamma, pol_hidden_dim=pol_hidden_dim,#128
                                       critic_hidden_dim=critic_hidden_dim,#128
                                       attend_heads=attend_heads, #4
                                       reward_scale=reward_scale)
    replay_buffer = ReplayBuffer(buffer_length, model.nagents,
                                 [obsp[0] if isinstance(obsp, tuple) else obsp.shape[0] for obsp in envObservationSpace],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in envActionSpace])
    t = 0

    for ep_i in range(0, n_episodes, n_rollout_threads): #12
        print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + n_rollout_threads, n_episodes))
        state = reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(episode_length):
            obs = observe(state)
            obs = np.array([obs])
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(model.nagents)]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
            action = actions[0]
            nextState = transit(state, action)
            next_obs = np.array([observe(nextState)])
            rewards = np.array([rewardFunc(state, action, nextState)])
            dones = np.array([isTerminal(nextState)])

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            state = nextState
            t += n_rollout_threads
            if (len(replay_buffer) >= batch_size and (t % steps_per_update) < n_rollout_threads): # 100 steps across rollouts -> 4 updates
                model.prep_training(device='cpu')

                for u_i in range(num_updates): #4
                    sample = replay_buffer.sample(batch_size)
                    model.update_critic(sample)
                    model.update_policies(sample)
                    model.update_all_targets()

                model.prep_rollouts(device='cpu')

        if ep_i % save_interval < n_rollout_threads:
            model.prep_rollouts(device='cpu')
            model.save(os.path.join(model_dir, modelName+ 'eps'+str(ep_i)))

    model.save(os.path.join(model_dir, modelName))


if __name__ == '__main__':
    main()
