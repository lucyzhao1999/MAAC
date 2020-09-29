import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('numpy').setLevel(logging.ERROR)

from myMAAC.src import *
from myMAAC.buffer import *
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from visualize.visualizeMultiAgent import *
from environment.chasingEnv.multiAgentEnv import *
from gym import spaces
getPosFromAgentState = lambda state: np.array([state[0], state[1]])


def main():
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
    envObservationSpace = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    worldDim = 2
    envActionSpace = [worldDim * 2 + 1 for agentID in range(numAgents)]

    layersWidths = [128]
    getSAEncodings = GetSAencodings(layersWidths, normInput=True)
    getSEncodings = GetSencodings(layersWidths, normInput= True)

    attentionLayerWidth = 32
    buildValueExtractor = BuildValueExtractor(attentionLayerWidth, activFunc=tf.nn.leaky_relu)
    buildSelectorExtractor = BuildSelectorExtractor(attentionLayerWidth)
    buildKeyExtractor = BuildKeyExtractor(attentionLayerWidth)

    policyLayersWidths = [128, 128]
    reward_scale = 10
    actorLR = 0.01

    session = tf.Session()

    actorList = [Actor(policyLayersWidths, agentObsDim, agentActionDim, reward_scale, actorLR, 'actorTrain'+str(agentID), session)
                 for agentObsDim, agentActionDim, agentID in zip(envObservationSpace, envActionSpace, range(numAgents))]
    actorTargetList = [Actor(policyLayersWidths, agentObsDim, agentActionDim, reward_scale, actorLR, 'actorTarget'+str(agentID), session)
                 for agentObsDim, agentActionDim, agentID in zip(envObservationSpace, envActionSpace, range(numAgents))]

    criticLR = 0.01
    gamma = 0.95
    numAttentionHeads = 4
    hiddenLayersWidth = 128
    critic = Critic(numAgents, envActionSpace, envObservationSpace, hiddenLayersWidth, criticLR, reward_scale, gamma, 'criticTrain', session,
                 numAttentionHeads, getSAEncodings, getSEncodings, buildValueExtractor, buildSelectorExtractor, buildKeyExtractor)

    criticTarget = Critic(numAgents, envActionSpace, envObservationSpace, hiddenLayersWidth, criticLR, reward_scale, gamma, 'criticTarget', session,
                 numAttentionHeads, getSAEncodings, getSEncodings, buildValueExtractor, buildSelectorExtractor, buildKeyExtractor)

    tau = 0.01
    updateParams = UpdateParams(tau, session)
    model = AttentionSAC(actorList, actorTargetList, critic, criticTarget, updateParams)

    sampleOneStep = SampleOneStep(transit, rewardFunc, isTerminal)

    batchSize = 32#fe1024
    numUpdatesPerTrain = 4

    bufferSize = 1e6
    buffer = ReplayBuffer(bufferSize)
    updateInterval = 100
    trainOneStep = TrainOneStep(model, buffer, batchSize, numUpdatesPerTrain, updateInterval)

    maxEpisode = 60000
    maxTimestep = 75

    fileName = 'maac_chasing'
    modelPath = os.path.join(dirName, '..', 'trainedModels', fileName)
    modelSaveRate = 10000
    saveModel = SaveModel(modelSaveRate, saveVariables, modelPath, session, saveAllmodels = False)

    trainMAAC = MAAC(model, buffer, maxEpisode, maxTimestep, reset, sampleOneStep, observe, saveModel, trainOneStep)

    saver = tf.train.Saver(max_to_keep=None)
    tf.add_to_collection("saver", saver)
    # writer = tf.summary.FileWriter('tensorBoard/', graph=session.graph)
    # tf.add_to_collection("writer", writer)

    session.run(tf.global_variables_initializer())

    trainMAAC()


if __name__ == '__main__':
    main()
