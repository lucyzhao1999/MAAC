import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow.contrib.layers as layers
import myMAAC.tf_util as U
from myMAAC.buffer import *
import time

def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


class SaveModel:
    def __init__(self, modelSaveRate, saveVariables, modelSavePath, sess, saveAllmodels = False):
        self.modelSaveRate = modelSaveRate
        self.saveVariables = saveVariables
        self.epsNum = 1
        self.modelSavePath = modelSavePath
        self.saveAllmodels = saveAllmodels
        self.sess = sess

    def __call__(self):
        self.epsNum += 1
        if self.epsNum % self.modelSaveRate == 0:
            modelSavePathToUse = self.modelSavePath + str(self.epsNum) + "eps" if self.saveAllmodels else self.modelSavePath
            with self.sess.as_default():
                self.saveVariables(self.sess, modelSavePathToUse)


class GetSAencodings:
    def __init__(self, layersWidths, normInput = True, activFunc = tf.nn.leaky_relu):
        self.layersWidths = layersWidths
        self.normInput = normInput
        self.activFunc = activFunc

    def __call__(self, obs_, action_, scope): # TODO: shared?
        saEncoding_ = tf.concat([obs_, action_], axis=1)
        saEncoding_ = tf.layers.batch_normalization(saEncoding_) if self.normInput else saEncoding_
        with tf.variable_scope('saEncoding' + scope):
            for layerWidth in self.layersWidths:
                saEncoding_ = layers.fully_connected(saEncoding_, num_outputs= layerWidth, activation_fn= self.activFunc)

        return saEncoding_


class GetSencodings:
    def __init__(self, layersWidths, normInput = True, activFunc = tf.nn.leaky_relu):
        self.layersWidths = layersWidths
        self.normInput = normInput
        self.activFunc = activFunc

    def __call__(self, obs_, scope):
        sEncoding_ = obs_
        sEncoding_ = tf.layers.batch_normalization(sEncoding_) if self.normInput else sEncoding_

        with tf.variable_scope('sEncoding' + scope):
            for layerWidth in self.layersWidths:
                sEncoding_ = layers.fully_connected(sEncoding_, num_outputs=layerWidth, activation_fn= self.activFunc)

        return sEncoding_

class BuildValueExtractor:
    def __init__(self, attentionLayerWidth, activFunc = tf.nn.leaky_relu):
        self.attentionLayerWidth = attentionLayerWidth
        self.activFunc = activFunc

    def __call__(self, saEncoding_, attID):
        with tf.variable_scope('valueExtractor' + str(attID), reuse=tf.AUTO_REUSE):
            activation_ = layers.fully_connected(saEncoding_, num_outputs= self.attentionLayerWidth, activation_fn= self.activFunc)

        return activation_


class BuildSelectorExtractor:
    def __init__(self, attentionLayerWidth):
        self.attentionLayerWidth = attentionLayerWidth

    def __call__(self, sEncoding_, attID):
        with tf.variable_scope('selectorExtractor' + str(attID), reuse=tf.AUTO_REUSE):
            activation_ = layers.fully_connected(sEncoding_, num_outputs= self.attentionLayerWidth)

        return activation_


class BuildKeyExtractor:
    def __init__(self, attentionLayerWidth):
        self.attentionLayerWidth = attentionLayerWidth

    def __call__(self, saEncoding_, attID):
        with tf.variable_scope('keyExtractor' + str(attID), reuse=tf.AUTO_REUSE):
            activation_ = layers.fully_connected(saEncoding_, num_outputs= self.attentionLayerWidth)

        return activation_


class Actor:
    def __init__(self, policyLayersWidths, agentObsDim, agentActionDim, reward_scale, actorLR, scope, session, gradNormClipping = 0.5, activFunc = tf.nn.leaky_relu): # 16,5 ... 14, 5
        self.policyLayersWidths = policyLayersWidths
        self.agentActionDim = agentActionDim
        self.agentObsDim = agentObsDim
        self.activFunc = activFunc
        self.session = session
        self.actorLR = actorLR
        self.reward_scale = reward_scale
        self.gradNormClipping = gradNormClipping

        with tf.variable_scope(scope):
            with tf.variable_scope('inputs'):
                self.agentObs_ = tf.placeholder(tf.float32, shape=(None, self.agentObsDim))

            with tf.variable_scope('trainNet'):
                self.actorActivation_ = tf.layers.batch_normalization(self.agentObs_)
                for layerWidth in self.policyLayersWidths:
                    self.actorActivation_ = layers.fully_connected(self.actorActivation_, num_outputs= layerWidth, activation_fn= self.activFunc)

                self.policyLogitsUnNormalized_ = layers.fully_connected(self.actorActivation_, num_outputs= self.agentActionDim, activation_fn= None)

            with tf.variable_scope('trainNetOutputs'):
                self.policyProbs_ = tf.nn.softmax(self.policyLogitsUnNormalized_, dim = 1)
                self.policyLogits_ = tf.log(self.policyProbs_)
                actionSampleID = tf.random.categorical(logits= self.policyLogits_, num_samples=1)
                self.actionOneHotWithSample = tf.one_hot(tf.reshape(actionSampleID, [-1, ]), depth = self.agentActionDim)

                self.actionID = tf.argmax(self.policyProbs_, axis = 1)
                self.actionOneHot = tf.one_hot(self.actionID, depth = self.agentActionDim)

                self.actionLogSoftmax_ = tf.nn.log_softmax(self.policyLogitsUnNormalized_, dim = 1)
                self.logPi_ = self.gatherWithIndex(self.actionLogSoftmax_, actionSampleID)

                self.regularizeTerm_ = tf.reduce_mean(self.policyLogitsUnNormalized_**2)

                self.entropy_ = -tf.reduce_mean(tf.reduce_sum(self.actionLogSoftmax_ * self.policyProbs_, axis = 1))

            with tf.variable_scope("updateParameters"):
                self.trainParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= scope + '/trainNet')

            with tf.variable_scope('train'):
                # # self.sample_ = tf.placeholder(tf.bool, shape=None)
                # # self.actionOut_ = tf.cond(self.sample_, lambda: self.actionOneHotWithSample, lambda: self.actionOneHot)
                #
                # self.policyProbTrain_ = tf.placeholder(tf.float32, shape=(None, self.agentObsDim))
                # self.logPiTrain_ = tf.placeholder(tf.float32, shape=(None, 1))
                # self.regularizeTermTrain_ = tf.placeholder(tf.float32, shape=(None))

                self.q_ = tf.placeholder(tf.float32, shape=(None, 1))
                self.allQ_ = tf.placeholder(tf.float32, shape=(None, self.agentActionDim))

                v_ = tf.reduce_sum(self.allQ_ * self.policyProbs_, axis = 1, keepdims= True)
                policyTarget = self.q_ - v_
                policyLoss_ = tf.reduce_mean(tf.stop_gradient(self.logPi_ * (self.logPi_ / self.reward_scale - policyTarget)))
                policyLoss_ += 1e-3 * self.regularizeTerm_  #

                optimizer = tf.train.AdamOptimizer(self.actorLR, name='actorOptimizer')
                self.actorTrainOpt_ = U.minimize_and_clip(optimizer, policyLoss_, self.trainParams_, self.gradNormClipping)

    def parameters(self):
        return self.trainParams_

    def actWithExploration(self, agentObs):
        action = self.session.run(self.actionOneHotWithSample, feed_dict = {self.agentObs_: agentObs})
        return action

    def actDeterministic(self, agentObs):
        action = self.session.run(self.actionOneHot, feed_dict = {self.agentObs_: agentObs})
        return action

    # def getPolicyOutputsWithSample(self, agentObs):
    #     actionOneHotWithSample, policyProbs, logPi, regularizeTerm, entropy = self.session.run([self.actionOneHotWithSample, self.policyProbs_, self.logPi_, self.regularizeTerm_, self.entropy_], feed_dict = {self.agentObs_: agentObs})
    #     return actionOneHotWithSample, policyProbs, logPi, regularizeTerm, entropy

    def gatherWithIndex(self, value_, index_):
        # torch: result_ = value_.gather(1, index_)
        indexReshaped_ = tf.stack([tf.cast(tf.range(tf.shape(index_)[0]), tf.int64), index_[:, 0]], axis=-1)
        result_ = tf.gather_nd(value_ , indexReshaped_)
        return result_

    def getActionAndLogPiWithSample(self, agentObs):
        actionOneHotWithSample, logPi = self.session.run([self.actionOneHotWithSample, self.logPi_], feed_dict = {self.agentObs_: agentObs})
        return actionOneHotWithSample, logPi

    def train(self, agentObs, agentQ, agentAllQ):
        agentQ = np.array(agentQ).reshape(-1, 1)
        self.session.run(self.actorTrainOpt_, feed_dict = {self.agentObs_: agentObs, self.q_: agentQ, self.allQ_: agentAllQ})


class Critic:
    def __init__(self, numAgents, actionDimList, obsDimList, hiddenLayersWidth, criticLR, reward_scale, gamma, scope, session,
                 numAttentionHeads, getSAEncodings, getSEncodings, buildValueExtractor, buildSelectorExtractor, buildKeyExtractor):
        self.numAgents = numAgents
        self.actionDimList = actionDimList
        self.obsDimList = obsDimList
        self.numAttentionHeads = numAttentionHeads #4
        self.hiddenLayersWidth = hiddenLayersWidth
        self.criticLR = criticLR
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.scope = scope
        self.session = session

        self.getSAEncodings = getSAEncodings
        self.getSEncodings = getSEncodings
        self.buildValueExtractor = buildValueExtractor
        self.buildSelectorExtractor = buildSelectorExtractor
        self.buildKeyExtractor = buildKeyExtractor

        with tf.variable_scope(scope):
            self.obsList_ = [tf.placeholder(tf.float32, shape=(None, agentObsDim)) for agentObsDim in self.obsDimList]
            self.actionList_ = [tf.placeholder(tf.float32, shape=(None, agentActDim)) for agentActDim in self.actionDimList]
            scopeList = ['agent'+ str(agentID) for agentID in range(self.numAgents)]

            saEncodings = [self.getSAEncodings(obs_, action_, scopeVal) for obs_, action_, scopeVal in zip(self.obsList_, self.actionList_, scopeList)]
            sEncodings = [self.getSEncodings(obs_, scopeVal) for obs_, scopeVal in zip(self.obsList_, scopeList)]

            allHeadSelectors = [[self.buildSelectorExtractor(sEncoding_, attID) for sEncoding_ in sEncodings] for attID in range(self.numAttentionHeads)]
            allHeadKeys = [[self.buildKeyExtractor(saEncoding_, attID) for saEncoding_ in saEncodings] for attID in range(self.numAttentionHeads)]
            allHeadValues = [[self.buildValueExtractor(saEncoding_, attID) for saEncoding_ in saEncodings] for attID in range(self.numAttentionHeads)]

            agentsOtherValues = [[] for _ in range(self.numAgents)]
            agentsAttLogits = [[] for _ in range(self.numAgents)]
            agentsAttProbs = [[] for _ in range(self.numAgents)]
            for currentHeadKeys, currentHeadValues, currentHeadSelectors in zip(allHeadKeys, allHeadValues, allHeadSelectors):
                for agentID, selector in zip(range(self.numAgents), currentHeadSelectors):
                    otherAgentsKeys = [key for id, key in enumerate(currentHeadKeys) if id != agentID]
                    otherAgentsValues = [value for id, value in enumerate(currentHeadValues) if id != agentID]

                    keysReshaped = tf.transpose(tf.stack(otherAgentsKeys), perm = [1, 2, 0])
                    selectorReshaped = tf.reshape(selector, [tf.shape(selector)[0], 1, -1])
                    attendLogits_ = tf.matmul(selectorReshaped, keysReshaped)
                    scaledAttendLogits_ = attendLogits_/tf.sqrt(tf.cast(otherAgentsKeys[0].shape[1], tf.float32)) #32

                    attentionWeights_ = tf.nn.softmax(scaledAttendLogits_, dim = 2)
                    otherValues_ =  tf.reduce_sum(tf.transpose(tf.stack(otherAgentsValues), perm = [1, 2, 0]) * attentionWeights_, axis = 2)

                    agentsOtherValues[agentID].append(otherValues_)
                    agentsAttLogits[agentID].append(attendLogits_)
                    agentsAttProbs[agentID].append(attentionWeights_)

            self.agentsQ_ = []
            self.agentsRegularize_ = []
            self.agentsAllQ = []
            # calculate Q per agent
            for agentID in range(self.numAgents):
                sEncoding_ = sEncodings[agentID]
                otherValue_ = agentsOtherValues[agentID]
                agentActionDim = self.actionDimList[agentID]

                with tf.variable_scope('criticEncoderAgent'+ str(agentID)):
                    activation_ = tf.concat([sEncoding_, *otherValue_], axis=1)
                    activation_ = layers.fully_connected(activation_, num_outputs= self.hiddenLayersWidth, activation_fn= tf.nn.leaky_relu)
                    self.qValForAllActions_ = layers.fully_connected(activation_, num_outputs= agentActionDim)
                    self.agentsAllQ.append(self.qValForAllActions_)

                    actionID_ = tf.argmax(self.actionList_[agentID], axis=1)
                    actionID_ = tf.reshape(actionID_, [-1, 1])
                    qVal_ = self.gatherWithIndex(self.qValForAllActions_, actionID_)#gather function in torch
                    self.agentsQ_.append(qVal_)

                    regularizeTerm_ = 1e-3 * tf.reduce_sum([tf.reduce_mean(logit ** 2) for logit in agentsAttLogits[agentID]])
                    self.agentsRegularize_.append(regularizeTerm_)

            with tf.variable_scope('parameters'):
                self.criticParams_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

            with tf.variable_scope('train'):
                optimizer = tf.train.AdamOptimizer(self.criticLR, name='criticOptimizer')
                self.qLoss_ = 0

                self.agentsNextLogPi_ = [tf.placeholder(tf.float32, shape=(None, )) for _ in range(self.numAgents)]
                self.rewards_ = [tf.placeholder(tf.float32, shape=(None, )) for _ in range(self.numAgents)]
                self.dones_ = [tf.placeholder(tf.float32, shape=(None, )) for _ in range(self.numAgents)]

                self.agentsTargetNextQ_ = [tf.placeholder(tf.float32, shape=(None, )) for _ in range(self.numAgents)]
                for agentID, nextQ_, log_pi, currentQ_, reg_ in zip(range(self.numAgents), self.agentsTargetNextQ_, self.agentsNextLogPi_, self.agentsQ_, self.agentsRegularize_):
                    agentReward_ = self.rewards_[agentID]
                    agentDone_ = self.dones_[agentID]

                    target_q = (agentReward_ + self.gamma * nextQ_ * (1 - agentDone_))
                    target_q -= log_pi / self.reward_scale
                    self.qLoss_ = self.qLoss_ + tf.losses.mean_squared_error(target_q, currentQ_)
                    self.qLoss_ += reg_  # regularizing attention

                self.trainOpt_ = optimizer.minimize(self.qLoss_, var_list= self.criticParams_)
                self.trainOpt_ = U.minimize_and_clip(optimizer, self.qLoss_, self.criticParams_, 10 * self.numAgents)

    def parameters(self):
        return self.criticParams_

    def scaleSharedGrads(self):
        selectorParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'selectorExtractor')
        keyParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'keyExtractor')
        valueParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'valueExtractor')
        saParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'saEncoding')
        sharedParams = [selectorParams, keyParams, valueParams, saParams]
        scaleParams = [[tf.assign(param, param* 1./self.numAgents) for param in paramList] for paramList in sharedParams]
        self.session.run(scaleParams)

    def getAllAgentsQ(self, actionList, obsList):
        stateDict = {agentObs_: agentObs for agentObs_, agentObs in zip(self.obsList_, obsList)}
        actionDict = {agentAction_: agentAction for agentAction_, agentAction in zip(self.actionList_, actionList)}
        allAgentsQ = self.session.run(self.agentsQ_, feed_dict = {**stateDict, **actionDict})
        return allAgentsQ

    def getAllAgentsSingleQAndAllQ(self, actionList, obsList):
        stateDict = {agentObs_: agentObs for agentObs_, agentObs in zip(self.obsList_, obsList)}
        actionDict = {agentAction_: agentAction for agentAction_, agentAction in zip(self.actionList_, actionList)}
        allAgentsQ, allAgentsAllQ = self.session.run([self.agentsQ_, self.agentsAllQ], feed_dict = {**stateDict, **actionDict})
        return allAgentsQ, allAgentsAllQ

    def gatherWithIndex(self, value_, index_):
        # torch: result_ = value_.gather(1, index_)
        indexReshaped_ = tf.stack([tf.cast(tf.range(tf.shape(index_)[0]), tf.int64), index_[:, 0]], axis=-1)
        result_ = tf.gather_nd(value_ , indexReshaped_)
        return result_

    def train(self, actionList, obsList, agentsNextLogPi, rewardsList, donesList, nextTargetQList):
        stateDict = {agentObs_: agentObs for agentObs_, agentObs in zip(self.obsList_, obsList)}
        actionDict = {agentAction_: agentAction for agentAction_, agentAction in zip(self.actionList_, actionList)}
        rewardDict = {agentRew_: agentRew for agentRew_, agentRew in zip(self.rewards_, rewardsList)}
        doneDict = {agentDone_: agentDone for agentDone_, agentDone in zip(self.dones_, donesList)}
        nextLogPiDict = {agentLogpi_: agentLogpi for agentLogpi_, agentLogpi in zip(self.agentsNextLogPi_, agentsNextLogPi)}
        nextTargetQDict = {nextQ_: nextQ for nextQ_, nextQ in zip(self.agentsTargetNextQ_, nextTargetQList)}

        self.session.run(self.trainOpt_, feed_dict = {**stateDict, **actionDict, **rewardDict, **doneDict, **nextLogPiDict, **nextTargetQDict})

class AttentionSAC:
    def __init__(self, actorList, actorTargetList, critic, criticTarget, updateParams):
        self.actorList = actorList
        self.actorTargetList = actorTargetList
        self.critic = critic
        self.criticTarget = criticTarget
        self.numAgents = len(self.actorList)
        self.updateParams = updateParams

    def unpackBuffer(self, sample):
        obs, acs, rews, next_obs, dones = sample

        observationList = list(zip(*obs))
        actionList = list(zip(*acs))
        rewardList = list(zip(*rews))
        nextObsList = list(zip(*next_obs))
        doneList = list(zip(*dones))

        return observationList, actionList, rewardList, nextObsList, doneList

    def trainCritic(self, sample):
        observations, actions, rewards, nextObservations, dones = self.unpackBuffer(sample)
        actorTargetOutputs = [actorTarget.getActionAndLogPiWithSample(agentNextObs) for actorTarget, agentNextObs in zip(self.actorTargetList, nextObservations)]

        nextActions, agentsNextLogPi = list(zip(*actorTargetOutputs))
        nextTargetQList = self.criticTarget.getAllAgentsQ(nextActions, nextObservations)
        self.critic.train(actions, observations, agentsNextLogPi, rewards, dones, nextTargetQList)

    def trainActor(self, sample):
        observations, actions, rewards, nextObservations, dones = self.unpackBuffer(sample)
        # actorOutputs = [actor.getPolicyOutputsWithSample(agentObs) for actor, agentObs in zip(self.actorList, observations)]
        # actionOneHotWithSampleList, policyProbsList, logPiList, regularizeTermList, entropyList = list(zip(*actorOutputs))

        actionOneHotWithSampleList = [actor.actWithExploration(agentObs) for actor, agentObs in zip(self.actorList, observations)]
        allAgentsQ, allAgentsAllQ = self.critic.getAllAgentsSingleQAndAllQ(actionOneHotWithSampleList, observations)

        for actor, agentObs, agentQ, agentAllQ in zip(self.actorList, observations, allAgentsQ, allAgentsAllQ):
            actor.train(agentObs, agentQ, agentAllQ)

    def step(self, observations, explore = False):
        observations = [np.array(agentObs).reshape(1, -1) for agentObs in observations]
        if explore:
            actions = [actor.actWithExploration(agentObs) for actor, agentObs in zip(self.actorList, observations)]
        else:
            actions = [actor.actDeterministic(agentObs) for actor, agentObs in zip(self.actorList, observations)]
        actions = [agentAction[0] for agentAction in actions]

        return actions

    def updateTargetParams(self):
        startTime = time.time()
        self.updateParams(self.critic.parameters(), self.criticTarget.parameters())
        endTime = time.time()
        # print('critic time', endTime - startTime)

        startTime = time.time()
        [self.updateParams(actor.parameters(), actorTarget.parameters()) for actor, actorTarget in zip(self.actorList, self.actorTargetList)]
        endTime = time.time()
        # print('actor time', endTime - startTime)

    def trainOneStep(self, sample):
        self.trainCritic(sample)
        self.trainActor(sample)
        self.updateTargetParams()


class UpdateParams:
    def __init__(self, tau, session):
        self.tau = tau
        self.session = session

    def __call__(self, parameters, targetParameters, updateList = False):
        updateParam_ = [targetParameters[i].assign((1 - self.tau) * targetParameters[i] + self.tau * parameters[i]) for i in range(len(parameters))]
        self.session.run(updateParam_)


class SampleOneStep:
    def __init__(self, transit, rewardFunc, isTerminal):
        self.transit = transit
        self.rewardFunc = rewardFunc
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        nextState = self.transit(state, action)
        rewards = self.rewardFunc(state, action, nextState)
        dones = self.isTerminal(nextState)
        return nextState, rewards, dones


class TrainOneStep:
    def __init__(self, model, buffer, batchSize, numUpdatesPerTrain, updateInterval):
        self.model = model
        self.buffer = buffer
        self.batchSize = batchSize
        self.numUpdatesPerTrain = numUpdatesPerTrain
        self.updateInterval = updateInterval
        self.runTime = 0

    def __call__(self):
        self.runTime += 1
        if self.buffer.length() < self.batchSize or self.runTime % self.updateInterval != 0:
            return

        for updateID in range(self.numUpdatesPerTrain):
            sample = self.buffer.sample(self.batchSize)
            self.model.trainOneStep(sample)


class MAAC:
    def __init__(self, attentionSAC, buffer, maxEpisode, maxTimestep, reset, sampleOneStep, observe, saveModel, trainOneStep):
        self.model = attentionSAC
        self.buffer = buffer
        self.maxEpisode = maxEpisode
        self.maxTimestep = maxTimestep
        self.reset = reset
        self.saveModel = saveModel
        self.sampleOneStep = sampleOneStep
        self.observe = observe
        self.trainOneStep = trainOneStep

    def __call__(self):
        for epsID in range(self.maxEpisode):
            print('episode', epsID)
            state = self.reset()

            for timestep in range(self.maxTimestep):
                obs = self.observe(state)
                action = self.model.step(obs)
                nextState, rewards, dones = self.sampleOneStep(state, action)
                nextObs = self.observe(nextState)
                self.buffer.add(obs, action, rewards, nextObs, dones)
                state = nextState
                self.trainOneStep()

            self.saveModel()














