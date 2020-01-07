from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import itertools
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


# Our Ofessive Agent
class OnePac(RAgent):
    """
    A reflex agent that seeks food. The agent acts based on its extensive training (using q learning), and makes perceptions based on particle filtering. Note that this agent is NOT programmed to play either offense of defense, but has learned to play offense.
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        ## Percept Learning- PF 1
        self.percepterOne=DisJointParticleFilter()
        self.percepterOne.initialize(gameState,self,0)
        ## PF -2
        self.percepterTwo=DisJointParticleFilter()
        self.percepterTwo.initialize(gameState,self,1)

        ## Q learning stuff- rates taken from project 3
        self.counter=0
        self.explorationRate = 0.1#Epsilon
        self.learningRate = 0.1# Alpha
        self.discountRate =0.8

        ##Note: We have not included the wieghts because that's top secret information! If your curious as to what values they have, all we will say is between 0 & 1
        self.weights= {'bias':1.0,'score':1.0,'onDefense':1.0,'closestEnemy':1.0, 'distanceToFood':1.0,'distanceToCapsule':1.0}

        # read from a file the wieghts
        try:
            with open('qlearndata.txt', "r") as file:
                self.weights =eval(file.read())
        except IOError:
            return


    def getQValue(self, gameState, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        """
        features= self.getFeatures(gameState,action)
        return features * self.weights


    def getReward(self, gameState,nextState):
        """
        returns the reward of the given state. States get rewards IF:
        1. They eat a food
        2. They kill an pacman
        3. they eat a capusle
        4. they kill a ghost
        """
        # init reward: have to compare to previous state
        reward=0
        nextPos= nextState.getAgentState(self.index).getPosition()

        #1 Eats food- AKA food existed before and doesn't now + 20 points
        oldFood= self.getFood(gameState).asList()
        newFood= self.getFood(nextState).asList()
        if nextPos in oldFood and nextPos not in newFood:
            reward+=10


        enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
        invaders = [a for a in enemies if a.getPosition() != None]
        if nextPos == [enemypos for enemypos in invaders]:
            #2  & #4 Kills an enemy (a scared ghost) 100 points
            if bool(enemy.gameState.data.agentStates[enemy.index].scaredTimer):
                reward+=100

        #3 eats a capsule +50 points
        oldCapsules= self.getCapsules(gameState)
        newCapsules= self.getCapsules(nextState)
        if nextPos in oldCapsules and nextPos not in newCapsules:
            reward+=40

        return reward

    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        qValuesList=[]
        legalActions=gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        if(len(legalActions)==0):
            return 0.0
        for action in legalActions:
            qValuesList.append(self.getQValue(gameState,action))
        return max(qValuesList)


    def computeActionFromQValues(self, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, return None.
        """
        legalActions = gameState.getLegalActions(self.index)
        #print(legalActions)
        legalActions.remove(Directions.STOP)
        if(len(legalActions)==0):
            return None
        #Compute HighestAction, tiebreak randomly, remove STOP
        #print(legalActions)
        hiQ=float("-inf")
        highaction=[]
        for action in legalActions:
            presentq=self.getQValue(gameState,action)

            #update the wieghts at this step-- note that it updates at a lot of states it never even uses
            self.update(gameState,action)
            #print('Q is: '+ str(presentq))
            if(presentq>hiQ):
                highaction=[action]
                hiQ=presentq
            if(presentq==hiQ):
                highaction.append(action)

        #If actions have the same, highest action, chose a random action to tiebreak so that lexiconaly prioritized actions (North before West) are not always chosen.
        return random.choice(highaction)


    def flipCoin( p ):
        r = random.random()
        return r < p

    def chooseAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, you should choose None as the action.

        """
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = None
        "*** YOUR CODE HERE ***"
        if(len(legalActions)<1):
            return None
        #Flip a coin with probability epsilon, do some random action if it comes up true
        if(util.flipCoin(self.explorationRate)):
            #print('random')
            return random.choice(legalActions)
        #else return the action with the highest q value
        choiceaction =self.computeActionFromQValues(gameState)
        return choiceaction

    def update(self, gameState, action):
        """
          Update the weights based on the features, using the following formula:
          #Q(s,a)= Q(s,a) + alpha(R(S) + gamma(max(Q(s',a') - Q(s,a)))). All this function does is computes this function and applies it
          to all the features.

        """
        "*** YOUR CODE HERE ***"
        #Update function:
        #Q(s,a)= Q(s,a) + alpha(R(S) + gamma(max(Q(s',a') - Q(s,a))))

        # Locals: reward function, next state

        self.counter+=1

        nextState= self.getSuccessor(gameState,action)
        reward= self.getReward(gameState,nextState)
        features = self.getFeatures(gameState,action)


        #print(features)
        for feature in features:
            oldQ = self.getQValue(gameState,action)
            bestQAtNextState= self.computeValueFromQValues(nextState)
            a=self.learningRate
            gamma=self.discountRate
            changeQ= a * ((reward + (gamma*bestQAtNextState)) - oldQ)
            #print('Change,Weight @ feature' + str(changeQ) +"*"+ str(features[feature]) + "+" + str(self.weights[feature]))
            self.weights[feature] = self.weights[feature] + changeQ*features[feature]
            #print('final Weight:' + str(self.weights[feature]))




    def getFeatures(self, gameState, action):
        features=util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()

        # Locals
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        #==============
        #Basic Features: Position of opponents, what side of the field we are on, Score of the game
        features['score']= self.getScore(gameState)
        features['bias']=1.0
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can  and can't see using percept learning
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        dists=[]
        for a in enemies:
            if(a.getPosition() != None):
                dists.append(self.distancer.getDistance(myPos, a.getPosition()))

            else: # if the enemy is hidden, use the best guess from our PF
                if(a == enemies[1] ): #enemy 1, note that this works when its flipped for some reason to do with the creation of the agents
                    self.percepterOne.observeState(gameState)
                    dists.append(sef.distancer.getDistance(myPos, self.percepterOne.bestChoice()))
                else:
                    self.percepterTwo.observeState(gameState)
                    dists.append(sef.distancer.getDistance(myPos, self.percepterTwo.bestChoice()))

        if(len(dists)==0):
            features['closestEnemy']=1
        else:
            features['closestEnemy'] = min(dists)



        #=============
        # Offesnive Features: Food,Capsules

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
          myPos = successor.getAgentState(self.index).getPosition()
          minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
          features['distanceToFood'] = minDistance

        if len(self.getCapsules(gameState)) > 0: # This should always be True,  but better safe than sorry
          myPos = successor.getAgentState(self.index).getPosition()
          minDistanceC = min([self.distancer.getDistance(myPos, capsule) for capsule in self.getCapsules(gameState)])
          features['distanceToCapsule'] = minDistanceC

        #print(features)
        features.divideAll(10.0)
        return features

    def getWeights(self, gameState, action):
        return self.weights

    def final(self,gameState):
        print('finals szn')
        self.observationHistory = []
        finalwieghts= self.weights
        file=open('qlearndata.txt','w')
        file.write(str(self.weights))

class TwoPac(RAgent):
  """
  A reflex agent that keeps its side Pacman-free.
  """
  def registerInitialState(self, gameState):
    RAgent.registerInitialState(self, gameState)
    self.percepter=JointParticleFilter()
    self.percepter.initialize(gameState,self)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    #self.getScore(successor)
    # Locals
    myFoodList = self.getFoodYouAreDefending(gameState).asList()
    features['successorScore'] = len(myFoodList)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    observation = self.getCurrentObservation()
    posOfOtherTeam = [observation.getAgentPosition(i) for i in range(observation.getNumAgents()) if i not in gameState.getBlueTeamIndices()]

    #==============
    #Basic Features: Position of opponents, what side of the field we are on, Score of the game
    features['score']= self.getScore(gameState)

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can and can't see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    dists=[]
    for a in enemies:
        if(a.isPacman and a.getPosition() != None):
            dists.append(self.distancer.getDistance(myPos, a.getPosition()))
    self.percepter.observeState(gameState)
    expectedPosPair=self.percepter.bestChoice()
    for i in expectedPosPair:
        # prioritize pacmen on our own side of the court.
        self.debugDraw([i],[1,0,0],clear=True)
        dists.append(self.distancer.getDistance(myPos,i))
    if(len(dists)==0):
        dists.append(1)
    features['invaderDistance'] = min(dists)
    features['enemyOneDistance']=dists[0]
    features['enemyTwoDistance']=dists[1]

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -100, 'onDefense': 300, 'invaderDistance': -100, 'stop': -100, 'reverse': -2}

class DisJointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=100):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState,activeAgent,opponentIndex):
        "Stores information about the game, then initializes particles."
        self.opponentIndexs= activeAgent.getOpponents(gameState)
        self.ghostAgents = []
        self.agent=activeAgent
        self.opponentIndex=opponentIndex

        legalPositions=[]
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                if not gameState.hasWall(x,y):
                    legalPositions.append((x,y))


        self.legalPositions = legalPositions
        self.initializeParticles()


    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.numParticles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space.

        """
        "*** YOUR CODE HERE ***"
        # locals
        numParticles=self.numParticles
        legalPositions=self.legalPositions

        # make particleList a shuffled list of  evenlegalPositions
        perms= legalPositions
        # shuffle, because we have to
        random.shuffle(perms)

        #fill the particlelist evenly with permutations
        particleList=[None]*numParticles
        i=0
        while i<numParticles-1:
            for item in perms:
                particleList[i]= item
                i+=1
                if i>numParticles-1:
                    break
        #return the list
        self.particles=particleList
        return particleList

    def observeState(self, gameState):
        """
        Resamples the set of particles using the likelihood of the noisy
        observations.
        """
        # locals
        pacmanPosition= gameState.getAgentPosition(self.agent.index)
        noisyreadings = gameState.getAgentDistances()
        noisyDistances=noisyreadings[self.opponentIndex]

        particleList=self.particles
        N=self.numParticles
        opponentIndex=self.opponentIndex

        #Step 2: init wieghts based on emissionModel (the probaility of a particluar ghost being at a distance away given the noisy reading)
        beliefs=util.Counter()
        for  i in range(0,N):
            #the wieght of every particle should be the multaplicative sum of the indovidual location's probabilities based on the inquiry
            weightAtState = 1
            weightAtState*= gameState.getDistanceProb(self.agent.getMazeDistance(particleList[i],pacmanPosition),noisyDistances)
            #update beliefs
            beliefs[particleList[i]]+= weightAtState

        if beliefs.totalCount()==0:
            particleList=self.initializeParticles()
            return particleList

        #Step 3-take sample based on beliefs
        particleList= util.nSample(beliefs.values(),beliefs.keys(),len(particleList))

        #Note: Had to rename to 'particles' because of Q7
        self.particles=particleList
        return particleList

    def getBeliefDistribution(self):

        particleList= self.particles

        belief=util.Counter()

        for particle in particleList: # iterate through all the locations stored at each particle
            belief[particle]+=1 #increment at that particle

        belief.normalize()
        return belief
    def bestChoice(self):
        beliefs = self.getBeliefDistribution()
        return beliefs.argMax()
