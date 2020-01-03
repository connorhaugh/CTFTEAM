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

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class RAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.distancer.getDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


class QAgent(RAgent):
    """
    A reflex agent that seeks food. The agent acts based on its extensive training (using q learning), and makes perceptions based on particle filtering. Note that this agent is NOT programmed to play either offense of defense.
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        ## Percept Learning stuff
        ## Q learning stuff- rates taken from project 3
        self.counter=0
        self.explorationRate = 0.1#Epsilon
        self.learningRate = 0.1# Alpha
        self.discountRate =0.8
        self.weights= {'bias':1.0,'score':1.0,'onDefense':1.0,'closestEnemy':1.0, 'distanceToFood':1.0,'distanceToCapsule':1.0}

        # read from a file the wieghts
        try:
            with open('qlearndata.txt', "r") as file:
                self.weights =eval(file.read())
        except IOError:
            return


## Q Learning stuff!!!
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

        if(reward!=0):
            print('I did a good thing MOm' + str(reward))
        #print(self.counter)
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
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
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


        return random.choice(highaction)


    def flipCoin( p ):
        r = random.random()
        return r < p

    def chooseAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
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
          #Q(s,a)= Q(s,a) + alpha(R(S) + gamma(max(Q(s',a') - Q(s,a))))

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

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        dists=[]
        for a in enemies:
            if(a.getPosition() != None):
                dists.append(self.distancer.getDistance(myPos, a.getPosition()))
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

class DefensiveAgent(RAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
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

class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=200):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState,activeAgent):
        "Stores information about the game, then initializes particles."
        self.numGhosts = 2
        self.opponentIndexs= activeAgent.getOpponents(gameState)
        self.ghostAgents = []
        self.agent=activeAgent

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
        numGhosts=self.numGhosts

        # Make permutations the cartesian product of numGhost number of legalpositions lists
        perms= [ ]
        perms = itertools.product(legalPositions,repeat=numGhosts)
        perms=list(perms)
        # shuffle, because for some reason we have to
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
        noisyDistances=[]
        for i in self.opponentIndexs:
            noisyDistances.append(noisyreadings[i])
        numGhosts=self.numGhosts
        particleList=self.particles
        N=self.numParticles

        #Step 2: init wieghts based on emissionModel (the probaility of a particluar ghost being at a distance away given the noisy reading)
        beliefs=util.Counter()
        for  i in range(0,N):
            #the wieght of every particle should be the multaplicative sum of the indovidual location's probabilities based on the inquiry
            weightAtState = 1
            for ghostIndex in range(0,numGhosts):
                weightAtState*= gameState.getDistanceProb(self.agent.getMazeDistance(particleList[i][ghostIndex],pacmanPosition),noisyDistances[ghostIndex])
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
        "*** YOUR CODE HERE ***"
        particleList= self.particles

        belief=util.Counter()

        for particle in particleList: # iterate through all the locations stored at each particle
            belief[particle]+=1 #increment at that particle

        belief.normalize()
        return belief
    def bestChoice(self):
        beliefs = self.getBeliefDistribution()
        return beliefs.argMax()

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # everything below is new stuff
    observation = self.getCurrentObservation()
    posOfOtherTeam = [observation.getAgentPosition(i) for i in range(observation.getNumAgents()) if i not in gameState.getRedTeamIndices()]
    # print("My observations on the other agents is: ", posOfOtherTeam)
    myFoodList = self.getFoodYouAreDefending(gameState).asList()
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    friendlyPos = [observation.getAgentPosition(i) for i in range(observation.getNumAgents()) if i not in gameState.getBlueTeamIndices() and i != self.index]


    #=========================================================================================

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # if there is a close enemy ghost, and we are pacman, then we need to make a bee line for the other side
    enemyDists = [self.getMazeDistance(myPos, enemyPos) for enemyPos in posOfOtherTeam if enemyPos != None]
    if len(enemyDists) > 0 and myState.isPacman:
      minDistFromEnemy = min(enemyDists)
      # get the min dist, and if it is close (within 5), then we dip out of there
      if minDistFromEnemy <= 5:
        # we need to make a beeline for the nearest friendly food
        minDistanceToSide = self.getMazeDistance(myPos, friendlyPos[0])
        features['retreat'] = minDistanceToSide



    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'retreat': -100}
