import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Walker(object):
    def __init__(self, maxSteps):
        self.maxSteps = maxSteps
        self.walkLength = maxSteps * 2 +1
        self.midStep = int(self.walkLength/2)
        
        
        self.walkMap = np.zeros((self.walkLength, 2), dtype="complex128")
        self.walkMap[self.midStep,0] = 1/np.sqrt(2)
        self.walkMap[self.midStep,1] = 1/np.sqrt(2)
        
        self.probabilites = np.zeros(self.walkLength)
        self.boundProbabilites = []
        self.positions = np.zeros(self.walkLength)
        self.boundPositions = []
        self.EE = []
        self.IPR = 0
        self.MOI = 0
        

    def resetMap(self):
        self.walkMap = np.zeros((self.walkLength, 2), dtype="complex128")
        self.walkMap[self.midStep,0] = 1/np.sqrt(2)
        self.walkMap[self.midStep,1] = 1/np.sqrt(2)
    
    def setSteps(self, maxSteps):
        self.maxSteps = maxSteps
        
    def setStateLoc(self, state_location):
        self.walkMap[state_location, 0] = 1/2
        self.walkMap[state_location, 1] = 1/2


    ## Quantum Walk Operator Functions ##


    def genPosition(self):
        p = []
        for x in range(0, self.maxSteps):
            if(self.walkMap[x,1] != 0 and self.walkMap[x,0] != 0):
                p.append(self.walkMap[x,:])
            else:
                p.append(0)
        return p

    ## Generate Coin ##
    def genCoin(self, theta, phi1, phi2):
        coin = np.array([[np.cos(theta), (np.exp(1j*phi1))*np.sin(theta)],
                         [np.exp(1j*phi2)*np.sin(theta), -1* np.exp(1j*(phi1 + phi2))*np.cos(theta)]])

        return coin


    ## Positive Translate Operator ##
    def posShift(self):
        lastNew = 0
        for x in range(0,self.walkLength-1):
            if(self.walkMap[x,0] != 0 and x != lastNew):
                self.walkMap[x+1,0] = self.walkMap[x,0]
                self.walkMap[x,0] = 0
                lastNew = x+1
                
    def invPosShift(self):
        lastNew = 0
        for x in range(0,self.walkLength-1):
            if(self.walkMap[x,0] != 0):
                if(x != lastNew and x!= 0):
                    self.walkMap[x-1,0] = self.walkMap[x,0]
                    self.walkMap[x,0] = 0
                    lastNew = x-1
        

    ## Negative Translate Operator ##
    def negShift(self):
        lastNew = 0
        for x in range(0,self.walkLength-1):
            if(self.walkMap[x,1] != 0):
                if(x != lastNew and x!= 0):
                    self.walkMap[x-1,1] = self.walkMap[x,1]
                    self.walkMap[x,1] = 0
                    lastNew = x-1
    
    def invNegShift(self):
        lastNew = 0
        for x in range(0,self.walkLength-1):
            if(self.walkMap[x,1] != 0 and x != lastNew):
                self.walkMap[x+1,1] = self.walkMap[x,1]
                self.walkMap[x,1] = 0
                lastNew = x+1

    ## Total Translate Operators ##
    def translateOp(self):
        self.posShift()
        self.negShift()
        
    def transInv(self):
        self.invPosShift()
        self.invNegShift()
        
    def noEETransOp(self):
        oldMap = self.walkMap
        newMap = np.zeros((self.walkLength, 2), dtype="complex128")

        for x in range(self.midStep, self.walkLength):
            if (x != self.walkLength - 1):
                newMap[x,0] = oldMap[x+1, 0] * 0.5 + oldMap[x-1, 0] * 0.5
                newMap[x,1] = oldMap[x+1, 1] * 0.5 + oldMap[x-1, 1] * 0.5

        for x in range(0, self.midStep):
            if (x != self.midStep + 1):
                newMap[x,0] = oldMap[x+1, 0] * 0.5 + oldMap[x-1, 0] * 0.5
                newMap[x,1] = oldMap[x+1, 1] * 0.5 + oldMap[x-1, 1] * 0.5

        # print(oldMap)
        # print('\n')
        # print(newMap)
        # print('\n')
        self.walkMap = newMap

    ## Probability Calc Functions ##
    def calcProb(self):
        pos = np.zeros(self.walkLength)
        probs = np.zeros(self.walkLength)

        for x in range(0,self.walkLength):
            pos[x] = (x-self.maxSteps)
            probs[x] = (np.abs(self.walkMap[x,0])**2 + np.abs(self.walkMap[x,1])**2)

        self.positions = pos
        self.probabilites = probs
        
        ## Check x position ##
    def calcProbBounded(self, lBound, uBound):
        pos = []
        probs = []

        for x in range(lBound,uBound):
            pos.append(x)
            probs.append(np.abs(self.walkMap[x,0])**2 + np.abs(self.walkMap[x,1])**2)

        self.boundPositions = pos
        self.boundProbabilites = probs
        

        
    def calcIPR(self):
        self.calcProb()
        
        IPRS = []

        for x in range(0,self.walkLength):
            IPRS.append(self.probabilites[x]**2)
        

        self.IPR = sum(IPRS)
        
    def calcMOI(self):
        self.calcProb()
        
        MoI = 0
        
        for x in range(0, self.walkLength):
            MoI += self.probabilites[x] * x**2
        
        self.MOI = MoI 
    
    ## Quantum Walk ##
    def runWalk(self, thetas, phi1, phi2):

        if(isinstance(thetas, list)):

            n = 0
            for x in range(0, self.maxSteps):

                if(len(thetas) > 1 and n % 3 == 0):
                    coin = self.genCoin(thetas[0], phi1, phi2)
                else:
                    coin = self.genCoin(thetas[1], phi1, phi2)

                self.walkMap = np.dot(self.walkMap, coin)
                self.translateOp()
                n += 1

        else:
            for x in range(0, self.maxSteps):

                coin = self.genCoin(thetas, phi1, phi2)

                self.walkMap = np.dot(self.walkMap, coin)
                self.translateOp()

            
        self.calcProb()

    ## Classical + Quantum Walk, random coin##
    def runCQWalkCoinOp(self, thetas, phi1, phi2, rng, pR):
        
        for x in range(0, self.maxSteps):

            cCoin = rng.choice([1,-1], p=[pR, 1-pR])

            if(cCoin == -1):
                coin = self.genCoin(thetas[0], phi1, phi2)
            elif(cCoin == 1):
                coin = self.genCoin(thetas[1], phi1, phi2)

            self.walkMap = np.dot(self.walkMap, coin)
            self.translateOp()

        self.calcProb()

    ## Classical + Quantum Walk, random translate##
    def runCQWalkTransOp(self, theta, phi1, phi2, rng, pR):
        
        coin = self.genCoin(theta, phi1, phi2)
        for x in range(0, self.maxSteps):
            self.walkMap = np.dot(self.walkMap, coin)

            transChoice = rng.choice([1,-1], p=[pR, 1-pR])

            if(transChoice == 1):
                self.translateOp()
            elif(transChoice == -1):
                self.transInv()

        self.calcProb()


    def runDRCoinWalk(steps, theta_base, phi1, phi2,delta_theta, pR, seed=42):
        """
        At each step, randomly applies either:
            C1: theta_base + delta_theta  (with probability pR)
            C2: theta_base - delta_theta  (with probability 1 - pR)
            * pR will usually be 0.5 in the context of flipping a coin to determine what gets used, 
              but can be adjusted to see the effect of biasing towards one coin or the other.
        """
        rng = np.random.default_rng(seed)
        w = Walker(steps)

        for _ in range(steps):
            # Randomly pick C1 or C2
            jitter = rng.choice([1, -1], p=[pR, 1-pR])
            theta = theta_base + jitter * delta_theta

            coin = w.genCoin(theta, phi1, phi2)
            w.walkMap = np.dot(w.walkMap, coin)
            w.translateOp()

        w.calcProb()
        return w
    
    def runCRCoinWalk(steps, theta, phi1, phi2, jitter):
        """
        At each step, randomly selects from a uniform distribution between [-w, w] and applies it
        to the base coin angle theta. This simulates a continuous range of coin jitters rather than
        just two discrete coins.
        """
        w = Walker(steps)
        rng = np.random.default_rng(seed=42)


        for _ in range(steps):
            # Allows for a continuous range of coin angles by adding a random jitter in the range [-w, w] to the base theta
            updated_theta = theta + np.random.uniform(-jitter, jitter)
            coin = w.genCoin(updated_theta, phi1, phi2)
            w.walkMap = np.dot(w.walkMap, coin)
            w.translateOp()

        w.calcProb()
        return w

    def runRTransWalk(steps, theta, phi1, phi2, P_a, seed=42):
        """
        At each step, given P_a rolls a P_n instead. If P_n > P_a, apply the normal translate operator.
        If P_n < P_a, apply the inverse translate operator instead. Note that the coin operator is 
        constant and for an unbiased translation operator, P_a = 0 OR 1 (allowing P_n to exceed/never
        exceed it every time)
        """
        rng = np.random.default_rng(seed)
        w = Walker(steps)

        coin = w.genCoin(theta, phi1, phi2)

        for _ in range(steps):
            w.walkMap = np.dot(w.walkMap, coin)

            P_n = rng.uniform(0, 1)

            if(P_n > P_a):
                w.translateOp()
            else:
                w.transInv()

        
        w.calcProb()
        return w
    
    ## Ploting Functions ##
    ## Probability Distribution ##
    def probDistribution(self, title, it=0, m=""):
        fig, ax = plt.subplots()
        if(m!= ""):
            fig.suptitle(title + " Quantum walk for " + str(self.maxSteps) + " Steps" + "\n" + m)
        elif(title != ""):
            fig.suptitle(title + " Quantum walk for " + str(self.maxSteps) + " Steps")
            
        ax.set_xlabel("x")
        ax.set_ylabel("P(x)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if(it != 0):
            plt.plot(self.positions, self.probabilites, label="$\Delta P_{R} = $ %f" %it)
            plt.legend(loc="upper right")
        else:
            plt.plot(self.positions, self.probabilites)

    ## Histogram of Probabilites ##
    def __plotHist(self):
        fig, ax = plt.subplots()
        fig.suptitle(" Quantum walk for " + str(self.maxSteps) + " Steps")
        ax.set_xlabel("P(x)")
        ax.set_ylabel("# of values")

        plt.hist(self.probabilites, bins = 20)
        
    

    ## Main Plotting Call ##
    def plotWalk(self, choice, title="Single"):

        if(choice == 'p'):
            self.probDistribution(title)
            plt.savefig('Figures/' + title +' Coin ProbDist ' + str(self.maxSteps) + ' steps.jpg', dpi=200)
            #plt.show()

        elif(choice == 'h'):
            self.__plotHist()
            plt.savefig('Figures/' + 'Histogram ' + str(self.maxSteps) + ' steps.jpg', dpi=200)
            #plt.show()
            
    def plotReal(self):
        fig, ax = plt.subplots(2,1)
        
        fig.suptitle("Real Components of Quantum Random Walk, %i Steps" %self.maxSteps)
        
        ax[0].plot(self.positions, np.real(self.walkMap[:,0]), label=r'$|x,+\rangle$')
        ax[1].plot(self.positions, np.real(self.walkMap[:,1]), color="red",label=r'$|x,-\rangle$')
        
        ax[0].set_xlabel("Walk Position")
        ax[1].set_xlabel("Walk Position")
        
        ax[0].set_ylabel("State Value")
        ax[1].set_ylabel("State Value")
        
        #ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        fig.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        
        plt.savefig('Figures/' + 'Real Comp' + ' ' + str(self.maxSteps) + ' Steps.pdf', bbox_inches="tight")
        plt.show()
        
    def plotImag(self):
        fig, ax = plt.subplots(2,1)
        
        fig.suptitle("Imaginary Components of Quantum Random Walk, %i Steps" %self.maxSteps)
        
        ax[0].plot(self.positions, np.imag(self.walkMap[:,0]), label=r'$|x,+\rangle$')
        ax[1].plot(self.positions, np.imag(self.walkMap[:,1]), color="red",label=r'$|x,-\rangle$')
        
        ax[0].set_xlabel("Walk Position")
        ax[1].set_xlabel("Walk Position")
        
        ax[0].set_ylabel("State Value")
        ax[1].set_ylabel("State Value")
        
        #ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        #ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
                   
        fig.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        
        plt.savefig('Figures/' + 'Imag Comp' + ' ' + str(self.maxSteps) + ' Steps.pdf', bbox_inches="tight")
        plt.show()