# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

import matplotlib.pyplot as plt

from util.activation_functions import Activation
from util.loss_functions import DifferentError
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
            
        """
        
        error = DifferentError()
        iter = 0
        maxError = []
       
        learning = False
        
    
        while not learning:
            
            trainingTupels = zip(self.trainingSet.input, self.trainingSet.label)
            gradient = np.zeros(self.trainingSet.input[0].shape)
            totalErrorrate = 0
            for inputData, label in trainingTupels:
                
                output= self.fire(inputData)
                err= error.calculateError(label, output)
                gradient= gradient + err * inputData
                
                if output >= 0.5 and label == 0:
                    totalErrorrate = totalErrorrate + 1
                if output < 0.5 and label == 1:
                    totalErrorrate = totalErrorrate + 1
            
            ErrorArr= [totalErrorrate]
            maxError.extend(ErrorArr)
            self.updateWeights(gradient)
       

            iter += 1

            if verbose is True:
                logging.info("Epoch: %i; Error: %i", iter, totalErrorrate)

            if iter >= self.epochs:
                rangeIteration = range(iter)
                plt.plot(rangeIteration, maxError)
                
                plt.xlabel('Iteration')
                plt.ylabel('Error')
                
                plt.title("Logisitic Regression\n Learningrate %f" %self.learningRate)
                plt.show()
                
                learning = True
          
                
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
            
        """
        return self.fire(testInstance) > 0.5
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad):
        
        self.weight += self.learningRate * grad
        

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
