import numpy as np
import pandas as pd

class Dataset():
    
    def __init__(self, dataFrame, binaryColumns, continuousColumns, discreteColumns, nullValues):
        
        self.data = []
        self.columnCount = 0
        
        for col in binaryColumns:
            self.addBinaryColumn(dataFrame, col)
        for col in continuousColumns:
            self.addContinuousColumn(dataFrame, col)
            
        for col in discreteColumns:
            self.addDiscreteColumn(dataFrame, col, nullValues)
            
    
    def addBinaryColumn(self, dataFrame, colName):
        self.data.append((dataFrame[colName] == dataFrame[colName][0]).values.astype(np.float32).reshape(1, len(dataFrame[colName])))
        self.columnCount += 1
    def addContinuousColumn(self, dataFrame, colName):
        self.data.append(dataFrame[colName].values.astype(np.float32).reshape(1, len(dataFrame[colName])))
        self.columnCount += 1
        
    def addDiscreteColumn(self, dataFrame, colName, nullValues):
        
        oneHotMap, columnCount = Dataset.getOneHotMap(dataFrame[colName], nullValues)
        dataFrame[colName] = dataFrame[colName].map(lambda x: oneHotMap[x])
        
        for j in range(len(dataFrame[colName][0])):
            self.data.append(np.array([k[j] for k in dataFrame[colName].values]).astype(np.float32).reshape(1, len(dataFrame[colName])))
            
        self.columnCount += columnCount
            
    def getOneHotMap(dataColumn, nullValues):
        
        values = list(set(dataColumn.values) - nullValues)
        oneHotMap = {}
        base = [0]*(len(values))
        for val in nullValues:
            oneHotMap[val] = base
        
        for i in range(len(values)):
            oneHot = base.copy()
            oneHot[i] = 1
            oneHotMap[values[i]] = oneHot
            
        return oneHotMap, len(values)
        