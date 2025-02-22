import JSONSplitManager,PoseModel
import os
#Will import other SplitManagers
class Main():
    def __init__(self, outputDir, inputDir, configFilePath, checkpointFilePath, dataSplitDir, outputShape, numLandmarks):
        self.outputDir = self.__validatePath(outputDir)
        self.inputDir = self.__validatePath(inputDir)
        self.configFilePath = self.__validatePath(configFilePath)
        self.checkPointFilePath = self.__validatePath(checkpointFilePath)
        self.dataSplitDir = self.__validatePath(dataSplitDir)
        self.outputShape = outputShape
        self.numLandmarks = numLandmarks
    
    def __validatePath():
        pass

    #Currently only working with JSON but I can expect a few other file types later
    def __splitManagerFactory():
        pass

    def run():
        pass


if __name__ == "__main__":
    #add args to Main class before execution
    extractKeypoints = Main()
    extractKeypoints.run()