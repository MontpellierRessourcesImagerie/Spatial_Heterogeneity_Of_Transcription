import numpy as np



class TableTool:


    @classmethod
    def addTableAToB(cls, tableA, tableB):
        if len(tableB.keys()) == 0:
            for key, value in tableA.items():
                    tableB[key] = value
            return
        for key, value in tableA.items():
            if key in tableB.keys():
                tableB[key] = np.append(tableB[key], [value])
        for key, value in tableA.items():
            if not key in tableB.keys():
                column = np.array([float('nan')] * len(list(tableA.keys())[0]))
                tableB[key] = np.append(column, [value])


    @classmethod
    def addColumnsTableAToB(cls, tableA, tableB):
        for key, value in tableA.items():
            tableB[key] = value