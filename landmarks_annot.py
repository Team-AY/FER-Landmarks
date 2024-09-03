import numpy as np

JAW = np.arange(1, 17+1)-1

RIGHT_EYEBROW = np.arange(18, 22+1)-1
LEFT_EYEBROW = np.arange(23, 27+1)-1

NOSE = np.arange(28, 36+1)-1

RIGHT_EYE = np.arange(37, 42+1)-1
LEFT_EYE = np.arange(43, 48+1)-1

MOUTH = np.arange(49, 68+1)-1

ALL = np.arange(1, 68+1)-1

def feature_selection(*features):
    positions = np.array([]).astype(int)
    for feature in features:
        positions = np.append(positions, feature)
    
    # if x and y are separate
    positions = [position*2 for position in positions] + [position*2+1 for position in positions]

    # sort the values in asecnding order
    positions.sort()

    # return the positions of the features
    return positions

if __name__=="__main__":
    print(feature_selection(ALL))