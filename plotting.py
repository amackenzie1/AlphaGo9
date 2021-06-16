from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle
from symmetries import symmetries

def plot_probs(probs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    xpos = [1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7]
    ypos = [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7]
    num_elements = len(xpos)
    zpos = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    dx = np.ones(49)*0.5
    dy = np.ones(49)*0.5
    dz = probs[:-1]
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
    ax1.set_zlim(0,0.1)
    plt.ion()
    plt.show()
    plt.pause(0.05)

if __name__ == "__main__":
    boardlist = pickle.load(open("training_data.p", "rb"))
    board, probs, reward = boardlist[0][40]
    print(probs)
    print(np.argmax(np.array(probs)))
    print(np.sum(board, axis=-1))
    print(len(boardlist[0]))

    boards, probs = symmetries(board, probs)

    for i, j in zip(boards, probs):
        print(j)
        print(np.argmax(np.array(j)))
        print(np.sum(i, axis=-1))
