import numpy as np
import copy
from ple import PLE
# from ple.games.raycastmaze import RaycastMaze
# from ple.games.pong_z import Pong
from ple.games.catcher_z import Catcher
import matplotlib.pyplot as pl
import time

from support import Support

from scipy import ndimage

fig = pl.ion()

class Node:
    """
    Define a node (stimulus, e.g. a ball), represent its temporal history (when was it visible) as a log-compressed supported dimension, as well as its distance from every wall (four boundaries of the enviroment) and its every feature (e.g. brightness). Represent history of every distance and every feature also as a supported dimension.
    """

    def __init__(self, time_buff_len, tstr_min, tstr_max, space_buff_len, sstr_min, sstr_max, n_features, feature_buff_len, fstr_min, fstr_max, name='', Nt=2010):
        self.Nt = Nt
        self.time_buff_len = time_buff_len
        self.space_buff_len = space_buff_len
        self.feature_buff_len = feature_buff_len
        self.name = name
        self.value = np.zeros(Nt)  # initialize node value
        # constuct a log-spaced supported dimension for temporal history
        self.memory = Support(xstr_min=tstr_min, xstr_max=tstr_max, buff_len=self.time_buff_len, g=1, Nt=self.Nt)
        self.space = []
        for i in range(4):  # represent distance from every wall
            self.space.append(Support(xstr_min=sstr_min, xstr_max=sstr_max, buff_len=self.space_buff_len, g=1, Nt=self.Nt))
        self.space_memory = []
        for i in range(4):  # represent memory of the distance
            for j in range(self.space_buff_len):
                self.space_memory.append(Support(xstr_min=tstr_min, xstr_max=tstr_max, buff_len=time_buff_len, g=1, Nt=self.Nt))
        self.features = []
        for i in range(n_features):  # represent every feature
            self.features.append(Support(xstr_min=fstr_min, xstr_max=fstr_max, buff_len=self.feature_buff_len, g=1, Nt=self.Nt))
        self.features_memory = []
        for i in range(n_features):  # represent memory of every feature
            for j in range(self.feature_buff_len):
                self.features_memory.append(Support(xstr_min=tstr_min, xstr_max=tstr_max, buff_len=self.feature_buff_len, g=1, Nt=self.Nt))

    def construct_place_fields(self):
        """
        Compute a product of vertical and horizontal spatial distances to constuct 2D support of the space (place cells).
        """
        self.place_fields = np.zeros((self.space_buff_len, self.space_buff_len))
        self.place_fields_memory = np.zeros((self.time_buff_len, self.space_buff_len, self.space_buff_len))
        for i in range(self.space_buff_len):
            for j in range(self.space_buff_len):
                self.place_fields[i, j] = self.space[1][i] * self.space[2][j] * self.space[3][self.space_buff_len-i] * self.space[4][self.space_buff_len-j]
        for k in range(self.time_buff_len):
            for i in range(self.space_buff_len):
                for j in range(self.space_buff_len):
                    self.place_fields_memory[k, i, j] = self.space_memory[1][i] * self.space_memory[space_buff_len][j] * self.space_memory[2*space_buff_len][self.space_buff_len-i] * self.space_memory[3*space_buff_len][self.space_buff_len-j]


class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]
        #return self.actions[0] # go down in pong, right in catch


###################################
#game = RaycastMaze(
#    map_size=6
#)  # create our game

#game = Pong(width=256, height=200)
game = Catcher(width=256, height=200)

fps = 30 # fps we want to run atx
frame_skip = 2
num_steps = 1
force_fps = False  # slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = 15000

# make a PLE instance.
p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
        force_fps=force_fps, display_screen=display_screen)

# our Naive agent!
agent = NaiveAgent(p.getActionSet())

# init agent and game.
p.init()

# lets do a random number of NOOP's
for i in range(np.random.randint(0, max_noops)):
    reward = p.act(p.NOOP)

tmp = []
threshold = 100

n_objects = 0
max_len = 256
max_height = 200
Nf = 256

n_features = 5  # max number of features
n_reward_values = 3  # total number of different possible reward values
n_actions = 3  # total number of action the agent could take (including no action)
'''
# Alocate space for the associative matrices
M_rew = np.zeros((n_reward_values, n_features))
M_act = np.zeros((n_reward_values, n_features))
M_mem = np.zeros((n_features, n_features, n_mem))
'''


fig = pl.figure(figsize=(9, 4), dpi=80, facecolor='w', edgecolor='k')
n_subplots = 5

objects = []  # main object list

# start our training loop
for i in range(nb_frames):
    # if the game is over
    if p.game_over():
        p.reset_game()

    obs = p.getScreenGrayscale()
    action = agent.pickAction(reward, obs)
    reward = p.act(action)
    #print reward

    if reward == -6:
        reward = 0

    if reward == 6:
        reward = 0

    # print f

    labeled, n_objects_now = ndimage.label(obs > threshold)
    if i > 0:
        if i > 1:
            d = np.zeros((n_objects_now, n_objects))  # store distances between presently and previously detected objects
        tmp_objects = []  # lists that stores new objects before we identify which old object corresponds to which new object
        for j in range(n_objects_now):
            rows, cols = np.where(labeled == j+1)
            if (np.max(rows) - np.min(rows) < max_height-1) & (np.max(cols) - np.min(cols) < max_len-1):
                tmp_object = Node(time_buff_len=5, tstr_min=1, tstr_max=50, space_buff_len=50, sstr_min=1, sstr_max=256, n_features=2, feature_buff_len=50, fstr_min=1, fstr_max=256)
                tmp_object.space[0].set_input(i, np.int(np.mean(np.unique(rows))))
                tmp_object.space[1].set_input(i, max_len-np.int(np.mean(np.unique(rows))))
                tmp_object.space[2].set_input(i,  np.int(np.mean(np.unique(cols))))
                tmp_object.space[3].set_input(i, max_height-np.int(np.mean(np.unique(cols))))
                tmp_object.features[0].set_input(i, np.max(rows) - np.min(rows))
                tmp_object.features[1].set_input(i, np.max(cols) - np.min(cols))
                if i == 1:
                    objects.append(tmp_object)
                    n_objects = len(objects)
                else:
                    #d = np.zeros((n_objects))
                    tmp_objects.append(tmp_object)
                    k = tmp_object.space[0].k
                    for gg in range(n_objects):
                        #Find euclidian distance form the object found in this frame from each previously found object.
                        for l in range(len(tmp_object.space)):
                            d[j, gg] = d[j, gg] + np.square(np.argmax(tmp_object.space[l].til_f[k:-k, i]) - np.argmax(objects[gg].space[l].til_f[k:-k, i-1]))
                        for l in range(len(tmp_object.features)):
                            d[j, gg] = d[j, gg] + np.square(np.argmax(tmp_object.features[l].til_f[k:-k, i]) - np.argmax(objects[gg].features[l].til_f[k:-k, i-1]))
        # assign objects detected at this step to objects detected previously
        if i > 1:
            # if n_objects_now > n_objects:  # if more new objects
            #     for l in range(n_objects):
            #         objects[np.argwhere(d == d.min())[0][0]] = tmp_objects[np.argwhere(d == d.min())[0][1]]
            #         d[np.argwhere(d == d.min())[0][0], :] = float("inf")
            #         d[:, np.argwhere(d == d.min())[0][1]] = float("inf")
            #     for l in range(n_objects_now-n_objects):
            #         objects.append(tmp_objects[np.argwhere(d == d.min())[0][1]])
            #         d[np.argwhere(d == d.min())[0][0], :] = float("inf")
            #         d[:, np.argwhere(d == d.min())[0][1]] = float("inf")
            #     n_objects = n_objects_now
            # else:
            for l in range(n_objects_now):
                objects[np.argwhere(d == d.min())[0][1]] = tmp_objects[np.argwhere(d == d.min())[0][0]]
                d[np.argwhere(d == d.min())[0][0], :] = float("inf")
                d[:, np.argwhere(d == d.min())[0][1]] = float("inf")

        pl.hold(False)
        for j in range(n_objects):
            for l in range(4):
                _ = pl.subplot(n_subplots, 4, j+1+l*4)
                _ = pl.plot(objects[j].space[l].til_f[objects[j].space[l].k:-objects[j].space[l].k, i])
            for l in range(2):
                _ = pl.subplot(n_subplots, 4, j+3+l*4)
                _ = pl.plot(objects[j].features[l].til_f[objects[j].features[l].k:-objects[j].features[l].k, i])
        _ = pl.subplot(n_subplots, 4, n_subplots*4)
        _ = pl.imshow(labeled.T, cmap='gray')

    #pl.savefig('figures/M_%d.png' % f)
    #p.saveScreen("tmp_game_screen_%d.png" % i)
