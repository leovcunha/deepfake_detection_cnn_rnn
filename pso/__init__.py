import numpy as np
import math


class Particle(object):
    """
    Particle class

    Parameters:
    lb: lower bound - list with size of hyperparameter numbers each element contain a lower limit
    ub: upper \bound - list with size of hyperparameter numbers each element contain a upper limit
    types: list of types of each hyperparameter (if int or float)
    """

    def __init__(self, lb, ub, types) -> None:

        assert len(lb) == len(ub), "different ranges for lb and ub"
        assert len(lb) == len(types), "types and bounds have diff range"
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.types = types
        self.inttypes = []  # which hyperparam indexes are integer?
        self.V = []
        self.pos = []
        self.fitness_values = []

        # initialize first position
        pos_1 = []
        v_1 = []
        for i in range(len(self.lb)):
            assert self.lb[i] <= self.ub[i]

            if self.lb[i] == self.ub[i]:
                pos_1.append(self.lb[i])
                v_1.append(0)
                continue
            rng = self.ub[i] - self.lb[i]
            # position initialized randomly with uniform distr.
            rand_pos = np.random.uniform(self.lb[i], self.ub[i])

            # velocity initialized with normal distribution and range
            # of ub-lb as 3 standard deviations
            rand_v = np.random.normal(0, rng / 3)
            if types[i] == int:
                rand_pos = int(rand_pos)
                rand_v = int(rand_v)
                self.inttypes.append(i)
            pos_1.append(rand_pos)
            v_1.append(rand_v)

        self.pos.append(np.array(pos_1))
        self.V.append(np.array(v_1))
        self.pbest = []
        self.pbest_score = []

    def update_pbest(self, fitness_val):
        """
        Compares fitness value with current pbest_Score if smaller updates pbest

        Parameters:
        fitness_val: loss score from the model training
        """
        self.fitness_values.append(fitness_val)

        if not self.pbest_score or fitness_val < self.pbest_score[-1]:
            self.pbest_score.append(fitness_val)
            self.pbest.append(self.pos[-1])
        else:
            self.pbest_score.append(self.pbest_score[-1])
            self.pbest.append(self.pbest[-1])

    def update_V(self, w, c1, c2, gbest):
        """
        Update velocity according with classic PSO algorithm

        Parameters:
        w: inercia
        c1: acceleration for cognitive component
        c2: acceleration for social component
        gbest: global best position
        """
        r1 = np.random.uniform(size=np.array(self.pos[-1]).shape)
        r2 = np.random.uniform(size=np.array(self.pos[-1]).shape)

        self.V.append(
            w * self.V[-1]
            + c1 * r1 * (self.pbest[-1] - self.pos[-1])
            + c2 * r2 * (gbest - self.pos[-1])
        )

    def update_pos(self):
        """
        Updates position according with classic PSO algorithm

        """
        self.pos.append(self.pos[-1] + self.V[-1])

        for idx in self.inttypes:
            self.pos[-1][idx] = int(self.pos[-1][idx])

        # if position values gets larger or smaller than limits clip the value
        clip_lower = self.pos[-1] < self.lb
        clip_upper = self.pos[-1] > self.ub

        self.pos[-1][clip_lower] = self.lb[clip_lower]
        self.pos[-1][clip_upper] = self.lb[clip_upper]


class PSO(object):
    """
    Sets up particles and run PSO algorithm

    Parameters:
    w: inercia
    hyperparam: dictionary with keys as names and value as lists lowest and highest\n
    value for each hyperparameters
    c1: acceleration for cognitive component
    c2: acceleration for social component
    num particles: integer number
    """

    def __init__(self, w: float, hyperparam: dict, c1=0.5, c2=0.5, num_particles=20):
        # initial paramters
        self.w = w  # inertia weight
        self.c1 = c1  # acceleration 1
        self.c2 = c2  # acceleration 2
        # particles info
        self.hyperparam = hyperparam
        # get type int or float of each hyperparam to use later
        self.hyperparam_types = []
        for param_space in self.hyperparam.values():
            if type(param_space[1]) == int:
                self.hyperparam_types.append(int)
            else:
                self.hyperparam_types.append(float)

        self.num_particles = num_particles
        self.particles = []
        self.gbest = []
        self.gbest_score = []
        # defining lower and upper bounds for each hyperparam
        self.lb = [hyperparam[i][0] for i in hyperparam]
        self.ub = [hyperparam[i][-1] for i in hyperparam]
        self.initialize_particles()

    def initialize_particles(self):

        for i in range(self.num_particles):
            self.particles.append(Particle(self.lb, self.ub, self.hyperparam_types))

    def update_gbest(self, g, position, fitness_val):
        """
        Update global best if fitness_val < gbest_score looking at current\n and future generations

        Parameters:

        g - generation
        position - current particle position
        fitness_val - value of current particle_position

        """
        if fitness_val < self.gbest_score[g]:
            self.gbest_score[g] = fitness_val
            self.gbest[g] = position
            for i in range(g + 1, len(self.gbest)):
                self.gbest_score[i] = self.gbest_score[g]
                self.gbest[i] = self.gbest[g]

    def run(self, train_model, generations: int, adaptative=False):
        """
        Run the complete PSO generations finding best hyperparameter combination

        Parameters:

        train_model - function to train model and return loss_val
        generations - integer number of


        """

        self.gbest = [0 for g in range(generations)]
        self.gbest_score = [math.inf for g in range(generations)]

        for g in range(generations):
            # fitness function
            print("generation ", g, ":\n")
            if adaptative:
                self.c1 = 2.5 - (2.5 - 0.5) * g / generations
                self.c2 = 0.5 + (2.5 - 0.5) * g / generations

            for particle in self.particles:
                # get params for each particle position and convert the int types
                params = particle.pos[-1].tolist()
                print("particle params: ", params)
                for i in particle.inttypes:
                    params[i] = int(params[i])
                # calculate loss value for each particle
                fitness_value = train_model(*params)

                # update local and global best
                particle.update_pbest(fitness_value)
                self.update_gbest(g, particle.pos[-1], fitness_value)
                print("fitness val ", fitness_value)
                print("personal best ", particle.pbest[-1])
                print("global best ", self.gbest[g])
                print("best score ", self.gbest_score[g])
                # update V
                particle.update_V(self.w, self.c1, self.c2, self.gbest[-1])
                particle.update_pos()
