# imports
import simpy
import numpy as np
import sys
import pandas as pd
import os
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial
import time
sys.path.append(r'C:\Users\Eshel\workspace\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

import os

import pandas as pd
import argparse
from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt

from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
# import seaborn as sns
import random
from scipy.stats import loguniform
# from butools.fitting import *
from datetime import datetime
# from fastbook import *
import itertools
from scipy.special import factorial
import pickle as pkl

is_print = False


def ser_moment_n(s, A, mom):
    '''
    ser_moment_n
    :param s:
    :param A:
    :param mom:
    :return:
    '''
    e = np.ones((A.shape[0], 1))
    try:
        mom_val = ((-1) ** mom) * factorial(mom) * np.dot(np.dot(s, matrix_power(A, -mom)), e)
        if mom_val > 0:
            return mom_val
        else:
            return False
    except:
        return False


def compute_first_n_moments(s, A, n=3):
    '''
    compute_first_n_moments
    :param s:
    :param A:
    :param n:
    :return:
    '''
    moment_list = []
    for moment in range(1, n + 1):
        moment_list.append(ser_moment_n(s, A, moment))
    return moment_list


## Defining a new class: of customers
class Customer:
    def __init__(self, p_id, arrival_time, type_cust):
        self.id = p_id
        self.arrival_time = arrival_time
        self.orig_arrival_time = arrival_time
        self.type = type_cust


is_print = False


class N_Queue_single_station:

    def __init__(self, lamda, mu, sim_time, num_stations, services, arrivals_norm):

        self.env = simpy.Environment()  # initializing simpy enviroment
        # Defining a resource with capacity 1
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)
        self.num_stations = num_stations

        self.sojourn = {}

        self.sojourn_total = []

        self.mu = mu  # service rate
        self.lamda = lamda  # inter-arrival rate

        self.servers = []
        self.df_waiting_times = pd.DataFrame([])  # is a dataframe the holds all information waiting time
        self.num_cust_durations = []
        self.df_waiting_times = []
        self.server = []
        self.last_event_time = []  # the time of the last event -
        self.num_cust_sys = []  # keeping track of number of customers in the system
        self.df_events = []  # is a dataframe the holds all information of the queue dynamic:
        self.last_depart = []
        self.inter_departures = {}

        self.services = services
        self.arrivals = arrivals_norm
        self.num_steady_size = 2000

        for station in range(num_stations):
            self.sojourn[station] = []
            self.servers.append(simpy.Resource(self.env, capacity=1))
            self.num_cust_durations.append(
                np.zeros(self.num_steady_size))  ## the time duration of each each state (state= number of cusotmers in the system)
            self.df_waiting_times.append(pd.DataFrame([]))  # is a dataframe the holds all information waiting time
            self.num_cust_sys.append(0)
            self.last_event_time.append(0)
            self.df_events.append(pd.DataFrame([]))
            self.last_depart.append(0)
            self.inter_departures[station] = []

    def run(self):

        station = 0
        self.env.process(self.customer_arrivals(station))  ## Initializing a process
        self.env.run(until=self.end_time)  ## Running the simulaiton until self.end_time

    def update_new_row(self, customer, event, station):

        new_row = {'Event': event, 'Time': self.env.now, 'Customer': customer.id,
                   'Queue lenght': len(self.servers[station].queue), 'System lenght': self.num_cust_sys[station],
                   'station': station}

        self.df_events[station] = pd.concat([self.df_events[station], pd.DataFrame([new_row])], ignore_index=True)

    #########################################################
    ################# Service block #########################
    #########################################################

    def service(self, customer, station):

        tot_time = self.env.now - self.last_event_time[station]
        self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time
        self.num_cust_sys[station] += 1
        self.last_event_time[station] = self.env.now

        with self.servers[station].request() as req:
            # Updating the a new cusotmer arrived
            # self.update_new_row(customer, 'Arrival', station)

            yield req

            # Updating the a new cusotmer entered service
            # self.update_new_row(customer, 'Enter service', station)
            # print('customer {} enter service at station {} at {}'.format(customer.id, station, self.env.now))
            ind_ser = np.random.randint(self.services[station].shape[0])
            yield self.env.timeout(self.services[station][ind_ser])
            self.sojourn[station].append(self.env.now - customer.arrival_time)

            inter_depart = self.env.now - self.last_depart[station]
            self.last_depart[station] = self.env.now
            self.inter_departures[station].append(inter_depart)

            # yield self.env.timeout(np.random.exponential(1 / self.mu))

            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][
                self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
            # we compute how much time the system had this number of customers

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now

            # Updating the a cusotmer departed the system
            # self.update_new_row(customer, 'Departure', station)

            if is_print:
                print('Departed customer {} at {}'.format(customer.id, self.env.now))

            # new_waiting_row = {'Customer': customer.id, 'WaitingTime': self.env.now - customer.arrival_time, 'station': station}
            #
            # self.df_waiting_times[station] = pd.concat([self.df_waiting_times[station], pd.DataFrame([new_waiting_row])],
            #                                   ignore_index=True)

            if station < self.num_stations - 1:
                # print('customer {} arrived at station {} at {}' .format(customer.id, station+1, self.env.now))
                customer.arrival_time = self.env.now
                self.env.process(self.service(customer, station + 1))
            else:
                self.sojourn_total.append(self.env.now-customer.orig_arrival_time)
                # print('customer {} departs the system station {} at {}'.format(customer.id, station, self.env.now))

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station):

        while True:


            ind_ser = np.random.randint(self.arrivals.shape[0])
            yield self.env.timeout(self.arrivals[ind_ser])
            # yield self.env.timeout(np.random.exponential(1 / self.lamda))

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, 1)
            # print('customer {} arrived at station {} at {}'.format(customer.id, station, self.env.now))

            self.id_current += 1

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))

    def get_steady_single_station(self):

        steady_list = []

        for station in range(self.num_stations):
            steady_list.append(self.num_cust_durations[station] / self.num_cust_durations[station].sum())

        return np.array(steady_list).reshape(self.num_stations, self.num_steady_size)


def get_ph():
    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_examples'
    else:
        path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1'

    files = os.listdir(path)

    ind_file = np.random.randint(len(files))

    data_all = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))

    ind_file1 = np.random.randint(len(data_all))

    data = data_all[ind_file1]
    return data


def get_ph_1():
    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_examples_scv_1'
    else:
        path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1'

    files = os.listdir(path)

    file = np.random.choice(files)

    data = pkl.load(open(os.path.join(path, file), 'rb'))

    return data


def create_Erlang4(lam):
    s = np.array([[1, 0, 0, 0]])

    A = np.array([[-lam, lam, 0, 0], [0, -lam, lam, 0], [0, 0, -lam, lam], [0, 0, 0, -lam]])

    return (s, A)



def sample_hyper_exponential(lambdas, probabilities, size=1):
    """
    Sample from a hyper-exponential distribution.

    Parameters:
    - lambdas: List of rates (1/mean) for the exponential distributions.
    - probabilities: List of probabilities associated with each rate.
    - size: Number of samples to generate.

    Returns:
    - samples: Array of generated samples.
    """
    # Ensure probabilities sum to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # Choose which exponential distribution to sample from for each value
    choices = np.random.choice(len(lambdas), size=size, p=probabilities)

    # Generate samples from the chosen exponential distributions
    samples = np.random.exponential(scale=1 / np.array(lambdas)[choices])

    return samples

dump = True

scv_1 = True

for sample in range(50):

    try:
        begin = time.time()
        num_stations = 5

        rate = 1   # np.random.uniform(0.5, 0.95)
        # a = np.array([0.0590414481559016, 1 - 0.0590414481559016])
        # A = np.array([[-0.118082896311803, 0], [0, -1.88191710368820]])
        # arrivals_norm = SamplesFromPH(ml.matrix(a), A, 50000000)
        # moms_arrive = np.array(compute_first_n_moments(a, A, 10)).flatten()
        # print(moms_arrive)

        # Example usage
        lambdas = [0.118082896311803, 1.88191710368820]  # Two exponential distributions with different rates
        probabilities = [0.0590414481559016, 1 - 0.0590414481559016]  # Their respective probabilities
        size = 60000000  # Number of samples to generate

        samples = sample_hyper_exponential(lambdas, probabilities, size)
        arrivals_norm = np.array(samples)
        moms_arrive = []
        for mom in range(1,11):
            moms_arrive.append( (arrivals_norm**mom).mean().item())

        services_times = {}
        moms_ser = {}
        means = {}
        for station in range(num_stations):
            if station == num_stations-1:
                means[station] = np.random.uniform(0.8,0.92)
            else:
                means[station] =  np.random.uniform(0.5,0.7)

        for station in range(num_stations):

            a = np.array([1.])
            A = np.array([[-1/means[station]]])

            moms_ser[station] = np.array(compute_first_n_moments(a, A, 10)).flatten()
            services_times[station] = np.random.exponential(means[station], 50000000)

        # pkl.dump((moms_arrive,  arrivals_norm), open('arrivals2.pkl', 'wb'))

        # moms_arrive, arrivals_norm =  pkl.load(open('arrivals.pkl', 'rb'))

        # moms_arrive = np.array(moms_arrive)
        # arrivals_norm = np.array(arrivals_norm)


        sim_time = 40000000
        mu = 1.0
        lamda = rate

        # pkl.dump(moms_arrive, open('mom_arrivals_check.pkl', 'wb'))

        # lamda, mu, sim_time, num_stations, services, arrivals_norm, moms_arrive, moms_ser = pkl.load(open('sim_setting.pkl', 'rb'))
        print('start sim')
        n_Queue_single_station = N_Queue_single_station(lamda, mu, sim_time, num_stations, services_times, arrivals_norm)
        n_Queue_single_station.run()

        mean_waiting = []
        for station in range(num_stations):
            curr_mean = np.array(n_Queue_single_station.sojourn[station]).mean()
            mean_waiting.append(curr_mean)
            print(station, curr_mean)
        print(np.array(n_Queue_single_station.sojourn_total).mean())
        # pkl.dump((mean_waiting),open('hyper_9_stations.pkl', 'wb'))
        sim_train = True
        if sim_train:
            input_ = np.concatenate((moms_arrive, moms_ser[0]), axis=0)
            output = n_Queue_single_station.get_steady_single_station()

            end = time.time()

            print(end-begin)

            inp_depart_0 = np.concatenate((moms_arrive, moms_ser[0]))
            inp_depart_0 = np.log(inp_depart_0)

            ###############################

            ########### output ############

            station = 0

            depart_0_moms = [(np.array(n_Queue_single_station.inter_departures[station])**mom).mean() for mom in range(1,11)]

            corrs_0 = []

            for corr_leg in range(1, 6):
                x1 = np.array(n_Queue_single_station.inter_departures[station][:-corr_leg])
                y1 = np.array(n_Queue_single_station.inter_departures[station][corr_leg:])
                for mom_1 in range(1,6):
                    for mom_2 in range(1,6):

                        r = np.corrcoef(x1**mom_1, y1**mom_2)
                        corrs_0.append(r[0, 1])

            corr_leg = 1
            x1 = np.array(n_Queue_single_station.inter_departures[station][:-corr_leg])
            y1 = np.array(n_Queue_single_station.inter_departures[station][corr_leg:])
            r = np.corrcoef(x1, y1)
            correlation0 = r[0, 1]

            out_depart_0 = np.concatenate((np.log(np.array(depart_0_moms)), np.array(corrs_0)))

            model_num = np.random.randint(1, 10000000)

            path_depart_0 = '/scratch/eliransc/non_renewal/depart_0_train_9_a'
            # path_depart_0 =  r'C:\Users\Eshel\workspace\data\hyper_training\depart_0_a'
            file_name = 'hyper_correlation_'+str(correlation0)+ '_means'+ str(means[0])+ '_' + str(means[1]) +  str(rate)[:5] + 'sim_time_' + str(sim_time) + 'depart_0_multi_corrs1_' + str(model_num)+ '.pkl'
            full_path_depart_0 = os.path.join(path_depart_0, file_name)

            if dump:

                pkl.dump((inp_depart_0, out_depart_0), open(full_path_depart_0, 'wb'))


            inp_depart_1 = np.concatenate((np.log(np.array(depart_0_moms)), np.array(corrs_0), np.log(np.array(moms_ser[1]))))

            ###############################
            ########### output ############
            ###############################

            for station in range(1, num_stations):

                print(station)

                depart_1_moms_ = [(np.array(n_Queue_single_station.inter_departures[station]) ** mom).mean() for mom in
                                  range(1, 11)]
                depart_1_moms = []
                for val in depart_1_moms_:
                    depart_1_moms.append(val.item())
                corrs_1 = []

                for corr_leg in range(1, 6):
                    x1 = np.array(n_Queue_single_station.inter_departures[station][:-corr_leg])
                    y1 = np.array(n_Queue_single_station.inter_departures[station][corr_leg:])
                    for mom_1 in range(1, 6):
                        for mom_2 in range(1, 6):
                            r = np.corrcoef(x1 ** mom_1, y1 ** mom_2)
                            corrs_1.append(r[0, 1])

                corr_leg = 1
                x1 = np.array(n_Queue_single_station.inter_departures[station][:-corr_leg])
                y1 = np.array(n_Queue_single_station.inter_departures[station][corr_leg:])
                r = np.corrcoef(x1, y1)
                correlation1 = r[0, 1]

                out_depart_1 = np.concatenate((np.log(np.array(depart_1_moms)), np.array(corrs_1)))

                path_depart_1 = '/scratch/eliransc/non_renewal/depart_1_train_9_a'
                # path_depart_1 = r'C:\Users\Eshel\workspace\data\deter_training\depart_1_a'
                file_name = 'hyper_correlation_' + str(correlation1) + '_station' + str(station) + '_' + str(
                    means[station]) + str(rate)[:5] + 'sim_time_' + str(sim_time) + 'depart_0_multi_corrs1_' + str(
                    model_num) + '.pkl'
                full_path_depart_1 = os.path.join(path_depart_1, file_name)

                if station == 1:
                    inp_depart_1 = np.concatenate(
                        (np.log(np.array(depart_0_moms)), np.array(corrs_0), np.log(np.array(moms_ser[station]))))
                else:
                    inp_depart_1 = np.concatenate(
                        (np.log(np.array(moms_prev)), np.array(corrs_prev), np.log(np.array(moms_ser[station]))))

                if dump:
                    pkl.dump((inp_depart_1, out_depart_1), open(full_path_depart_1, 'wb'))

                ####### Input ################

                if station == 1:
                    inp_steady_1 = np.concatenate(
                        (np.log(np.array(depart_0_moms)), np.array(corrs_0), np.log(np.array(moms_ser[station]))))
                else:
                    inp_steady_1 = np.concatenate(
                        (np.log(np.array(moms_prev)), np.array(corrs_prev), np.log(np.array(moms_ser[station]))))

                ###############################
                ########### output ############

                out_steady_1 = n_Queue_single_station.get_steady_single_station()[station]

                path_steady_1 = '/scratch/eliransc/non_renewal/steady_1_train_9_a'
                # path_steady_1 = r'C:\Users\Eshel\workspace\data\deter_training\steady_1_a'
                file_name = 'hyper_correlation_' + str(correlation1) + '_station' + str(station) + '_' + str(
                    means[station]) + str(rate)[:5] + 'sim_time_' + str(sim_time) + 'steady_1_multi_corrs1_' + str(
                    model_num) + '.pkl'
                full_path_steady_1 = os.path.join(path_steady_1, file_name)

                print(station, out_steady_1[:5])

                if dump:
                    pkl.dump((inp_steady_1, out_steady_1), open(full_path_steady_1, 'wb'))

                corrs_prev = corrs_1
                moms_prev = depart_1_moms_

    except:
        print('Exceeded 500 customers')