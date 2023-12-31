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
        mom_val = ((-1) ** mom) *factorial(mom)*np.dot(np.dot(s, matrix_power(A, -mom)), e)
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
        self.type = type_cust


is_print = False


class N_Queue_single_station:

    def __init__(self, lamda, mu, sim_time, num_stations,  services, arrivals_norm):

        self.env = simpy.Environment()  # initializing simpy enviroment
         # Defining a resource with capacity 1
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)
        self.num_stations = num_stations

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


        for station in range(num_stations):
            self.servers.append(simpy.Resource(self.env, capacity=1))
            self.num_cust_durations.append(np.zeros(500))  ## the time duration of each each state (state= number of cusotmers in the system)
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
                   'Queue lenght': len(self.servers[station].queue), 'System lenght': self.num_cust_sys[station], 'station' :station}

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
            ind_ser = np.random.randint(self.services.shape[0])
            yield self.env.timeout(self.services[ind_ser])

            inter_depart = self.env.now - self.last_depart[station]
            self.last_depart[station] = self.env.now
            self.inter_departures[station].append(inter_depart)

            # yield self.env.timeout(np.random.exponential(1 / self.mu))

            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
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

            if station < self.num_stations-1:
                # print('customer {} arrived at station {} at {}' .format(customer.id, station+1, self.env.now))
                customer.arrival_time = self.env.now
                self.env.process(self.service(customer, station+1))
            else:
                pass
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

        return np.array(steady_list).reshape(self.num_stations,500)


def get_ph():

    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_random/medium_ph_1'
    else:
        path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1'

    files = os.listdir(path)

    ind_file = np.random.randint(len(files))
    ind_file = 0




    data_all = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))

    # ind_file1 = np.random.randint(len(data_all))
    ind_file1 = 0
    data = data_all[ind_file1]
    print(data[2])
    return data



# services = get_ph()
# moms_ser = np.array(compute_first_n_moments(services[0], services[1], 10)).flatten()
#
# arrivals = get_ph()
# rate = np.random.uniform(0.5, 1)
# arrivals_norm = arrivals[3]/rate
#
# A = arrivals[1]*rate
# a = arrivals[0]
# moms_arrive = np.array(compute_first_n_moments(a, A, 10)).flatten()
#
#
# sim_time = 700
# lamda = rate
# mu = 1.0
# num_stations = 2
#
# rate = 0.5
# arrivals_norm = np.random.exponential(1/rate, 500000)
# services_exp = np.random.exponential(1, 500000)
#
# n_Queue_single_station = N_Queue_single_station(lamda, mu, sim_time, num_stations, services_exp, arrivals_norm)
# n_Queue_single_station.run()
#
# input_ = np.concatenate((moms_arrive, moms_ser), axis=0)
# output = n_Queue_single_station.get_steady_single_station()
#
# print('Stop')