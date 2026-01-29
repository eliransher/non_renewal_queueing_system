# imports
import simpy
import numpy as np
import sys
import pandas as pd
import os
import pickle as pkl
# from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
# from scipy.special import factorial
import time
from tqdm import tqdm
from scipy.special import factorial
from scipy.linalg import expm, sinm, cosm

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
        self.type = type_cust


is_print = False


class N_Queue_single_station:

    def __init__(self,  sim_time, num_stations, services, arrivals_norm, num_servers):

        self.env = simpy.Environment()  # initializing simpy enviroment
        # Defining a resource with capacity 1
        self.end_time = sim_time  # The time simulation terminate
        self.id_current = 1  # keeping track of cusotmers id
        # an event can one of three: (1. arrival, 2. entering service 3. service completion)
        self.num_stations = num_stations


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
        self.sojourn_times = []

        self.services = services
        self.arrivals = arrivals_norm

        for station in range(num_stations):
            self.servers.append(simpy.Resource(self.env, capacity=num_servers))
            self.num_cust_durations.append(
                np.zeros(500))  ## the time duration of each each state (state= number of cusotmers in the system)
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


            yield req

            ind_ser = np.random.randint(self.services.shape[0])
            yield self.env.timeout(self.services[ind_ser])



            tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            self.num_cust_durations[station][
                self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
            # we compute how much time the system had this number of customers

            self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            self.last_event_time[station] = self.env.now

            sojourn_time = self.env.now.item() -  customer.arrival_time.item()
            self.sojourn_times.append(sojourn_time)

    #########################################################
    ################# Arrival block #########################
    #########################################################

    def customer_arrivals(self, station):

        while True:

            # ind_ser = np.random.randint(self.arrivals.shape[0])
            self.id_current += 1
            yield self.env.timeout(self.arrivals[self.id_current%self.arrivals.shape[0]])

            curr_id = self.id_current
            arrival_time = self.env.now
            customer = Customer(curr_id, arrival_time, 1)

            if is_print:
                print('Arrived customer {} at {}'.format(customer.id, self.env.now))

            self.env.process(self.service(customer, station))

    def get_steady_single_station(self):

        steady_list = []

        for station in range(self.num_stations):
            steady_list.append(self.num_cust_durations[station] / self.num_cust_durations[station].sum())

        return np.array(steady_list).reshape(self.num_stations, 500)


def sample_map(low_max_size=11):
    option = np.random.randint(1, 6)
    # print('A', option)

    if option == 1:
        D0a, D1a = generate_renewal_MAP(low_max_size)
    elif option == 2:
        D0a, D1a = give_strong_pos_cor(low_max_size)
    elif option == 3:
        D0a, D1a = give_strong_neg_cor(low_max_size)
    elif option == 4:
        D0a, D1a = random_map(
            n=low_max_size,
            corr_strength=0.75,  # try positive mild correlation
            corr_cap=0.5,  # keep |rho1| <= 0.25
            heavy_tail_mix=0.6,  # more variability in SCV/skew/kurt
            max_tries=5000,
        )
    elif option == 5:
        D0a, D1a = random_map_negative(
            n=low_max_size,
            neg_strength=0.85,  # stronger alternation tendency
            rho_cap=0.99,  # keep |rho1| <= 0.25
            want_rho_at_most=-0.031,  # ensure negative
            heavy_tail_mix=0.7,
            max_tries=15000
        )
    return D0a, D1a


def generate_renewal_MAP(max_degree):
    if True:
        degree = np.random.randint(10, max_degree)
        option = np.random.randint(1, 4)
        if option == 1:

            n = 10
            a, T = get_PH_general_with_zeros(degree, n)
            a = a / a.sum()
            a = np.array(a).flatten()
            T = np.array(T)
        elif option == 2:

            a, T, _, _ = sample_coxian(degree=degree, max_rate=20)
            a = np.array(a).flatten()
            T = np.array(T)
        else:

            dat = sample(degree)
            a = dat[0]
            T = dat[1]

        D0, D1 = ph_to_map_renewal(a, T)
        return D0, D1

    # except:
    #     return print('numerical error') #generate_renewal_MAP(max_degree)


def map_nth_moment(D0, D1, n):
    """
    Compute E[X^n] for a MAP(D0,D1).
    """
    m = D0.shape[0]
    e = np.ones((m, 1))

    invD0 = np.linalg.inv(-D0)
    P = invD0 @ D1

    # stationary distribution of P
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    pi = pi.reshape(1, -1)

    moment = math.factorial(n) * (pi @ np.linalg.matrix_power(invD0, n) @ e)[0, 0]
    return moment



def create_mom_cor_vector(D0, D1):
    mom_cors = []

    for mom in range(1, 11):
        mom_cors.append(map_nth_moment(D0, D1, mom))

    for k in range(1, 6):
        for i in range(1, 6):
            for j in range(1, 6):
                mom_cors.append(map_power_corr(D0, D1, k=k, i=i, j=j))
    return np.array(mom_cors)

def get_ph():
    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_samples'
    else:
        path = r'C:\Users\Eshel\workspace\data\PH_samples'

    folders = os.listdir(path)

    folder_ind =   np.random.randint(len(folders))
    files = os.listdir(os.path.join(path, folders[folder_ind]))
    ind_file1 = np.random.randint(len(files))

    data_all = pkl.load(open(os.path.join(path,folders[folder_ind], files[ind_file1]), 'rb'))

    return data_all

def sample_rate(size):
    x = np.random.dirichlet(np.ones(size))
    if x.min() > 0.01:
        return x
    else:
        return sample_rate(size)

for sample in range(1):


    num_arrival_streams = 2

    arrival_rates = sample_rate(num_arrival_streams)

    begin = time.time()

    rho = np.random.uniform(0.02, 0.95)

    arrival_dict = {}

    num_arrival_streams = 2

    arrival_rates = sample_rate(num_arrival_streams)

    for stream in range(num_arrival_streams):
        D0, D1 = sample_map()
        x = SamplesFromMAP(ml.matrix(D0.copy()), ml.matrix(D1.copy()), 400000)
        x = x / arrival_rates[stream]
        D0 = D0 * arrival_rates[stream]
        D1 = D1 * arrival_rates[stream]
        resa = create_mom_cor_vector(D0.copy(), D1.copy())
        arrival_dict[stream] = (D0, D1, arrival_rates[steam], x, resa)


    # moms_arrive = arrivals[2]


    services = get_ph()

    num_servers = np.random.randint(1, 6)
    rate = 1/(rho * num_servers)

    services_norm =  services[3] / rate

    A = services[1] * rate
    a = services[0]


    moms_ser = np.array(compute_first_n_moments(a, A, 10)).flatten()

    mom_1_ser = moms_ser[0]
    mom_2_ser = moms_ser[1]

    var_ser = mom_2_ser - mom_1_ser ** 2
    scv_ser = var_ser / mom_1_ser ** 2

    if rho > 0.8:
        rho_factor = 1.25
    elif rho > 0.6:
        rho_factor = 1.1
    elif rho > 0.4:
        rho_factor = 1.05
    else:
        rho_factor = 1.

    if scv_ser > 10:
        scv_ser_factor = 1.25
    elif scv_ser > 4:
        scv_ser_factor = 1.15
    elif scv_ser > 2:
        scv_ser_factor = 1.05
    else:
        scv_ser_factor = 1.


    sim_time = 60000000
    sim_time = int(sim_time * rho_factor * scv_ser_factor)
    mu = 1.0
    num_stations = 1

    print(num_servers)

    lamda = rate
    inps = []
    outputs1 = []
    outputs2 = []

    try:

        for trails in tqdm(range(1)):

            n_Queue_single_station = N_Queue_single_station(sim_time, num_stations, services_norm, arrivals[3],
                                                            num_servers)
            n_Queue_single_station.run()

            input_ = np.concatenate((moms_arrive, moms_ser), axis=0)
            # output = n_Queue_single_station.get_steady_single_station()

            end = time.time()

            print(end - begin)

            model_num = np.random.randint(1, 10000000)

            ########### output ############

            station = 0

            ####### Input ################

            inp_steady_0 = np.concatenate((np.log(moms_arrive), np.log(moms_ser), np.array([num_servers])))
            inps.append(inp_steady_0)
            ###############################
            ########### output ############

            output1 = n_Queue_single_station.get_steady_single_station()[0]
            output2 = np.array(n_Queue_single_station.sojourn_times).mean().item()

            mean_val = (np.arange(output1.shape[0])*output1).sum()

            outputs1.append(output1)
            outputs2.append(output2)



        if sys.platform == 'linux':
            path_steady_0 = '/scratch/eliransc/n_servers_single'
        else:
            path_steady_0 = r'C:\Users\Eshel\workspace\data\ggc_training_data'

        file_name =  'rho_' + str(rho)[:5] + '_num_servers_' + str(num_servers) + '_sim_time_' + str(sim_time) + 'steady_' + str(
            model_num) + '.pkl'

        full_path_steady_0 = os.path.join(path_steady_0, file_name)
        pkl.dump((inps, outputs1, outputs2), open(full_path_steady_0, 'wb'))

    except:
        print('cannot find ph dist')

