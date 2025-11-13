




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

    def __init__(self, lamda, mu, sim_time, num_stations, services, arrivals_norm):

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
        self.sojourn_times_per_station = {}

        self.services = services
        self.arrivals = arrivals_norm
        self.num_steady_size = 2000

        for station in range(num_stations):
            self.servers.append(simpy.Resource(self.env, capacity=1))
            self.num_cust_durations.append(
                np.zeros(self.num_steady_size))  ## the time duration of each each state (state= number of cusotmers in the system)
            self.df_waiting_times.append(pd.DataFrame([]))  # is a dataframe the holds all information waiting time
            self.num_cust_sys.append(0)
            self.last_event_time.append(0)
            self.df_events.append(pd.DataFrame([]))
            self.last_depart.append(0)
            self.inter_departures[station] = []
            self.sojourn_times_per_station[station] = []

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

        # tot_time = self.env.now - self.last_event_time[station]
        # self.num_cust_durations[station][self.num_cust_sys[station]] += tot_time
        # self.num_cust_sys[station] += 1
        # self.last_event_time[station] = self.env.now

        with self.servers[station].request() as req:
            # Updating the a new cusotmer arrived
            # self.update_new_row(customer, 'Arrival', station)

            yield req

            # Updating the a new cusotmer entered service
            # self.update_new_row(customer, 'Enter service', station)
            # print('customer {} enter service at station {} at {}'.format(customer.id, station, self.env.now))
            ind_ser = np.random.randint(self.services[station].shape[0])
            yield self.env.timeout(self.services[station][ind_ser])

            sojourn_time = self.env.now - customer.arrival_time

            self.sojourn_times_per_station[station].append(sojourn_time)

            inter_depart = self.env.now - self.last_depart[station]
            self.last_depart[station] = self.env.now
            self.inter_departures[station].append(inter_depart)

            # yield self.env.timeout(np.random.exponential(1 / self.mu))

            # tot_time = self.env.now - self.last_event_time[station]  # keeping track of the last event
            # self.num_cust_durations[station][
            #     self.num_cust_sys[station]] += tot_time  # Since the number of customers in the system changes
            # we compute how much time the system had this number of customers

            # self.num_cust_sys[station] -= 1  # updating number of cusotmers in the system
            # self.last_event_time[station] = self.env.now

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

        return np.array(steady_list).reshape(self.num_stations, self.num_steady_size)

def give_samples_moms_erlang4(rho):
    # lam = 4 / (rho)
    #
    # s, A = create_Erlang4(lam)
    # samples = SamplesFromPH(ml.matrix(s), A, 20000000)

    lam = rho / 4
    samples = np.random.gamma(shape=4, scale=lam, size=20000000)



    moms = []
    for mom in range(1, 11):
        moms.append((samples ** mom).mean())

    return (moms, samples)


def get_ph_larve_scv():
    scv = 0
    while (scv < 3.5) | (scv > 12):

        if sys.platform == 'linux':
            path = '/scratch/eliransc/ph_samples'
        else:
            path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1'

        files = os.listdir(path)

        ind_file = np.random.randint(len(files))

        data_all = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))

        ind_file1 = np.random.randint(len(data_all))

        data = data_all[ind_file1]
        scv = (data[2][1] - data[2][0] ** 2) / data[2][0] ** 2

    return data


def get_ph_by_scv_val(lb, ub):


    scv_flag = True


    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_samples'
    else:
        path = r'C:\Users\Eshel\workspace\data\ph_examples'

    files = os.listdir(path)

    ind_file = np.random.randint(len(files))

    path = os.path.join(path, files[ind_file])

    files = os.listdir(path)

    ind_file = np.random.randint(len(files))

    data_all = pkl.load(open(os.path.join(path, files[ind_file]), 'rb'))

    ind_file1 = np.random.randint(len(data_all))

    # data = data_all[ind_file1]
    #
    # if data[2][1] - 1 > lb:
    #     if data[2][1] - 1 < ub:
    #         scv_flag = False

    return data_all

def get_ph():
    if sys.platform == 'linux':
        path = '/scratch/eliransc/ph_samples'
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
        path = '/scratch/eliransc/ph_samples'
    else:
        path = r'C:\Users\user\workspace\data\ph_random\ph_mean_1'

    files = os.listdir(path)

    file = np.random.choice(files)

    data = pkl.load(open(os.path.join(path, file), 'rb'))

    return data


dump = True

scv_1 = True

for sample in range(5000):
    print('##################################')
    if True:
        begin = time.time()
        num_stations = 2
        # if scv_1 == True:
        #     arrivals = get_ph_larve_scv()
        # else:
        #     arrivals = get_ph_larve_scv()  ## should be returned to get_ph()

        arrivals = get_ph_by_scv_val(0, 15)

        rate = 1  # np.random.uniform(0.5, 0.95)

        arrivals_norm = arrivals[3]/rate
        A = arrivals[1]*rate
        a = arrivals[0]
        moms_arrive = np.array(compute_first_n_moments(a, A, 10)).flatten()

        services_times = {}
        moms_ser = {}
        rates_ser = [np.random.uniform(0.0,0.8), np.random.uniform(0.0,0.95)]
        for station in range( num_stations):
            services = get_ph_by_scv_val(0, 15) ## should be returned to get_ph()
            rate = rates_ser[station] #np.random.uniform(0.75, 0.9)
            ser_norm = services[3] * rate

            A = services[1] / rate
            a = services[0]

            moms_ser[station] = np.array(compute_first_n_moments(a, A, 10)).flatten()
            services_times[station] = ser_norm



        sim_time = 70000000
        mu = 1.0
        lamda = rate
        model_num = np.random.randint(1, 1000000)
        print(model_num)



        path_depart_0 = '/scratch/eliransc/sojourn/depart_0'
        path_depart_1 = '/scratch/eliransc/sojourn/depart_1'

        path_sojourn_0 = '/scratch/eliransc/sojourn/sojourn_0'
        path_sojourn_1 = '/scratch/eliransc/sojourn/sojourn_1'


        sojourn_times_by_0 = []
        sojourn_times_by_1 = []

        for ind in range(1):

            # lamda, mu, sim_time, num_stations, services, arrivals_norm, moms_arrive, moms_ser = pkl.load(open('sim_setting.pkl', 'rb'))
            print('starting sim')
            n_Queue_single_station = N_Queue_single_station(lamda, mu, sim_time, num_stations, services_times, arrivals_norm)
            n_Queue_single_station.run()
            print('ending sim')
            input_ = np.concatenate((moms_arrive, moms_ser[0]), axis=0)
            # output = n_Queue_single_station.get_steady_single_station()

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



            file_name = 'depart_0_trial_num_' + str(ind) +  'correlation_'+str(correlation0)+ '_' +  str(rate)[:5] + 'sim_time_' + str(sim_time) + '_model_num_' + str(model_num)+ '.pkl'
            full_path_depart_0 = os.path.join(path_depart_0, file_name)

            station = 0
            print('station 0')
            station0 = []
            for mom in range(1, 21):
                station0.append((np.array(n_Queue_single_station.sojourn_times_per_station[0]) ** mom).mean())
            sojourn_times_by_0.append(station0)

            print('station 1')
            station1 = []
            for mom in range(1, 21):
                station1.append((np.array(n_Queue_single_station.sojourn_times_per_station[1]) ** mom).mean())
            sojourn_times_by_1.append(station1)

            print(station0, station1)



            if dump:

                pkl.dump((inp_depart_0, out_depart_0), open(full_path_depart_0, 'wb'))


            inp_depart_1 = np.concatenate((np.log(np.array(depart_0_moms)), np.array(corrs_0), np.log(np.array(moms_ser[1]))))

            ###############################
            ########### output ############

            station = 1

            depart_1_moms = [(np.array(n_Queue_single_station.inter_departures[station])**mom).mean() for mom in range(1,11)]

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

            # '/scratch/eliransc/non_renewal/check_sim_accuracy'

            file_name = 'depart_1_trial_num_' + str(ind) +  'correlation_'+str(correlation1)+ '_' + str(rate)[:5] + 'sim_time_' + str(sim_time) + '_model_num_' + str(model_num)+ '.pkl'
            full_path_depart_1 = os.path.join(path_depart_1, file_name)
            if dump:
                pkl.dump((inp_depart_1, out_depart_1), open(full_path_depart_1, 'wb'))

            ####### Input ################


            inp_sojourn_0 = np.concatenate((moms_arrive, moms_ser[0]))
            inp_sojourn_0 = np.log(inp_sojourn_0)

            ###############################
            ########### output ############

            station = 0
            depart_1_moms = [(np.array(n_Queue_single_station.inter_departures[station])**mom).mean() for mom in range(1, 11)]
            out_sojourn_0 = np.log(np.array(station0))


            # path_steady_0 = '/scratch/eliransc/non_renewal/steady_0_train_long'
            #
            file_name = str(rate)[:5] + 'sim_time_' + str(sim_time) + 'sojourn_0_multi_corrs1_' + str(model_num)+ '.pkl'
            full_path_steady_0 = os.path.join(path_sojourn_0, file_name)
            if dump:
                pkl.dump((inp_sojourn_0, out_sojourn_0), open(full_path_steady_0, 'wb'))


            ####### Input ################

            inp_sojourn_1 = np.concatenate((np.log(np.array(depart_0_moms)), np.array(corrs_0), np.log(np.array(moms_ser[1]))))

            ###############################
            ########### output ############

            station = 1

            out_steady_1 = np.log(np.array(station1))

            file_name = 'steady_1_trial_num_' + str(ind) +  'correlation_' + str(correlation0)+ '_' + str(rate)[:5] + 'sim_time_' + str(sim_time) + '_model_num_' + str(model_num)+ '.pkl'
            full_path_sojourn_1 = os.path.join(path_sojourn_1, file_name)
            if dump:
                pkl.dump((inp_sojourn_1, out_steady_1), open(full_path_sojourn_1, 'wb'))


        # print('Sojourn times by station 0:')
        # for mom in range(1, 21):
        #     print(mom, 100*(sojourn_times_by_0[0][mom-1] - sojourn_times_by_0[1][mom-1])/sojourn_times_by_0[1][mom-1])
        #
        # print('Sojourn times by station 1:')
        # for mom in range(1, 21):
        #     print(mom, 100*(sojourn_times_by_1[0][mom-1] - sojourn_times_by_1[1][mom-1])/sojourn_times_by_1[1][mom-1])
        ###############################
        ######### Full system #########
        ###############################

        ####### Input ################

        # inp_full_system = np.concatenate((np.log(np.array(moms_arrive)),  np.log(np.array(moms_ser[0])), np.log(np.array(moms_ser[1]))))

        ###############################
        ########### output ############

        # out_full = (out_steady_0, out_steady_1)
        # out_full_inter = (out_depart_0, out_depart_1)
        #
        # path_sys = '/scratch/eliransc/non_renewal/full_system'
        #
        # file_name = str(rate)[:5] + 'sim_time_' + str(sim_time) + 'full_sys_multi_corrs1_' + str(model_num) + '.pkl'
        # full_path_sys = os.path.join(path_sys, file_name)
        # pkl.dump((inp_full_system, out_full, out_full_inter), open(full_path_sys, 'wb'))


    # except:
    #     print('Exceeded 500 customers')