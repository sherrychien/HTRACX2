
# coding: utf-8

import numpy as np
from scipy.spatial import distance


def backprop(Input_pattern, Hidden_out_activations, Output_out_activations, Target, Learning_rate, IH_wts, HO_wts):
    s = np.shape(Input_pattern)
    Error_Raw = Output_out_activations - Target
    Error_Op = Error_Raw * tanh_deriv(Output_out_activations)
    dE_dW = np.transpose(Hidden_out_activations).dot(Error_Op)
    HO_dwts = (-1) * Learning_rate * dE_dW
    HO_wts = HO_wts + HO_dwts

    Error_Hidden = Error_Op.dot(HO_wts.transpose()) 
    Error_Hidden = Error_Hidden* tanh_deriv(Hidden_out_activations)
    Error_Hidden = Error_Hidden[0,0:np.int((s[1]-1)/2)]
    dE_dW = np.transpose(Input_pattern).dot([Error_Hidden])
    IH_dwts = (-1) * Learning_rate * dE_dW
    IH_wts = IH_wts + IH_dwts

    return IH_wts, HO_wts

def tanh_compress(x):
    a = np.tanh(x)
    return a


def tanh_deriv(tanh_compress_output):
    Fahlman_offset = 0.01
    a = 1 - tanh_compress_output * tanh_compress_output + Fahlman_offset
    return a



# In[3]:
import numpy as np

def feedforward(Input, IH_wts, HO_wts):

    bias_node = np.array([-1])

    Hid_net_act = Input.dot(IH_wts)
    Hid_out_act = tanh_compress(Hid_net_act)

    Hid_out_act = np.concatenate((Hid_out_act, [bias_node]), axis = 1)

    Out_net_act = Hid_out_act.dot(HO_wts)
    Output_out_act = tanh_compress(Out_net_act)

    return Hid_out_act, Output_out_act

def TRACX(t, delta_lower, delta_self, Hid_lower, Hid_self, RHS_lower, RHS_self, IH_wts, HO_wts, learning_rate, tau, noise_level, learning, objective_func):
    bias_node = np.array([-1])
    k = 1/2. #scaling factor for delta_self
    a = tanh_compress(k*delta_self)
    b = tanh_compress(k*delta_lower)
    Delta = 10
    Target = 0
    Output = 0
    Q =np.zeros((np.shape(Hid_lower)))
    if t == 0:
        t1 = (1-b)*Hid_lower + b*RHS_lower
    else:
        ctx=tau/(tau+a)
        inp=a/(tau+a)
        t2 = ctx*Hid_self + inp*RHS_self
        t1 = (1-b)*Hid_lower+ b*RHS_lower
        Input = np.concatenate((t2,t1),axis = 0)
        s = np.shape(Input)
        Input = np.concatenate(([Input], [bias_node]), axis = 1)
        Q, Output = feedforward(Input, IH_wts, HO_wts)
        Target = Input[0,0:s[0]]
        if learning == True:
            IH_wts, HO_wts = backprop(Input, Q, Output, Target, learning_rate, IH_wts, HO_wts)
            Q, Output = feedforward(Input, IH_wts, HO_wts)
        Q = Q[0, 0:int(s[0]/2)]

        for i in range(len(Q)):
            Q[i] = (Q[i]/np.sum(np.abs(Q)))*26 #normalization of the Hid

        n = np.random.rand(int(s[0]/2),)-0.5
        n = n*noise_level #noise *m
        Q = Q+n
        if objective_func == 'max':
            Delta = np.abs(Output-Target)
            Delta = Delta.max()
            print (Delta)

        elif objective_func == 'ED':
            # Delta = distance.euclidean(Output,Target)
            Delta = np.sum((Output-Target)**2)
    return {'Delta':Delta, 'IH_wts':IH_wts ,'HO_wts':HO_wts, 'Hid':Q, 'RHS':t1, 'Target':Target, 'Output':Output }



def HTRACX_learn(bipolar_seq, IH_wts, HO_wts, IH_wts1, HO_wts1, IH_wts2, HO_wts2, learning_rate_initial, tau_array, Modeltype, objective_func):
    s = np.shape(bipolar_seq)
    arr_size = np.shape(bipolar_seq[0])

    bias_node = np.array([-1])

    Hid, Hid1, Hid2 = (np.zeros((arr_size))for i in range(3))
    RHS1, RHS2 = (np.zeros((arr_size))for i in range(2))

    D, D1, D2 = (np.zeros((0))for i in range(3))
    Hid_array, Hid1_array, RHS1_array, Hid2_array, RHS2_array = (np.zeros((0,s[1]))for i in range(5))
    Output_arr,Output1_arr,Output2_arr = (np.zeros((0,2*s[1]))for i in range(3))
    Target_arr,Target1_arr,Target2_arr = (np.zeros((0,2*s[1]))for i in range(3))
    

    delta1 = 10
    delta2 = 10
    tau = tau_array[0]
    tau1 = tau_array[1]
    tau2 = tau_array[2]
    t = 0
    m = 0
    # k = 1/2.
    decay_rate=0
    while t+1 < s[0]:
        learning_rate = learning_rate_initial[0] / (1 + decay_rate * t)
        learning_rate1 = learning_rate_initial[1] / (1 + decay_rate * t)
        learning_rate2 = learning_rate_initial[2] / (1 + decay_rate * t)
        if t == 0:
            In_t2 = bipolar_seq[t]
        else:
            a = tanh_compress(delta)
            ctx=tau/(tau+a)
            inp=a/(tau+a)
            In_t2 = ctx*Hid+ inp*In_t1
            # In_t2 = In_t1
        In_t1 = bipolar_seq[t + 1]
        RHS0 = In_t1    
        Input = np.concatenate((In_t2,In_t1),axis = 0)
        Input = np.concatenate(([Input], [bias_node]), axis = 1)

        Hid, Output = feedforward(Input, IH_wts, HO_wts)

        Target = Input[0,0:2*s[1]]

        IH_wts, HO_wts = backprop(Input, Hid, Output, Target, learning_rate, IH_wts, HO_wts)
        
        Hid, Output = feedforward(Input, IH_wts, HO_wts)

        Hid = Hid[0, 0:s[1]]

        for i in range(len(Hid)):
            Hid[i] = (Hid[i]/np.sum(np.abs(Hid)))*26

        Hid_array = np.concatenate((Hid_array, [Hid]), axis = 0)
        n = np.random.rand(s[1],)-0.5
        n = n*m
        Hid = Hid+n

        if objective_func=='max':
            delta = np.abs(Output-Target)
            delta = delta.max()
        elif objective_func=='ED':
            delta = distance.euclidean(Output,Target)
            # delta = np.sum((Output-Target)**2)/np.size(Output)

        D = np.concatenate((D,[delta]),axis = 0)

        Output_arr = np.concatenate((Output_arr,Output),axis = 0)
        Target_arr = np.concatenate((Target_arr,[Target]),axis = 0)

        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta
        #TRACX(t, delta_lower, delta_self, Hid_lower, Hid_self, RHS_lower, RHS_self, IH_wts, HO_wts, learning_rate, tau, noise_level, learning, learning_rule='max'):
        H1 = TRACX(t, delta_lower, delta1, Hid, Hid1, RHS0, RHS1, IH_wts1, HO_wts1, learning_rate1, tau1, 0, True, 'ED')
        Hid1 = H1['Hid']
        IH_wts1 = H1['IH_wts']
        HO_wts1 = H1['HO_wts']
        RHS1 = H1['RHS']
        delta1 = H1['Delta']
        Hid1_array = np.concatenate((Hid1_array, [Hid1]), axis = 0)
        RHS1_array = np.concatenate((RHS1_array, [H1['RHS']]), axis = 0)
        D1 = np.concatenate((D1,[H1['Delta']]),axis = 0)
        if t>0:
            Output1_arr = np.concatenate((Output1_arr,H1['Output']),axis = 0)
            Target1_arr = np.concatenate((Target1_arr,[H1['Target']]),axis = 0)
        
        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta1
        H2 = TRACX(t, delta_lower, delta2, Hid1, Hid2, RHS1, RHS2, IH_wts2, HO_wts2, learning_rate2, tau2, 0, True, 'ED')
        Hid2 = H2['Hid']
        IH_wts2 = H2['IH_wts']
        HO_wts2 = H2['HO_wts']
        RHS2 = H2['RHS']
        delta2 = H2['Delta']
        Hid2_array = np.concatenate((Hid2_array, [Hid2]), axis = 0)
        RHS2_array = np.concatenate((RHS2_array, [H2['RHS']]), axis = 0)
        D2 = np.concatenate((D2,[H2['Delta']]),axis = 0)
        if t>0:
            Output2_arr = np.concatenate((Output2_arr,H2['Output']),axis = 0)
            Target2_arr = np.concatenate((Target2_arr,[H2['Target']]),axis = 0)
        

        t = t+1


    D = D[1:t]
    D1 = D1[1:t]
    D2 = D2[1:t]
    Hid_array = Hid_array[1:t]
    Hid1_array = Hid1_array[1:t]
    Hid2_array = Hid2_array[1:t]

    return {'IH_wts':IH_wts, 'HO_wts':HO_wts , 'IH_wts1':IH_wts1, 'HO_wts1':HO_wts1, 'IH_wts2':IH_wts2, 'HO_wts2':HO_wts2, 'D':D, 'D1':D1, 'D2':D2, 'Hid':Hid_array, 'Hid1':Hid1_array, 'Hid2':Hid2_array, 'RHS1': RHS1_array,\
     'RHS2':RHS2_array, 'Output':Output_arr, 'Output1':Output1_arr,'Output2':Output2_arr, 'Target':Target_arr, 'Target1':Target1_arr,'Target2':Target2_arr} 

# In[5]:

def HTRACX_test(seq, IH_wts, HO_wts, IH_wts1, HO_wts1, IH_wts2, HO_wts2, tau_array, Modeltype, objective_func ):
    bipolar_seq = seq
    s = np.shape(bipolar_seq)
    arr_size = np.shape(bipolar_seq[0])

    bias_node = np.array([-1])
    
    Hid, Hid1, Hid2 = (np.zeros((arr_size))for i in range(3))
    RHS1, RHS2 = (np.zeros((arr_size))for i in range(2))

    D, D1, D2 = (np.zeros((0))for i in range(3))
    P= np.zeros((0,2))
    P1= np.zeros((0,2))
    P2= np.zeros((0,2))

    Hid_array, Hid1_array, RHS1_array, Hid2_array, RHS2_array = (np.zeros((0,s[1]))for i in range(5))
    Output_arr,Output1_arr,Output2_arr = (np.zeros((0,2*s[1]))for i in range(3))
    Target_arr,Target1_arr,Target2_arr = (np.zeros((0,2*s[1]))for i in range(3))
    

    delta1 = 10
    delta2 = 10
    tau = tau_array[0]
    tau1 = tau_array[1]
    tau2 = tau_array[2]
    a = 0
    m = 0
    # k = 1/2.
    t = 0
    p=np.array([0,0])
    while t+1 < s[0]:
        if t == 0:
            In_t2 = bipolar_seq[t]
        else:
            a = tanh_compress(delta)
            ctx=tau/(tau+a)
            inp=a/(tau+a)
            In_t2 = ctx*Hid + inp*In_t1
            p=np.array([ctx,inp])
            # In_t2 = tau*(1-a)*Hid + (1-tau)*a*In_t1
            
            
        P = np.concatenate((P,[p]),axis= 0 )
        In_t1 = bipolar_seq[t + 1]  
        RHS0 = In_t1
        Input = np.concatenate((In_t2,In_t1),axis = 0)
        Input = np.concatenate(([Input], [bias_node]), axis = 1)

        Hid, Output = feedforward(Input, IH_wts, HO_wts)

        Target = Input[0,0:2*s[1]]
        
        Hid = Hid[0, 0:s[1]]
        for i in range(len(Hid)):
            Hid[i] = (Hid[i]/np.sum(np.abs(Hid)))*26
        Hid_array = np.concatenate((Hid_array, [Hid]), axis = 0)
        n = np.random.rand(s[1],)-0.5
        n = n*m
        Hid = Hid+n
        if objective_func=='max':
            delta = np.abs(Output-Target)
            delta = delta.max()
        elif objective_func=='ED':
            delta = distance.euclidean(Output,Target)
            # delta = np.sum((Output-Target)**2)/np.size(Output)

        D = np.concatenate((D,[delta]),axis = 0)
        Output_arr = np.concatenate((Output_arr,Output),axis = 0)
        Target_arr = np.concatenate((Target_arr,[Target]),axis = 0)

        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta
        #TRACX_nolearn(t, delta_lower, delta_self, Hid_lower, Hid_self, RHS_lower, RHS_self, IH_wts, HO_wts, tau)
        H1 = TRACX(t, delta_lower, delta1, Hid, Hid1, RHS0, RHS1, IH_wts1, HO_wts1, 0, tau1, 0, False, 'ED')
        Hid1 = H1['Hid']
        RHS1 = H1['RHS']
        delta1 = H1['Delta']
        Hid1_array = np.concatenate((Hid1_array, [H1['Hid']]), axis = 0)
        RHS1_array = np.concatenate((RHS1_array, [H1['RHS']]), axis = 0)
        D1 = np.concatenate((D1,[H1['Delta']]),axis = 0)
        if t>0:
            Output1_arr = np.concatenate((Output1_arr,H1['Output']),axis = 0)
            Target1_arr = np.concatenate((Target1_arr,[H1['Target']]),axis = 0)
        
        if Modeltype == 'Hid_only':
            delta_lower = 0
        else:
            delta_lower=delta1

        H2 = TRACX(t, delta_lower, delta2, Hid1, Hid2, RHS1, RHS2, IH_wts2, HO_wts2, 0, tau2, 0, False, 'ED')
        Hid2 = H2['Hid']
        RHS2 = H2['RHS'] 
        delta2 = H2['Delta']
        Hid2_array = np.concatenate((Hid2_array, [H2['Hid']]), axis = 0)
        RHS2_array = np.concatenate((RHS2_array, [H2['RHS']]), axis = 0)
        D2 = np.concatenate((D2,[H2['Delta']]),axis = 0)
        if t>0:
            Output2_arr = np.concatenate((Output2_arr,H2['Output']),axis = 0)
            Target2_arr = np.concatenate((Target2_arr,[H2['Target']]),axis = 0)

        
        t = t+1

    D = D[1:t] #D_array start from the 3rd input
    D1 = D1[1:t]
    D2 = D2[1:t]
    Hid_array = Hid_array[1:t] #Hid_array start from the 3rd input
    Hid1_array = Hid1_array[1:t]
    Hid2_array = Hid2_array[1:t]
    return {'D':D, 'D1':D1, 'D2':D2, 'Hid':Hid_array, 'Hid1':Hid1_array, 'Hid2':Hid2_array, 'RHS1': RHS1_array, 'RHS2':RHS2_array, \
    'Output':Output_arr, 'Output1':Output1_arr,'Output2':Output2_arr, 'Target':Target_arr, 'Target1':Target1_arr,'Target2':Target2_arr} 

