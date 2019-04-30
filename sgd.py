import numpy as np
import sys

# def normalize(datafile):
#     max = []
#     min = []

    
#     for i in range(len(datafile[0])):
#         max.append(0)
#         min.append(0)
        
#     for j in range(len(datafile)):
#         for k in range(len(datafile[0])-1):
#             if (datafile[j][k] > max[k]) :
#                 max[k] = datafile[j][k]
#             if (datafile[j][k] < min[k]) :
#                 min[k] = datafile[j][k]

#     for i in range(len(datafile)):
#         for j in range(len(datafile[0])-1):
#             if (max[j] - min[j] != 0):
#                 datafile[i][j] = (datafile[i][j] - min[j])/(max[j] - min[j])
                
#     return datafile
#################
### Read data ###

f = open(sys.argv[1])
data = np.loadtxt(f)
data_s = data.copy() 
train = data[:,1:]
# train = normalize(train)

trainlabels = data[:,0]
###append ones at the end of train data
onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

# print("train=",train)
# print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
# test = normalize(test)

testlabels = data[:,0]
###append ones at the end of test data
onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]
# print(np.shape(test))
# print(np.shape(train))
# exit()
###Set a size for minibatch and create a random row number array
mini_batch_size = int(sys.argv[3])
# mini_batch_array = np.random.randint(0,len(train),mini_batch_size)

###build minibatch train data
np.random.shuffle(data_s)
train_s = data_s[:mini_batch_size,1:]
onearray = np.ones((train_s.shape[0],1))
train_s = np.append(train_s,onearray,axis=1)

# print(np.shape(train_s))
# exit()
# print(train_s)
# exit()

trainlabel_s = data_s[:mini_batch_size,0]
# print(np.shape(trainlabel_s))
# exit()

# print(mini_batch_array)
# exit()

#hidden_nodes = int(sys.argv[3])

hidden_nodes = 3

##############################
### Initialize all weights ###

#w = np.random.rand(1,hidden_nodes)
w = np.random.rand(hidden_nodes)
# print("w=",w)
# print(np.shape(w))
# exit()

#check this command
#W = np.zeros((hidden_nodes, cols), dtype=float)
# W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)
# print("W=",W)
# print(np.shape(W))
# exit()

epochs = 200
eta = 0.001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
hidden_layer_mb = np.matmul(train_s, np.transpose(W))


sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
hidden_layer_mb = np.array([sigmoid(xi) for xi in hidden_layer_mb])
# print("hidden_layer=",hidden_layer)
# print("hidden_layer=",hidden_layer_mb)
# exit()
# print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
# print("output_layer=",output_layer)

# obj = np.sum(np.square(output_layer - trainlabels))
# print("obj=",obj)

# obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))

# print("Obj=",obj)
obj = np.sum(np.square(output_layer - trainlabels))

###############################
### Begin gradient descent ####
# print(train)
# print((train_s))
# exit()
while(prevobj - obj > 0.001 or i < epochs ):
#while(prevobj - obj > 0):

        #Update previous objective
        prevobj = obj

        #Calculate gradient update for final layer (w)
        #dellw is the same dimension as w
        for k in range(rows):
                dellw = (np.dot(hidden_layer_mb[0,:],w)-trainlabel_s[0])*hidden_layer_mb[0,:]
                for j in range(1, len(train_s)):
                        dellw += (np.dot(hidden_layer_mb[j,:],np.transpose(w))-trainlabel_s[j])*hidden_layer_mb[j,:]

                #Update w
                w = w - eta*dellw

        #       print("dellf=",dellf)
                
                #Calculate gradient update for hidden layer weights (W)
                #dellW has to be of same dimension as W

                #Let's first calculate dells. After that we do dellf and dellv.
                #Here s, u, and v are the three hidden nodes
                #dells = df/dz1 * (dz1/ds1, dz1,ds2)
                dells = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabel_s[0])*w[0] * (hidden_layer_mb[0,0])*(1-hidden_layer_mb[0,0])*train_s[0]
                dellu = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabel_s[0])*w[1] * (hidden_layer_mb[0,1])*(1-hidden_layer_mb[0,1])*train_s[0]
                dellv = np.sum(np.dot(hidden_layer_mb[0,:],w)-trainlabel_s[0])*w[2] * (hidden_layer_mb[0,2])*(1-hidden_layer_mb[0,2])*train_s[0]
                for j in range(1, len(train_s)):
                        dells += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabel_s[j])*w[0] * (hidden_layer_mb[j,0])*(1-hidden_layer_mb[j,0])*train_s[j]
                        dellu += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabel_s[j])*w[1] * (hidden_layer_mb[j,1])*(1-hidden_layer_mb[j,1])*train_s[j]
                        dellv += np.sum(np.dot(hidden_layer_mb[j,:],w)-trainlabel_s[j])*w[2] * (hidden_layer_mb[j,2])*(1-hidden_layer_mb[j,2])*train_s[j]

                # exit()

                #TODO: Put dells, dellu, and dellv as rows of dellW
                dellW = np.array([dells, dellu, dellv])

                #Update W
                W = W - eta*dellW
        # print(W)



        #Recalculate objective
        hidden_layer = np.matmul(train, np.transpose(W))
        # print("hidden_layer=",hidden_layer)

        hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
        # print("hidden_layer=",hidden_layer)

        output_layer = (np.matmul(hidden_layer, np.transpose(w)))
        # print("output_layer=",output_layer)

        obj = np.sum(np.square(output_layer - trainlabels))
        print("obj=",obj)
        # print("prevobj-obj =", prevobj - obj,i)
        i = i + 1
        print(i)
        # print("Objective=",obj)
        # hidden_layer_mb = np.matmul(train_s, np.transpose(W))
        # hidden_layer_mb = np.array([sigmoid(xi) for xi in hidden_layer_mb])
       
        np.random.shuffle(data_s)
        train_s = data_s[0:mini_batch_size,1:].copy()
        onearray = np.ones((train_s.shape[0],1))
        train_s = np.append(train_s,onearray,axis=1)
        trainlabel_s = data_s[0:mini_batch_size,0].copy()
        # print(trainlabel_s)

       
# predictions = np.sign(np.dot([[11,11,1]], np.transpose(w)))
hidden_layer = np.matmul(test, np.transpose(W))
predictions_bo = (np.matmul(sigmoid(hidden_layer),np.transpose(w)))

predictions = np.sign(predictions_bo)
print(predictions)
# print(w)