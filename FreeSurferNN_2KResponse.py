# -*- coding: utf-8 -*-
"""
@author: Cedric Beaulac
File created on 28/08/21
Postdoc Project #1
Feature Extraction Technique #1
Using FreeSurfer output as NN classifier input
"""


####################################
# Import packages
####################################

from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from ignite.metrics import Accuracy
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn, optim
from torch.nn import functional as F

# Importing the clinical data for diagnostic and the predictor data
# NN trained on ADNI1 for genetic purposes (sub-cohort that only contains DAT and NC)

# Load all FreeFreesurfer extracted statistics. These are the predictors for the neural network.
Predictors = pd.read_csv(r"Data_02.csv", index_col=0)

# Load a subset of 56 features selected by experts. These are a smaller set of predictors for the neural network.
SPredictors = pd.read_csv(r"ExpertFeatures_01 .csv", index_col=0)

# Load clinical data which contains the response variable (AD Diagnosis)
Clinical = pd.read_csv(r"ClinicalInfo.csv")

# Filter clinical data to only include records present in predictors. This ensures that we have predictor and response data for the same set of records.
ClinicalID = Clinical[Clinical["RID"].isin(np.array(Predictors.index, dtype=int))]

# Set RID as index for filtered clinical data. This makes it easier to join with predictors data.
ClinicalID = ClinicalID.set_index("RID")

# Filter clinical data to only include baseline visit records. This is because we're interested in predicting diagnosis based on baseline visit data.
ClinicalF = ClinicalID[ClinicalID["VISCODE"] == "bl"]

# Create response variable by selecting the 'AUX.STRATIFICATION' column and filtering to only include specific categories.
Response = ClinicalF["AUX.STRATIFICATION"]
Response = Response.loc[
    (Response == "sDAT")
    | (Response == "sNC")
    | (Response == "eDAT")
    | (Response == "uNC")
]

# Recode response variable to have simpler categories.
Response.loc[(Response == "sDAT") | (Response == "eDAT")] = "DAT"
Response.loc[(Response == "sNC") | (Response == "uNC")] = "NC"

# Join predictors with response to create final dataset. This dataset will be used for training the neural network.
PandaData = Predictors.join(Response, how="right")

# Do the same for the smaller set of predictors.
PandaSData = SPredictors.join(Response, how="right")

####################################
# Arguments
####################################

parser = argparse.ArgumentParser(
    description="Freesurfer feature extraction (Cedric Beaulac)"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=100,
    metavar="N",
    help="input batch size for training (default: 50)",
)
parser.add_argument(
    "--xdim",
    type=int,
    default=PandaData.shape[1] - 1,
    metavar="N",
    help="dimension of the predictor",
)
parser.add_argument(
    "--n", type=int, default=PandaData.shape[0], metavar="N", help="number of subjects"
)
parser.add_argument(
    "--ntr", type=int, default=150, metavar="N", help="number of training subjects"
)
parser.add_argument(
    "--nval", type=int, default=50, metavar="N", help="number of training subjects"
)
parser.add_argument("--nc", type=int, default=2, metavar="N", help="number of class")
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--nMC",
    type=int,
    default=50,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
args = parser.parse_args()

# torch.manual_seed(args.seed)

device = torch.device("cpu")  # should be CUDA when running on the big powerfull server


###################################
# Define the NN Classifier
####################################
class NeuralNetwork(nn.Module):
    def __init__(self, feature_count):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(args.xdim, 750)
        self.layer2 = nn.Linear(750, feature_count)
        self.layer3 = nn.Linear(feature_count, args.nc)
        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = F.log_softmax(self.layer3(x), dim=1)
        return output, x


####################################
# Set up Data set
####################################

# Numpy/tensor data

npData = np.array(PandaData)
c, v = np.unique(npData[:, -1], return_inverse=True)
npData[:, -1] = v
npData = npData.astype(float)

perm = np.random.permutation(npData.shape[0])
Data = torch.tensor(npData[perm, :])

npSData = np.array(PandaSData)
c, v = np.unique(npSData[:, -1], return_inverse=True)
npSData[:, -1] = v
npSData = npSData.astype(float)

SData = npSData[perm, :]

# Prepare torch data and normalize predictors
TrainData = Data[0 : args.ntr]
TrainData[:, 0:-1] = torch.nn.functional.normalize(TrainData[:, 0:-1])
ValData = Data[args.ntr : (args.ntr + args.nval)]
ValData[:, 0:-1] = torch.nn.functional.normalize(ValData[:, 0:-1])
TestData = Data[(args.ntr + args.nval) :]
TestData[:, 0:-1] = torch.nn.functional.normalize(TestData[:, 0:-1])

# Divide small data set as well
TrainSData = SData[0 : (args.ntr + args.nval)]
TestSData = SData[(args.ntr + args.nval) :]

####################################
# Define the training procedure
####################################
# Define data loaders
train_loader = torch.utils.data.DataLoader(TrainData, batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(ValData)
test_loader = torch.utils.data.DataLoader(TestData)


# training function
def train(epoch, optimizer):
    model.train()
    train_loss = 0
    for id, data in enumerate(train_loader):
        data = data.to(device)
        inputs, target = data[:, 0:-1], data[:, -1].type(torch.LongTensor)
        # print(target[0:10])
        # target = torch.cat((0*torch.ones(125),torch.ones(125))).type(torch.LongTensor)
        output, f = model(inputs.float())
        # print(output[1:10,:])
        loss = F.nll_loss(output, target)
        loss.backward()
        # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,loss ))
        optimizer.step()


# accuracy function for assesment
def accuracy(args, model, data):
    inputs, target = data[:, 0:-1], data[:, -1].type(torch.LongTensor)
    output, f = model(inputs.float())
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = correct / data.shape[0]
    return acc


# accuracy function for assesment
def AUC(args, model, data):
    inputs, target = data[:, 0:-1], data[:, -1].type(torch.LongTensor)
    output, f = model(inputs.float())
    pred = output.argmax(dim=1, keepdim=True)
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    auc = metrics.auc(fpr, tpr)
    return auc


# return pred for CV
def predictions(args, model, data):
    inputs, target = data[:, 0:-1], data[:, -1].type(torch.LongTensor)
    output, f = model(inputs.float())
    pred = output.argmax(dim=1, keepdim=True)
    return pred


# initialize model parameters
def model_init(args, model, std):
    torch.nn.init.normal_(model.fc1.weight, 0, std)
    torch.nn.init.normal_(model.fc3.weight, 0, std)
    return model


####################################
# Model Training with Cross validated parameters
####################################
perm = np.random.permutation(npData.shape[0])
Data = torch.tensor(npData[perm, :])
TrainData = Data[0 : (args.ntr + args.nval)]
TrainData[:, 0:-1] = torch.nn.functional.normalize(TrainData[:, 0:-1])
TestData = Data[(args.ntr + args.nval) :]
TestData[:, 0:-1] = torch.nn.functional.normalize(TestData[:, 0:-1])
model = NN1(f=56)
model = model_init(args, model, 5)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
for epoch in range(1, 750 + 1):
    train(epoch, optimizer)
    acc = accuracy(args, model, TestData)
    print("====> Test Data Accuracy:{:.4f}".format(acc))

Accuracy = accuracy(args, model, TestData)


# Processing all of the data through the NN to extract the features
Response = ClinicalF["AUX.STRATIFICATION"]
# PandaData is the data containing our data set with all the features
PandaData = Predictors.join(Response, how="right")
model.eval()
Features = model(
    torch.nn.functional.normalize(torch.tensor(PandaData.iloc[:, 0:-1].values)).type(
        torch.FloatTensor
    )
)[1]
FeaturesData = pd.DataFrame(Features.detach().numpy(), index=PandaData.index)
# Save the features
FeaturesData.to_csv("FreeSurfer+NN_Features_2k_CV.csv")
# Save the NN
torch.save(model.state_dict(), "model")


####################################
# Rigourous comparison between NN features AND 56 expert-selected features
####################################

# Define vectors to store the results
NNAccuracy = np.zeros(args.nMC)
NNAUC = np.zeros(args.nMC)

LRAccuracy = np.zeros(args.nMC)
LRAUC = np.zeros(args.nMC)

for i in range(0, args.nMC):
    # Random permutation and setting all the needed data (Comments in previous sections)
    perm = np.random.permutation(npData.shape[0])
    Data = torch.tensor(npData[perm, :])
    TrainData = Data[0 : (args.ntr + args.nval)]
    TrainData[:, 0:-1] = torch.nn.functional.normalize(TrainData[:, 0:-1])
    ValData = Data[args.ntr : (args.ntr + args.nval)]
    ValData[:, 0:-1] = torch.nn.functional.normalize(ValData[:, 0:-1])
    TestData = Data[(args.ntr + args.nval) :]
    TestData[:, 0:-1] = torch.nn.functional.normalize(TestData[:, 0:-1])
    SData = npSData[perm, :]
    TrainSData = SData[0 : (args.ntr + args.nval)]
    TestSData = SData[(args.ntr + args.nval) :]
    # Training NN model
    model = NN1(f=56)
    model = model_init(args, model, 5)
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    for epoch in range(1, 750 + 1):
        train(epoch, optimizer)
        acc = accuracy(args, model, TestData)
        # print('====> Test Data Accuracy:{:.4f}'.format(acc))
    Accuracy = accuracy(args, model, TestData)
    NNAccuracy[i] = Accuracy
    auc = AUC(args, model, TestData)
    NNAUC[i] = auc
    # Training Logistic Regression
    lrm = LogisticRegression(penalty="l1", solver="saga", max_iter=10000).fit(
        TrainSData[:, 0:-1], TrainSData[:, -1]
    )
    pred = lrm.predict(TestSData[:, 0:-1])
    correct = np.sum(pred == TestSData[:, -1])
    acc = correct / TestSData.shape[0]
    LRAccuracy[i] = acc
    fpr, tpr, thresholds = metrics.roc_curve(TestSData[:, -1], pred)
    LRAUC[i] = metrics.auc(fpr, tpr)
    print(
        "====> MC:{:.0f}, NN Accu:{:.5f}, LR Accu:{:.5f}".format(i + 1, Accuracy, acc)
    )
