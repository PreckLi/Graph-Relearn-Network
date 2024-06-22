import numpy as np
from torch.nn import functional as F
from sklearn.cluster import SpectralClustering


def train(model, optimizer, data, epoch):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    acc, preds = accuracy(output[data.train_mask], data.y[data.train_mask])
    print(f'epoch:{epoch}, loss:{loss}, acc:{acc}')
    return loss


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels), preds.cpu()


def test(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    acc, preds = accuracy(output[data.test_mask], data.y[data.test_mask])
    print(f'loss:{loss} , test acc:{acc}')
    return acc, preds


def eval(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    acc, _ = accuracy(output[data.val_mask], data.y[data.val_mask])
    print(f'loss:{loss} , val acc:{acc}')
    return loss, acc


def class_pred(model, data, set):
    model.eval()
    output = model(data.x, data.edge_index)
    if set == "tr":
        _, preds = accuracy(output[data.train_mask], data.y[data.train_mask])
    if set == "va":
        _, preds = accuracy(output[data.val_mask], data.y[data.val_mask])
    if set == "te":
        _, preds = accuracy(output[data.test_mask], data.y[data.test_mask])
    return preds


def train_predictor(predictor, optimizer, data, x, epoch, changable_edge_index, invs, changable_mask):
    predictor.train()
    optimizer.zero_grad()
    output = predictor.forward(x, changable_edge_index)
    loss = F.nll_loss(output[invs], data.y[changable_mask])
    predictor_acc, _ = accuracy(output[invs], data.y[changable_mask])
    print("predictor acc:", predictor_acc)
    loss.backward()
    optimizer.step()
    print(f'epoch:{epoch}, predictor loss:{loss}')
    return loss.detach()


def change_test(array, last_k, method):
    changable_list = list()
    for i in range(len(array)):
        temp_list = list()
        for epoch in range(array.shape[-1] - last_k, array.shape[-1]):
            if array[i][epoch - 1] == array[i][epoch]:
                temp_list.append(0)
            else:
                temp_list.append(1)
        changable_list.append(temp_list)
    if method == "class":
        count0 = 0
        for i in changable_list:
            for j in i:
                if j == 0:
                    count0 += 1
        result = count0 / (len(array) * last_k)
    if method == "node":
        model = SpectralClustering(n_clusters=2)
        model.fit(changable_list)
        cluster = model.labels_
        result = np.sum(cluster == 0)/cluster.size
        # count_stable = 0
        # for i in changable_list:
        #     if i.count(1) == 0:
        #         count_stable += 1
        # result = count_stable / len(array)

    return result
