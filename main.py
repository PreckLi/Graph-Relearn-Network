import argparse
import time

import numpy as np
import torch
from torch.optim import Adam
from neighbor_supporter import Supporter, Supporter2, aggregate
from train_test import train, test, train_predictor, class_pred, eval, change_test
from utils import load_data, load_model, get_changable_nodes, get_neighs, get_logger, get_changable_nodes2, \
    negative_sample, compute_mad, compute_dirichlet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--train_ratio', type=float, default=0.6, help='train set ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set ratio')
parser.add_argument('--gnn_style', type=str, default='gcn', help="gnn style")
parser.add_argument('--hid_size', type=int, default=128, help="hidden size")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--k', type=float, default=0.2, help="attention ratio on main loss")
parser.add_argument('--split', type=str, default='random', help="split")
parser.add_argument('--gat_heads', type=int, default=8, help="gat heads")
parser.add_argument('--trans_heads', type=int, default=4, help="transformer conv heads")
parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay")
parser.add_argument('--epochs', type=int, default=200, help="training epochs")
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--khops', type=int, default=2, help='k-hop neighs')
parser.add_argument('--up', type=int, default=200, help='the upper epoch')
parser.add_argument('--area', type=int, default=100, help='the low epoch')
parser.add_argument('--use_predictor', type=bool, default=False, help='use predictor')
parser.add_argument('--sample_ratio', type=float, default=0.6, help="sample ratio")
parser.add_argument('--sample_style', type=str, default='random', help="gnn style? degree, random, walk")
args = parser.parse_args()


def main():
    logger = get_logger(f'./exp_log/{args.gnn_style}_{args.dataset}_exp.log')
    logger.info('start training!')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    data = load_data(args)
    preds_result_list = list()
    test_pred_list = list()
    val_set_pred_list = list()
    acc_list = list()
    proportion_list = list()
    device = torch.device(args.device)

    for i in range(10):
        print(f'----------------------------------------range{i}-----------------------------------------------')
        min_loss = 1e10
        max_acc = 0
        patience = 0

        model = load_model(args, data)
        optimizer_1 = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        if args.use_predictor:
            # predictor = Predictor(data.num_node_features, args.hid_size, data.num_classes)
            predictor = Supporter2(data.num_node_features, args.hid_size, data.num_classes, args.dropout).to(device)
            optimizer_2 = Adam(predictor.parameters(), args.lr, weight_decay=args.weight_decay)
        model = model.to(device)
        data = load_data(args).to(device)
        if args.use_predictor:
            m = 0.0
            epoch = 0
            pred_duration = 1.5
            while True:
                if epoch < args.epochs - 1:
                    for epoch in range(int(m * args.up), int((m + 1.0) * args.up)):
                        loss = train(model, optimizer_1, data, epoch)
                        loss.backward()
                        optimizer_1.step()
                        acc_test, _ = test(model, data)
                        val_loss, val_acc = eval(model, data)
                        if val_acc > max_acc:
                            max_acc = val_acc
                            torch.save(model.state_dict(),
                                       f'state/latest_{args.gnn_style}_{args.dataset}_{args.use_predictor}.pth')
                            print("Model saved at epoch{}".format(epoch))
                        tr_pred = class_pred(model, data, "tr")
                        preds_result_list.append(tr_pred.cpu().numpy())
                        te_pred = class_pred(model, data, "te")
                        test_pred_list.append(te_pred.cpu().numpy())

                    pred_array = np.array(preds_result_list).T
                    changable_nodes = get_changable_nodes(pred_array, args, m)
                    print("Changable nodes:", changable_nodes)
                    sample_nodes = negative_sample(changable_nodes, data.edge_index, args)
                    neighs_list, changable_subsets, changable_edge_index, invs = get_neighs(sample_nodes, data.edge_index, args)
                    changable_edge_index = changable_edge_index.to(device)
                    changable_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
                    # changable_mask[changable_nodes] = True
                    changable_mask[sample_nodes] = True
                    # agg_x = aggregate(data.x, neighs_list)

                    for epoch in range(int((m + 1.0) * args.up), int((m + pred_duration) * args.up)):
                        loss = args.k * train(model, optimizer_1, data, epoch)
                        # pred_loss = train_predictor(predictor, optimizer_2, data, agg_x, epoch, changable_mask)
                        pred_loss = train_predictor(predictor, optimizer_2, data, data.x[changable_subsets], epoch,
                                                    changable_edge_index, invs, changable_mask)
                        loss += (1 - args.k) * pred_loss
                        loss.backward()
                        optimizer_1.step()
                        val_loss, val_acc = eval(model, data)
                        if val_acc > max_acc:
                            torch.save(model.state_dict(),
                                       f'state/latest_{args.gnn_style}_{args.dataset}_{args.use_predictor}.pth')
                            print("Model saved at epoch{}".format(epoch))
                            min_loss = loss
                            patience = 0
                        else:
                            patience += 1
                        if patience > args.patience:
                            break

                        acc_test, _ = test(model, data)
                        tr_pred = class_pred(model, data, 'tr')
                        preds_result_list.append(tr_pred.cpu().numpy())
                        va_pred = class_pred(model, data, 'va')
                        val_set_pred_list.append(va_pred.cpu().numpy())
                        te_pred = class_pred(model, data, 'te')
                        test_pred_list.append(te_pred.cpu().numpy())
                    m += pred_duration
                    if patience > args.patience:
                        break
                else:
                    break
            model.load_state_dict(torch.load(f'state/latest_{args.gnn_style}_{args.dataset}_{args.use_predictor}.pth'))
            # mad = compute_mad(model, data)
            d_e = compute_dirichlet(model, data)
            acc_test, _ = test(model, data)
        torch.save(test_pred_list, f"pred_files/{args.gnn_style}_test_pred.pth")
        if not args.use_predictor:
            for epoch in range(args.epochs + args.up):
                loss = train(model, optimizer_1, data, epoch)
                loss.backward()
                optimizer_1.step()
                acc_test, _ = test(model, data)
                va_pred = class_pred(model, data, 'va')
                val_set_pred_list.append(va_pred.cpu().numpy())
                te_pred = class_pred(model, data, 'te')
                test_pred_list.append(te_pred.cpu().numpy())
            # mad = compute_mad(model, data)
            d_e = compute_dirichlet(model, data)
        acc_list.append(acc_test.cpu().numpy())
        torch.save(test_pred_list, f"pred_files/{args.gnn_style}_test_pred.pth")
        val_pred_array = np.array(val_set_pred_list).T
        proportion = change_test(val_pred_array, 50, "node")
        proportion_list.append(proportion)
        print(f"Stable nodes proportion:{proportion}")
    acc_mean = round(np.mean(acc_list), 4)
    # acc_max = round(max(acc_list), 4)
    # acc_min = round(min(acc_list), 4)
    acc_std = round(np.std(acc_list), 4)

    print('acc_list:{}'.format(acc_list))
    print('acc_mean:{}'.format(acc_mean))
    # print('acc_max:{}'.format(acc_max))
    # print('acc_min:{}'.format(acc_min))
    print('acc_std:{}'.format(acc_std))
    pro_mean = round(np.mean(proportion_list), 4)
    logger.info(
        f'====================mean:{acc_mean}, std:{acc_std}, pro_mean:{pro_mean}====================')


if __name__ == '__main__':
    since = time.time()
    main()
    print(f'cost time:{time.time() - since}')
    print('---------args-----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------start!----------\n')
