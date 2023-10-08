from Model import *
from utils import *
import pickle
import torch
from Layers import *
from torch import optim
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--rnn-length', type=int, default='20',
                    help='rnn length')
parser.add_argument('--wantdate', type=int, default='7',
                    help='date to predict')
parser.add_argument('--hidn-rnn', type=int, default='78',
                    help='rnn hidden nodes')
parser.add_argument('--heads-att', type=int, default='2',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='39',
                    help='attention hidden nodes')
parser.add_argument('--dropout', type=float, default='0.3',
                    help='dropout rate')


def load_dataset(device1):
    with open('./data/CSI100E/x_num_standard.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('./data/CSI100E/y_1.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/CSI100E/x_newtext.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)
    with open('./data/CSI100E/edge_new.pkl', 'rb') as handle:
        edge_list=pickle.load(handle)
    with open('./data/CSI100E/interactive.pkl', 'rb') as handle:##the information of executives working in the company
        interactive_metric=pickle.load(handle)

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=device1)
    x.to(torch.double)
    x_sentiment = torch.tensor(stock_sentiments, device=device1)
    x_sentiment.to(torch.double)
    y = torch.tensor(y_load, device=device1).squeeze()
    y = (y>0).to(torch.long)
    inter_metric=torch.tensor(interactive_metric,device=device1)
    inter_metric=inter_metric.squeeze(2)
    inter_metric=inter_metric.transpose(0, 1)
    return x, y, x_sentiment,edge_list,inter_metric

def evaluate(model,wantdate, x_eval, x_sentiment_eval, y_eval,edge_list,device1):
    model.eval()
    seq_len = len(x_eval)
    #print('seq_len=',seq_len)#seq_len= 70
    seq = list(range(seq_len))[rnn_length:]
    #print('seq=',seq)#seq= [20, 21, 22, 23, 24-- 69]
    t=seq_len - rnn_length - wantdate
    #print('t',t)
    i=seq[t]

    output= model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], edge_list,inter_metric,device1)
    #此output即所需
    output = output.detach().cpu()
    output=np.exp(output.numpy())


    # output = output.detach().cpu()
    # preds.append(np.exp(output.numpy()))
    # trues.append(y_eval[i][:73].cpu().numpy())
    # #print('preds=',np.array(preds).shape)#tpreds= (50, 73, 2)
    # #print('trues=',np.array(trues).shape)#trues= (50, 73)
    # acc, auc = metrics(trues, preds)
    return output#换成preds的值



if __name__ == '__main__':
    args = parser.parse_args()
    device1 = "cuda:" + args.device
    device1=device1
    print(device1)
    criterion = torch.nn.NLLLoss()
    set_seed(1021)
    # load dataset
    print("loading dataset")
    x, y, x_sentiment,edge_list,inter_metric = load_dataset(device1)
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    hidn_rnn = args.hidn_rnn
    N_heads = args.heads_att
    hidn_att= args.hidn_att
    rnn_length = args.rnn_length
    t_mix = 1

    wantdate = args.wantdate

    #train-valid-test split
    x_train = x[: -100]
    x_eval = x[-100 - rnn_length : -50]
    x_test = x[-50 - rnn_length:]

    y_train = y[: -100]
    y_eval = y[-100 - rnn_length : -50]
    y_test = y[-50 - rnn_length:]

    x_sentiment_train = x_sentiment[: -100]
    x_sentiment_eval = x_sentiment[-100 - rnn_length : -50]
    x_sentiment_test = x_sentiment[-50 - rnn_length:]

    model = GraphCNN(num_stock=NUM_STOCK, d_market = D_MARKET,d_news= D_NEWS,out_c=2,
                      d_hidden = D_MARKET*2, hidn_rnn = hidn_rnn, hid_c= hidn_att, n_heads=N_heads,dropout = args.dropout,t_mix = t_mix)
    model.load_state_dict(torch.load('./SavedModels/evalauc0.5938_acc0.5715_testauc0.5659_acc0.5362.txt'))
    model.cuda(device=device1)
    model.to(torch.double)

    output = evaluate(model,wantdate, x_test, x_sentiment_test, y_test,edge_list,device1)#i设置输入*****
    print('end ',output)


