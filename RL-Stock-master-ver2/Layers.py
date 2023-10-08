from utils import *
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from torch.nn import  Dropout



class Graph_Linear(nn.Module):##the linear layer
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):
        super(Graph_Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        #print(self.W.shape)#torch.Size([73, 78, 234])
        #print(x.shape)#torch.Size([73, 10])
        output = torch.bmm(x.unsqueeze(1), self.W)
        output = output.squeeze(1)
        if self.bias:
            output = output + self.b
        return output

#################################################################################
class Graph_Linear1(nn.Module):##the linear layer
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):
        super(Graph_Linear1, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        #print(self.W.shape)#torch.Size([73, 10, 117])
        #print(x.shape)#torch.Size([39])
        output = torch.bmm(x.unsqueeze(1), self.W)
        output = output.squeeze(1)
        if self.bias:
            output = output + self.b
        return output
########################################################################

class Fuse_inlinear(nn.Module):
    def __init__(self,num_nodes,input_size=1):
        super(Fuse_inlinear,self).__init__()
        self.ww=nn.Parameter(torch.rand(num_nodes,input_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self,x,y):
        output1=torch.mul(x,self.ww)+torch.mul(y,1-self.ww)
        return output1



class Graph_Tensor(nn.Module):##feature fusion for historical price and financial news
    def __init__(self, num_stock, d_hidden, d_market, d_news, bias=True):
        super(Graph_Tensor, self).__init__()
        self.num_stock = num_stock
        self.d_hidden = d_hidden
        self.d_market = d_market
        self.d_news = d_news
        self.seq_transformation_news = nn.Conv1d(d_news, d_hidden, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_markets = nn.Conv1d(d_market, d_hidden, kernel_size=1, stride=1, bias=False)
        self.tensorGraph = nn.Parameter(torch.zeros(num_stock, d_hidden, d_hidden, d_hidden))
        self.W = nn.Parameter(torch.zeros(num_stock, 2 * d_hidden, d_hidden))
        self.b = nn.Parameter(torch.zeros(num_stock, d_hidden))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, market, news):
        t, num_stocks = news.size()[0], news.size()[1]

        news_transformed = news.reshape(-1, self.d_news)
        news_transformed = torch.transpose(news_transformed, 0, 1).unsqueeze(0)
        news_transformed = self.seq_transformation_news(news_transformed)
        news_transformed = news_transformed.squeeze().transpose(0, 1)
        news_transformed = news_transformed.reshape(t, num_stocks, self.d_hidden)

        market_transformed = market.reshape(-1, self.d_market)
        market_transformed = torch.transpose(market_transformed, 0, 1).unsqueeze(0)
        market_transformed = self.seq_transformation_markets(market_transformed)
        market_transformed = market_transformed.squeeze().transpose(0, 1)
        market_transformed = market_transformed.reshape(t, num_stocks, self.d_hidden)

        x_news_tensor = news_transformed.unsqueeze(2)
        x_news_tensor = x_news_tensor.unsqueeze(2)
        x_market_tensor = market_transformed.unsqueeze(-1)
        temp_tensor = x_news_tensor.matmul(self.tensorGraph).squeeze()
        temp_tensor = temp_tensor.matmul(x_market_tensor).squeeze()
        x_linear = torch.cat((news_transformed, market_transformed), axis=-1)
        temp_linear = torch.bmm(x_linear.transpose(0, 1), self.W)
        temp_linear = temp_linear.transpose(0, 1)

        output = torch.tanh(temp_tensor + temp_linear + self.b)
        return output

class Graph_GRUCell(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size##78
        self.bias = bias
        self.x2h = Graph_Linear(num_nodes, input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(num_nodes, hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):
        #print(self.hidden_size)
        #print(x.shape)##[73, 10]

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        #print("gate_x ", gate_x.shape,"gate_h ", gate_h.shape)#gate_x  torch.Size([73, 234]) gate_h  torch.Size([73, 234])
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        #print("gate_x ", gate_x.shape,"gate_h ", gate_h.shape)##gate_x  torch.Size([73, 234]) gate_h  torch.Size([73, 234])
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        #print("i_r",i_r.shape," h_r",h_r.shape)##i_r torch.Size([73, 78])  h_r torch.Size([73, 78])
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        #print("r",resetgate.shape," in",inputgate.shape," new",newgate.shape)##r torch.Size([73, 78])  in torch.Size([73, 78])  new torch.Size([73, 78])
        hy = newgate + inputgate * (hidden - newgate)
        #print("hy",hy.shape)##hy torch.Size([73, 78])
        return hy

class Graph_GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)

        """ for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)#++++++++++++++++++++++ """
        return hidden

############################################################
class Graph_GRUCell1(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_GRUCell1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Graph_Linear1(num_nodes, input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear1(num_nodes, hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):
        #print(self.input_size)##[73,39]!!!!!!!!!!!!
        # print(self.hidden_size)#39
        #print(hidden.shape)#torch.Size([39, 78])
        gate_x = self.x2h(x)#gate_x  torch.Size([73, 117])
        gate_h = self.h2h(hidden)
        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()

        #print("gate_x ", gate_x.shape)##gate_x  torch.Size([73, 117])
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        #print("i_r",i_r.shape," h_r",h_r.shape)##i_r torch.Size([73, 39])  h_r torch.Size([73, 39])

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        #print("r",resetgate.shape," in",inputgate.shape," new",newgate.shape)##r torch.Size([73, 78])  in torch.Size([73, 78])  new torch.Size([73, 78])
        hy = newgate + inputgate * (hidden - newgate)
        #print("hy",hy.shape)##hy torch.Size([73, 78])
        return hy

class Graph_GRUModel1(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel1, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell1 = Graph_GRUCell1(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        x=x.unsqueeze(0)
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell1(x[seq], hidden)#x_c torch.Size([73, 39])->7339才对
            #print(hidden.shape)#torch.Size([73, 39])
        return hidden
########################################################

class Graph_LSTMCell(nn.Module):###LSTMcell++++++++++++++++++++++++++++++
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Graph_Linear(num_nodes, input_size, 2 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(num_nodes, hidden_size, 2 * hidden_size, bias=bias)
        self.c2h = Graph_Linear(num_nodes, hidden_size, 2 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):##hidden=hc???
        h,c=hidden
        #c=hidden
        gate_x = self.x2h(x)
        gate_h = self.h2h(h)
        gate_c = self.c2h(c)
        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()
        #gate_c = gate_c.squeeze()
        #print("x",x.shape,"gate_x ", gate_x.shape,"gate_h ", gate_h.shape)##x torch.Size([73, 10]) gate_x  torch.Size([73, 156]) gate_h  torch.Size([73, 156])

        combined = torch.cat([gate_x, gate_h], dim=1)
        #print("combined",combined.shape)##combined torch.Size([73, 312])

        #i_r, i_i, i_n, i_b = gate_x.chunk(4, 1)#chunk能够按照某维度，对张量进行均匀切分，并且返回结果是原张量的视图。
        #h_r, h_i, h_n, h_b = gate_h.chunk(4, 1)
        #C_r, h_i, h_n, h_b = gate_h.chunk(4, 1)
        #print("hiddensize=",self.hidden_size)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_size, dim=1) #hidden_size=78 312/78=4
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        #print("cc_i",cc_i.shape, " cc_f",cc_f.shape," cc_o",cc_o.shape," cc_g",cc_g.shape)##cc_i torch.Size([73, 78])  cc_f torch.Size([73, 78])  cc_o torch.Size([73, 78])  cc_g torch.Size([73, 78])
        #print("i",i.shape, " f",f.shape," o",o.shape," g",g.shape)##i torch.Size([73, 78])  f torch.Size([73, 78])  o torch.Size([73, 78])  g torch.Size([73, 78])

        #ft = torch.sigmoid(i_r + h_r)###ft
        #it = torch.sigmoid(i_i + h_i)###it
        #ot = torch.sigmoid(i_n + h_n)###ot
        #c_hat = torch.tanh(i_b + h_b)###c_hat
        #ct=ft * c + it *c_hat

        ct = f * c + i * g
        ctt = torch.sigmoid(ct)
        ht = o * ctt
        #print("ct",ct.shape,"ht",ht.shape)##ct torch.Size([73, 78]) ht torch.Size([73, 78])
        return ht, ct


class Graph_LSTMModel(nn.Module):###++++++++++++++++++++++++++++++
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = Graph_LSTMCell(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = {torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype),
            torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)}
        for seq in range(x.size(0)):
            hidden = self.lstm_cell(x[seq], hidden)
            #hidden1 = self.lstm_cell(x[seq], hidden)
        return hidden



class Graph_Attention(nn.Module):##Learning implicit relation

    def __init__(self, in_features, out_features, dropout, alpha ,alpha1,concat=True, residual=False):
        super(Graph_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.alpha1 = alpha1
        self.concat = concat
        self.residual = residual

        self.seq_transformation_r = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        self.seq_transformation_s = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.coef_revise = False
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def get_relation(self, input_r):
        num_stock = input_r.shape[0]
        seq_r = torch.transpose(input_r, 0, 1).unsqueeze(0)
        logits = torch.zeros(num_stock, num_stock, device=input_r.device, dtype=input_r.dtype)
        seq_fts_r = self.seq_transformation_r(seq_r)
        f_1 = self.f_1(seq_fts_r)
        f_2 = self.f_2(seq_fts_r)
        logits += (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        coefs = F.elu(logits)
        coefs=F.softmax(coefs,dim=1)
        abc=torch.zeros_like(coefs)
        coefs = torch.where(coefs < self.alpha1, abc, coefs)
        if not isinstance(self.coef_revise,torch.Tensor):
            self.coef_revise = torch.zeros(73, 73, device = input_r.device) + 1.0 - torch.eye(73, 73,device = input_r.device)#note that:if you want to run this code on CSI300E,you should change 73 to 185(the number of firm nodes)
        coefs_eye = coefs.mul(self.coef_revise)
        return coefs_eye


    def forward(self, input_r,c):
        # unmasked attention
        coefs_eye = self.get_relation(input_r)*c
        coef=coefs_eye.nonzero(as_tuple=False)
        return coef


##dual attention networks

###intra-class attention
class RSMPConv(MessagePassing):
    def __init__(self, in_hid, out_hid, 
                 num_edge_types,negative_slope=0.2,heads=1):
        super(RSMPConv, self).__init__(aggr='add')

        self.in_hid = in_hid
        self.out_hid = out_hid
        self.num_edge_types = num_edge_types
        self.negative_slope=negative_slope
        
        self.rel_wi=nn.Parameter(torch.Tensor(num_edge_types,out_hid*2,1))
        self.rel_bt=nn.Parameter(torch.Tensor(out_hid*2,1))
        # self.w_wi=nn.Linear(in_hid, out_hid, bias=False)
        self.w_bt=nn.Linear(out_hid,out_hid,bias=False)
        self.q_trans=nn.Parameter(torch.Tensor(out_hid,1))

        self.norm=nn.LayerNorm(out_hid)
        self.norm_list=nn.ModuleList()
        for i in range(num_edge_types):
            self.norm_list.append(nn.LayerNorm(out_hid))


        self.skip = nn.Parameter(torch.ones(1))
        self.beta_weight=nn.Parameter(torch.ones(1))
        self.overall_beta=nn.Parameter(torch.randn(num_edge_types))
        # self.drop=Dropout(0.2)

        glorot(self.rel_wi)
        glorot(self.rel_bt)
        glorot(self.q_trans)

    def forward(self, x, edge_idx, edge_type):
        # x=self.w_wi(x)

        out_list=[]
        edg_list=[]
        for i in range(self.num_edge_types):
            mask = (edge_type == i)
            edge_index = edge_idx[:, mask]
            if mask.sum() !=0:
                rs=self.w_bt(F.leaky_relu(self.norm_list[i](self.propagate(edge_index, x=x,edge_type=i)),self.negative_slope))   #Nxd       
                out_list+=[rs]
                edg_list+=[i]
        beta=[]
        for i in edg_list:
            beta+=[F.leaky_relu(out_list[i]@self.q_trans,self.negative_slope).sum(0)]
        overall_beta=F.softmax(torch.FloatTensor(beta),dim=0)
        res=0
        for i in edg_list:
            res+=out_list[i]*overall_beta[i]
        
        final_weight=torch.sigmoid(self.skip)
        res = self.norm(F.gelu(res) * (final_weight) + x* (1 - final_weight))

        return res


    def message(self,edge_index,x_i, x_j,edge_type):
        
        node_f = torch.cat((x_i, x_j), 1)                                       #nx2d

        temp = torch.matmul(node_f, self.rel_wi[edge_type]).to(x_i.device)      #nx1

        alpha=softmax(temp,edge_index[1])
        rs=x_j*alpha                                                            #nxd
        return rs

###inter-class attention
class HetGATConv(MessagePassing):
    def __init__(self, in_hid, out_hid, negative_slope=0.2,norm=True,dual=True,global_weight=True):
        super(HetGATConv, self).__init__(aggr='add',)

        self.in_hid = in_hid
        self.out_hid = out_hid
        self.negative_slope=negative_slope
        self.norm=norm
        self.dual=dual
        self.global_weight=global_weight

        
        self.rel_wi=nn.Parameter(torch.Tensor(2,out_hid*2,1))
        self.rel_bt=nn.Parameter(torch.Tensor(out_hid*2,1))
        self.w_bt=nn.Linear(out_hid,out_hid,bias=False)
        self.w_out=nn.Linear(out_hid,out_hid,bias=False)
        self.q_trans=nn.Parameter(torch.Tensor(out_hid,1))

        self.out_norm=nn.LayerNorm(out_hid)

        self.skip = nn.Parameter(torch.ones(1))
        # self.drop=Dropout(0.2)

        glorot(self.rel_wi)
        glorot(self.rel_bt)
        glorot(self.q_trans)
        

    def forward(self, c_hid,p_hid, edge_idx, edge_type):
        out_list=[]
        num_edge_types=2
        edg_list=[]
        for i in range(num_edge_types):
            mask = (edge_type == i)
            edge_index = edge_idx[:, mask]
            if mask.sum() !=0:
                rs=self.w_bt(F.leaky_relu(self.propagate(edge_index=edge_index, x=(c_hid,p_hid),edge_type=i),self.negative_slope))   #Nxd
                out_list+=[rs]
                edg_list+=[i]
        beta=[]
        for i in edg_list:
            beta+=[F.leaky_relu(out_list[i]@self.q_trans,self.negative_slope).sum(0)]
        overall_beta=F.softmax(torch.FloatTensor(beta),dim=0)
        res=0
        for i in range(len(edg_list)):
            res+=out_list[i]*overall_beta[i]
        final_weight=torch.sigmoid(self.skip)
        res = self.out_norm(F.gelu(res)* (final_weight) + p_hid* (1 - final_weight))
        res=F.gelu(res)* (final_weight) + p_hid* (1 - final_weight)

        return res


    def message(self,x_i, x_j,edge_index,edge_type):

        node_f = torch.cat((x_i, x_j), 1)                                       #nx2d

        temp = torch.matmul(node_f, self.rel_wi[edge_type]).to(x_i.device)      #nx1

        alpha=softmax(temp,edge_index[1])
        #
        rs=x_j*alpha                                                            #nxd
        return rs

class SMPLayer(nn.Module):###intra-class
    def __init__(self,in_hid,out_hid,num_m1,num_m2,n_heads=8,n_layers=2,dropout=0.4,hgt_layer=1,**kwargs):
        super(SMPLayer,self).__init__()
        self.hetgat=nn.ModuleList()
        self.layer=n_layers

        self.hgt=nn.ModuleList()
        self.norm=nn.LayerNorm(out_hid)
        self.drop=Dropout(dropout)
        self.proj_c=nn.Linear(in_hid,out_hid,bias=False)
        self.proj_p=nn.Linear(in_hid,out_hid,bias=False)

        for _ in range(hgt_layer):
            if _ == 0:
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m1,heads=n_heads))
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m2,heads=n_heads))
            else:
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m1,heads=n_heads))
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m2,heads=n_heads))

        for n in range(n_layers):
            self.hetgat.append(HetGATConv(out_hid, out_hid))
            self.hetgat.append(HetGATConv(out_hid, out_hid))
    #c_gh,p_gh:[x,edge_index,edge_type];t_gh[edge_index,edge_type]
    def forward(self,c_gh,p_gh,t_gh):
        h_c=c_gh[0]
        h_p=p_gh[0]
        #print("hc",h_c.shape," hp",h_p.shape)#hc torch.Size([73, 78])  hp torch.Size([163, 78])
        h_c=self.proj_c(h_c)#指向self.proj_c=nn.Linear(in_hid,out_hid,bias=False)
        h_p=self.proj_p(h_p)
        #print("hc1",h_c.shape," hp1",h_p.shape)#hc1 torch.Size([73, 39])  hp1 torch.Size([163, 39])
        """ edge_indx, edge_type = t_gh[0], t_gh[1]
        for ly in range(int(len(self.hetgat) / 2)):
            p_hid = self.hetgat[2 * ly](h_c, h_p, edge_indx, edge_type)

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            c_hid = self.hetgat[2 * ly + 1](h_p, h_c, edge_indx, edge_type)

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            h_c = c_hid
            h_p = p_hid  """
            #print("h_c",h_c.shape," h_p",h_p.shape)#h_c torch.Size([73, 39])  h_p torch.Size([163, 39])
            
        for hl in range(int((len(self.hgt)/2))):
            # if hl==0:
            h_c=self.hgt[2*hl](h_c,c_gh[1],c_gh[2])
            h_p=self.hgt[2*hl+1](h_p,p_gh[1],p_gh[2])

        h_c=self.drop(self.norm(h_c))
        h_p=self.drop(self.norm(h_p))
        #print("hc",h_c.shape," hp",h_p.shape)#hc torch.Size([73, 39])  hp torch.Size([163, 39])
        return h_c,h_p


class SMPLayer1(nn.Module):###inter-class
    def __init__(self,in_hid,out_hid,num_m1,num_m2,n_heads=8,n_layers=2,dropout=0.4,hgt_layer=1,**kwargs):
        super(SMPLayer1,self).__init__()
        self.hetgat=nn.ModuleList()
        self.layer=n_layers

        self.hgt=nn.ModuleList()
        self.norm=nn.LayerNorm(out_hid)
        self.drop=Dropout(dropout)
        self.proj_c=nn.Linear(in_hid,out_hid,bias=False)
        self.proj_p=nn.Linear(in_hid,out_hid,bias=False)

        for _ in range(hgt_layer):
            if _ == 0:
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m1,heads=n_heads))
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m2,heads=n_heads))
            else:
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m1,heads=n_heads))
                self.hgt.append(RSMPConv(out_hid, out_hid, num_m2,heads=n_heads))

        for n in range(n_layers):
            self.hetgat.append(HetGATConv(out_hid, out_hid))
            self.hetgat.append(HetGATConv(out_hid, out_hid))
    #c_gh,p_gh:[x,edge_index,edge_type];t_gh[edge_index,edge_type]
    def forward(self,c_gh,p_gh,t_gh,h_c,h_p):
        #h_c=c_gh[0]
        #h_p=p_gh[0]
        #h_c=self.proj_c(h_c)#指向self.proj_c=nn.Linear(in_hid,out_hid,bias=False)
        #h_p=self.proj_p(h_p)
        edge_indx, edge_type = t_gh[0], t_gh[1]
        for ly in range(int(len(self.hetgat) / 2)):
            p_hid = self.hetgat[2 * ly](h_c, h_p, edge_indx, edge_type)

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            c_hid = self.hetgat[2 * ly + 1](h_p, h_c, edge_indx, edge_type)

            edge_indx = torch.stack((edge_indx[1], edge_indx[0]))
            h_c = c_hid
            h_p = p_hid 
            
        """ for hl in range(int((len(self.hgt)/2))):
            # if hl==0:
            h_c=self.hgt[2*hl](h_c,c_gh[1],c_gh[2])
            h_p=self.hgt[2*hl+1](h_p,p_gh[1],p_gh[2]) """

        h_c=self.drop(self.norm(h_c))
        h_p=self.drop(self.norm(h_p))
        return h_c,h_p


class AttentionBlock(nn.Module):########+++++++++++++++++++++++++++++++++++++
    def __init__(self,time_step,dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        print(inputs)
        inputs_t = torch.transpose(inputs,0,1) # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = torch.nn.functional.softmax(attention_weight,dim=-1)
        attention_probs = torch.transpose(attention_probs,2,1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec#, attention_probs