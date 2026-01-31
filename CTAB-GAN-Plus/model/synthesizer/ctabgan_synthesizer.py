import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from model.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from tqdm import tqdm, trange
import time


class Classifier(Module):
    def __init__(self,input_dim, dis_dims,st_ed):
        super(Classifier,self).__init__()
        dim = input_dim-(st_ed[1]-st_ed[0])
        seq = []
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        
        if (st_ed[1]-st_ed[0])==1:
            seq += [Linear(dim, 1)]
        
        elif (st_ed[1]-st_ed[0])==2:
            # 2026.01.25: BCEWithLogitsLoss 사용을 위해 Sigmoid 제거 (수치 안정성 개선)
            # 기존 코드: seq += [Linear(dim, 1),Sigmoid()]
            seq += [Linear(dim, 1)]
        else:
            seq += [Linear(dim,(st_ed[1]-st_ed[0]))] 
        
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        label=None
        
        if (self.str_end[1]-self.str_end[0])==1:
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        if ((self.str_end[1]-self.str_end[0])==2) | ((self.str_end[1]-self.str_end[0])==1):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)

def get_st_ed(target_col_index,output_info):
    st = 0
    c= 0
    tc= 0

    for item in output_info:
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
            if item[2] == 'yes_g':
                c+=1
        elif item[1] == 'softmax':
            st += item[0]
            c+=1
        tc+=1    
    
    ed= st+output_info[tc][0] 

    return (st,ed)

def random_choice_prob_index_sampling(probs,col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval

class Cond(object):
    def __init__(self, data, output_info):
       
        self.model = []
        st = 0
        counter = 0
        for item in output_info:
           
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0  
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))  
        self.p_sampling = []
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax': 
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)  
                tmp_sampling = np.sum(data[:, st:ed], axis=0)     
                tmp = np.log(tmp + 1)  
                tmp = tmp / np.sum(tmp) 
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p[self.n_col, :item[0]] = tmp 
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
                
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        
        # 2026.01.21: 각 컬럼의 실제 옵션 수만큼만 사용하여 인덱스 선택 (CUDA device-side assert 오류 수정)
        # 기존 코드: opt1prime = random_choice_prob_index(self.p[idx]) 
        opt1prime = np.zeros(batch, dtype=np.int64)
        for i in np.arange(batch):
            col_idx = idx[i]
            num_options = self.interval[col_idx, 1]  # 해당 컬럼의 실제 옵션 수
            p_col = self.p[col_idx, :num_options]
            if p_col.sum() > 0:
                p_col_normalized = p_col / p_col.sum()
                opt1prime[i] = np.random.choice(num_options, p=p_col_normalized)
            else:
                opt1prime[i] = 0
            # opt1prime[i]가 num_options를 초과하지 않도록 명시적으로 제한
            opt1prime[i] = min(opt1prime[i], num_options - 1)
            
        # 2026.01.21: 인덱스 범위 검증 추가
        # 기존 코드: vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        for i in np.arange(batch):
            col_idx = idx[i]
            num_options = self.interval[col_idx, 1]
            start_idx = self.interval[col_idx, 0]
            # opt1prime[i]를 다시 한번 제한
            opt_val = min(max(opt1prime[i], 0), num_options - 1)
            target_idx = start_idx + opt_val
            # target_idx가 유효한 범위 내에 있는지 확인
            if 0 <= target_idx < self.n_opt:
                vec[i, target_idx] = 1
            else:
                # 범위를 벗어난 경우 첫 번째 옵션으로 대체
                vec[i, start_idx] = 1
            
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
      
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        # 2026.01.21: 인덱스 범위 검증 추가 (CUDA device-side assert 오류 수정)
        # 기존 코드: vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
        for i in np.arange(batch):
            col_idx = idx[i]
            num_options = self.interval[col_idx, 1]
            opt_val = min(opt1prime[i], num_options - 1)
            target_idx = self.interval[col_idx, 0] + opt_val
            # target_idx가 n_opt를 초과하지 않도록 보장
            if target_idx < self.n_opt:
                vec[i, target_idx] = 1
            
        return vec

def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            continue

        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]

class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
                
    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]

class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        info = len(layers)-2
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)

def determine_layers_disc(side, num_channels):
    # 2025.11.29
    # assert side >= 4 and side <= 64
    assert side >= 4 and side <= 96

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []
    num_c = num_channels
    num_s = side / 2
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c * 2
        num_s = num_s / 2

    layers_D = []

    for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            LayerNorm(ln),
            LeakyReLU(0.2, inplace=True),
        ]

    layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)] 

    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    # 2025.11.29
    # assert side >= 4 and side <= 64
    assert side >= 4 and side <= 96

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []

    num_c = num_channels * (2 ** (len(layer_dims) - 2))
    num_s = int(side / (2 ** (len(layer_dims) - 1)))
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c / 2
        num_s = num_s * 2

    layers_G = [ConvTranspose2d(random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)]

    for prev, curr, ln in zip(reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms):
        layers_G += [LayerNorm(ln), ReLU(True), ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)]
    return layers_G

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
    
    return res

def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer, device='cpu', lambda_=10):
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1,  device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    interpolates = transformer.transform(interpolates)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates,_ = netD(interpolates) 

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0] 
    
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_
    
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

class CTABGANSynthesizer:
    def __init__(self,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=150,
                 lr=2e-4,
                 device="cpu"):
                 

        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, general=[], non_categorical=[], type={}):

        problem_type = None
        target_index=None
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed, general_list=general, non_categorical_list=non_categorical)
        self.transformer.fit() 
        
        # # DEBUG: 범주형 변수 인코딩 확인
        # print("="*80)
        # print("[DEBUG] 범주형 변수 인코딩 검증")
        # print("="*80)
        # print(f"[DEBUG] categorical columns: {categorical}")
        # print(f"[DEBUG] mixed columns: {mixed}")
        # print(f"[DEBUG] train_data shape: {train_data.shape}")
        # print(f"[DEBUG] train_data columns: {list(train_data.columns)}")
        
        # # 각 범주형 컬럼의 고유값 확인
        # for col in categorical:
        #     if col in train_data.columns:
        #         unique_vals = train_data[col].unique()
        #         print(f"[DEBUG] {col}: {len(unique_vals)} unique values, "
        #               f"dtype={train_data[col].dtype}, "
        #               f"min={train_data[col].min()}, max={train_data[col].max()}")
        #         if len(unique_vals) <= 20:
        #             print(f"[DEBUG]   values: {sorted(unique_vals)}")
        #         # NaN 확인
        #         nan_count = train_data[col].isna().sum()
        #         if nan_count > 0:
        #             print(f"[DEBUG]   WARNING: {nan_count} NaN values found!")
        
        # # transformer output_info 확인
        # print(f"[DEBUG] transformer.output_info: {self.transformer.output_info[:10]}...")  # 처음 10개만
        # print(f"[DEBUG] transformer.output_dim: {self.transformer.output_dim}")
        # print("="*80)
        
        train_data = self.transformer.transform(train_data.values)
        
        # # DEBUG: 변환된 데이터 확인
        # print(f"[DEBUG] transformed train_data shape: {train_data.shape}")
        # print(f"[DEBUG] transformed train_data dtype: {train_data.dtype}")
        # print(f"[DEBUG] transformed train_data min: {train_data.min():.4f}, max: {train_data.max():.4f}")
        # print(f"[DEBUG] transformed train_data contains NaN: {np.isnan(train_data).any()}")
        # print(f"[DEBUG] transformed train_data contains Inf: {np.isinf(train_data).any()}")
        
        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        print(f"##### data_dim: {data_dim}")
        self.cond_generator = Cond(train_data, self.transformer.output_info)
        
        # # DEBUG: cond_generator 확인
        # print(f"[DEBUG] cond_generator.n_opt: {self.cond_generator.n_opt}")
        # print(f"[DEBUG] cond_generator.n_col: {self.cond_generator.n_col}")
        # if hasattr(self.cond_generator, 'interval'):
        #     print(f"[DEBUG] cond_generator.interval: {self.cond_generator.interval}")
        # print("="*80)
        		
        # 2025.11.29
        # sides = [4, 8, 16, 24, 32]
        sides = [4, 8, 16, 24, 32, 64, 96]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            print(f"i: {i}, i*i: {i*i}, col_size_d: {col_size_d}")
            if i * i >= col_size_d:
                self.dside = i
                break
        
        # 2025.11.29
        # sides = [4, 8, 16, 24, 32]
        sides = [4, 8, 16, 24, 32, 64, 96]
        col_size_g = data_dim
        for i in sides:
            print(f"i: {i}, i*i: {i*i}, col_size_g: {col_size_g}")
            if i * i >= col_size_g:
                self.gside = i
                break
        print(f"self.gside: {self.gside}, self.dside: {self.dside}")

        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        
        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)
        optimizer_params = dict(lr=self.lr, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier=None
        optimizerC= None
        if target_index != None:
            st_ed= get_st_ed(target_index,self.transformer.output_info)
            classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)
        
        
        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)
        
        epsilon = 0
        epoch = 0
        steps = 0
        ci = 1
        
        for i in tqdm(range(self.epochs)):
				
            
            for _ in range(ci):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                
                # # DEBUG: 조건부 벡터 검증
                # print(f"[DEBUG] c shape: {c.shape}, dtype: {c.dtype}")
                # print(f"[DEBUG] c min: {c.min():.4f}, max: {c.max():.4f}")
                # print(f"[DEBUG] c contains NaN: {np.isnan(c).any()}, Inf: {np.isinf(c).any()}")
                # if col is not None:
                #     print(f"[DEBUG] col shape: {col.shape}, min: {col.min()}, max: {col.max()}")
                # if opt is not None:
                #     print(f"[DEBUG] opt shape: {opt.shape}, min: {opt.min()}, max: {opt.max()}")
                
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                c_perm = c[perm]
                
                real = torch.from_numpy(real.astype('float32')).to(self.device)
                
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)
                
                fake_cat = torch.cat([fakeact, c], dim=1)
                real_cat = torch.cat([real, c_perm], dim=1)
                
                real_cat_d = self.Dtransformer.transform(real_cat)
                fake_cat_d = self.Dtransformer.transform(fake_cat)
                
                optimizerD.zero_grad()
                
                d_real,_ = discriminator(real_cat_d)
                

                d_real = -torch.mean(d_real)
                d_real.backward() 
                

                d_fake,_ = discriminator(fake_cat_d)
                
                d_fake = torch.mean(d_fake)

                d_fake.backward() 
                
                pen = calc_gradient_penalty_slerp(discriminator, real_cat, fake_cat,  self.Dtransformer , self.device)

                pen.backward()
                
                # 2026.01.25: Gradient Clipping 추가 (CTAB-GAN 참고, 수치 안정성 개선)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
            
                optimizerD.step()
                
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            
            condvec = self.cond_generator.sample_train(self.batch_size)

            c, m, col, opt = condvec
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

            optimizerG.zero_grad()

            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, self.transformer.output_info)

            fake_cat = torch.cat([fakeact, c], dim=1) 
            fake_cat = self.Dtransformer.transform(fake_cat)
                
            y_fake,info_fake = discriminator(fake_cat)
            
            cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)

            _,info_real = discriminator(real_cat_d)
            

            g = -torch.mean(y_fake) + cross_entropy
            
            # 2026.01.25: NaN 체크 및 Skip 처리 (CTAB-GAN 참고, 수치 안정성 개선)
            if torch.isnan(g) or torch.isinf(g):
                print(f"Warning: Generator loss (g) is NaN/inf at step {i+1}. Skipping update.")
                continue
                
            g.backward(retain_graph=True)
            loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
            loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
            loss_info = loss_mean + loss_std 

            # 2026.01.25: NaN 체크 및 Skip 처리 (CTAB-GAN 참고, 수치 안정성 개선)
            if torch.isnan(loss_info) or torch.isinf(loss_info):
                print(f"Warning: loss_info is NaN/inf at step {i+1}. Skipping update.")
                continue
                
            loss_info.backward()
            
            # 2026.01.25: Gradient Clipping 추가 (CTAB-GAN 참고, 수치 안정성 개선)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            
            optimizerG.step()

            # 2025.11.29
            if (i + 1) % 10 == 0:
                print(f"Step: {i+1}/{self.epochs} Loss_mean (norm): {loss_mean:.4f} Loss_std (norm): {loss_std:.4f} Loss_info: {loss_info:.4f}")

            if problem_type:
                        
                fake = self.generator(noisez)
                
                faket = self.Gtransformer.inverse_transform(fake)
                
                fakeact = apply_activate(faket, self.transformer.output_info)
                
                real_pre, real_label = classifier(real)
                fake_pre, fake_label = classifier(fakeact)
                    
                c_loss = CrossEntropyLoss() 
                
                if (st_ed[1] - st_ed[0])==1:
                    c_loss= SmoothL1Loss()
                    real_label = real_label.type_as(real_pre)
                    fake_label = fake_label.type_as(fake_pre)
                    real_label = torch.reshape(real_label,real_pre.size())
                    fake_label = torch.reshape(fake_label,fake_pre.size())
                    
                
                elif (st_ed[1] - st_ed[0])==2:
                    # 2026.01.25: BCELoss -> BCEWithLogitsLoss 변경 (수치 안정성 개선)
                    # 기존 코드: c_loss = BCELoss()
                    c_loss = BCEWithLogitsLoss()
                    real_label = real_label.type_as(real_pre)
                    fake_label = fake_label.type_as(fake_pre)

                loss_cc = c_loss(real_pre, real_label)
                loss_cg = c_loss(fake_pre, fake_label)
                
                # 2026.01.25: NaN 체크 및 Skip 처리 (CTAB-GAN 참고, 수치 안정성 개선)
                if torch.isnan(loss_cg) or torch.isinf(loss_cg) or torch.isnan(loss_cc) or torch.isinf(loss_cc):
                    print(f"Warning: Classifier loss is NaN/inf at step {i+1}. Skipping update.")
                    continue

                optimizerG.zero_grad()
                loss_cg.backward()
                # 2026.01.25: Gradient Clipping 추가 (CTAB-GAN 참고, 수치 안정성 개선)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
                optimizerG.step()

                optimizerC.zero_grad()
                loss_cc.backward()
                # 2026.01.25: Gradient Clipping 추가 (CTAB-GAN 참고, 수치 안정성 개선)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=0.5)
                optimizerC.step()
                            

            
   
    @torch.no_grad()
    def sample(self, n, seed=0):
        print(n)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        sample_batch_size = 8092
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // sample_batch_size + 1

        data = []
        
        for i in range(steps):
            noisez = torch.randn(sample_batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(sample_batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(sample_batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket,output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result,resample = self.transformer.inverse_transform(data)

        t0 = time.time()
        while len(result) < n and (time.time() - t0) <= 600:
            data_resample = []    
            steps_left = resample// sample_batch_size + 1
            # print(f"Sampling: {len(result)}/{n}")
            for i in range(steps_left):
                noisez = torch.randn(sample_batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(sample_batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(sample_batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                    
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            data_resample = np.concatenate(data_resample, axis=0)

            res,resample = self.transformer.inverse_transform(data_resample)
            result  = np.concatenate([result,res],axis=0)
        
        return result[0:n]

