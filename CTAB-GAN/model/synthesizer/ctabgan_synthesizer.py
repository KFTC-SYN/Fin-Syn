import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from tqdm import tqdm


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
            seq += [Linear(dim, 1),Sigmoid()]
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

# def apply_activate(data, output_info):
#     data_t = []
#     st = 0
#     for item in output_info:
#         if item[1] == 'tanh':
#             ed = st + item[0]
#             data_t.append(torch.tanh(data[:, st:ed]))
#             st = ed
#         elif item[1] == 'softmax':
#             ed = st + item[0]
#             data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
#             st = ed
#     return torch.cat(data_t, dim=1)


def apply_activate(data, output_info):
    # 입력 데이터 안정성 체크
    if torch.isnan(data).any() or torch.isinf(data).any():
        # NaN/inf를 0으로 대체
        data = torch.where(torch.isnan(data) | torch.isinf(data), torch.zeros_like(data), data)
    
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            # tanh 입력 클리핑 (수치 안정성)
            tanh_input = torch.clamp(data[:, st:ed], min=-10.0, max=10.0)
            activated = torch.tanh(tanh_input)
            # NaN 체크
            if torch.isnan(activated).any() or torch.isinf(activated).any():
                activated = torch.where(torch.isnan(activated) | torch.isinf(activated), 
                                       torch.zeros_like(activated), activated)
            data_t.append(activated)
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            # gumbel_softmax 입력 클리핑 및 tau 값 증가 (수치 안정성)
            softmax_input = torch.clamp(data[:, st:ed], min=-10.0, max=10.0)
            activated = F.gumbel_softmax(softmax_input, tau=0.5)  # tau를 0.2에서 0.5로 증가
            # NaN 체크
            if torch.isnan(activated).any() or torch.isinf(activated).any():
                activated = torch.where(torch.isnan(activated) | torch.isinf(activated), 
                                       torch.zeros_like(activated), activated)
            data_t.append(activated)
            st = ed
    result = torch.cat(data_t, dim=1)
    
    # 최종 결과 안정성 체크
    if torch.isnan(result).any() or torch.isinf(result).any():
        result = torch.where(torch.isnan(result) | torch.isinf(result), 
                            torch.zeros_like(result), result)
    
    return result

def get_st_ed(target_col_index,output_info):
    st = 0
    c= 0
    tc= 0
    for item in output_info:
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
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
        opt1prime = random_choice_prob_index(self.p[idx]) 
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
      
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
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
    # 2025.11.24
    # assert side >= 4 and side <= 32
    assert side >= 4 and side <= 512, f"side must be between 4 and 512, got {side}"

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    print()
    layers_D += [

        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), 
        Sigmoid() 
    ]
    
    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    # 2025.11.24
    # assert side >= 4 and side <= 32
    assert side >= 4 and side <= 512, f"side must be between 4 and 512, got {side}"

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    return layers_G


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

class CTABGANSynthesizer:
    def __init__(self,
                 lr=2e-4,
                 class_dim=(256, 256, 256, 256),
                 random_dim=128,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=1024,
                 epochs=1,
                 device=torch.device("cpu")):
                 

        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, type={}, no_train=False):
        print("Fit started.")
        problem_type = None
        target_index=None
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed)
        self.transformer.fit() 
        
        train_data = self.transformer.transform(train_data.values)
        
        # 변환된 데이터 안정성 체크 및 float32 호환성 보장
        if np.isnan(train_data).any() or np.isinf(train_data).any():
            nan_count = np.isnan(train_data).sum() + np.isinf(train_data).sum()
            print(f"Warning: Transformed train_data contains NaN/inf ({nan_count} values). Replacing with zeros.")
            train_data = np.where(np.isnan(train_data) | np.isinf(train_data), 0.0, train_data)
        
        # float32 범위로 클리핑 (변환 시 NaN/inf 방지)
        train_data = np.clip(train_data, -3.4e38, 3.4e38)  # float32 범위
        
        # float32로 변환하여 NaN 발생 여부 확인
        train_data_f32 = train_data.astype('float32')
        if np.isnan(train_data_f32).any() or np.isinf(train_data_f32).any():
            nan_count = np.isnan(train_data_f32).sum() + np.isinf(train_data_f32).sum()
            print(f"Warning: train_data after float32 conversion contains NaN/inf ({nan_count} values). Replacing with zeros.")
            train_data_f32 = np.where(np.isnan(train_data_f32) | np.isinf(train_data_f32), 0.0, train_data_f32)
            train_data = train_data_f32.astype('float64')  # 다시 float64로 변환 (Sampler는 float64 사용)
        else:
            train_data = train_data_f32.astype('float64')  # float64로 변환 (Sampler는 float64 사용)
        
        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        
        print(f"data_dim: {data_dim}")
        self.cond_generator = Cond(train_data, self.transformer.output_info)
        		
        # 2025.11.24
        # sides = [4, 8, 16, 24, 32]
        sides = [4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512]
        col_size_d = data_dim + self.cond_generator.n_opt
        print(f"##### col_size_d: {col_size_d}")
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        
        # 2025.11.24
        # sides = [4, 8, 16, 24, 32]
        sides = [4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512]
        col_size_g = data_dim
        print(f"##### col_size_g: {col_size_g}")
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
        
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
        
        # 초기화 후 파라미터 안정성 체크
        for name, param in discriminator.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: Discriminator parameter {name} contains NaN/inf after initialization!")
                # NaN/inf 파라미터를 0으로 초기화
                param.data = torch.where(torch.isnan(param.data) | torch.isinf(param.data), 
                                        torch.zeros_like(param.data), param.data)
        
        for name, param in self.generator.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: Generator parameter {name} contains NaN/inf after initialization!")
                # NaN/inf 파라미터를 0으로 초기화
                param.data = torch.where(torch.isnan(param.data) | torch.isinf(param.data), 
                                        torch.zeros_like(param.data), param.data)

        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)
        
        
        if no_train: return

        print("Training started.")
        
        for i in range(self.epochs):
            # for _ in range(steps_per_epoch):
                
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample_train(self.batch_size)

            c, m, col, opt = condvec
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            perm = np.arange(self.batch_size)
            np.random.shuffle(perm)
            real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
            c_perm = c[perm]
                
            # numpy 배열에서 NaN/inf 체크 및 대체 (torch 변환 전)
            if np.isnan(real).any() or np.isinf(real).any():
                nan_count = np.isnan(real).sum() + np.isinf(real).sum()
                print(f"Warning: real data contains NaN/inf at epoch {i} ({nan_count} values). Replacing with zeros.")
                # NaN/inf를 0으로 대체
                real = np.where(np.isnan(real) | np.isinf(real), 0.0, real)
            
            # float32 범위로 클리핑 (변환 시 NaN/inf 방지)
            # float32의 실제 범위: 약 -3.4e38 ~ 3.4e38, 하지만 안전하게 더 작은 범위 사용
            real = np.clip(real, -1e20, 1e20)
            
            # float32로 변환하여 NaN 발생 여부 확인
            real_f32 = real.astype('float32')
            if np.isnan(real_f32).any() or np.isinf(real_f32).any():
                nan_count = np.isnan(real_f32).sum() + np.isinf(real_f32).sum()
                print(f"Warning: real data after float32 conversion contains NaN/inf at epoch {i} ({nan_count} values). Replacing with zeros.")
                real_f32 = np.where(np.isnan(real_f32) | np.isinf(real_f32), 0.0, real_f32)
            
            # torch.tensor()를 사용하여 복사본 생성 (torch.from_numpy()는 메모리 공유로 인한 문제 가능)
            real_tensor = torch.tensor(real_f32, dtype=torch.float32, device=self.device)
            
            # torch 변환 후 추가 안정성 체크
            if torch.isnan(real_tensor).any() or torch.isinf(real_tensor).any():
                # 안전하게 NaN/inf 개수 계산 (int64 오버플로우 방지)
                nan_mask = torch.isnan(real_tensor) | torch.isinf(real_tensor)
                nan_count = nan_mask.sum().item()
                # 비정상적으로 큰 값이면 shape 정보만 출력
                if nan_count > real_tensor.numel():
                    print(f"Warning: real data (after torch conversion) contains NaN/inf at epoch {i} (shape: {real_tensor.shape}). Replacing with zeros.")
                else:
                    print(f"Warning: real data (after torch conversion) contains NaN/inf at epoch {i} ({nan_count} values). Replacing with zeros.")
                real_tensor = torch.where(nan_mask, torch.zeros_like(real_tensor), real_tensor)
            
            real = real_tensor
                
            fake = self.generator(noisez)
            
            # Generator 출력값 안정성 체크
            if torch.isnan(fake).any() or torch.isinf(fake).any():
                print(f"Warning: Generator output contains NaN/inf at epoch {i}. Skipping update.")
                continue
            
            faket = self.Gtransformer.inverse_transform(fake)
            
            # faket 안정성 체크
            if torch.isnan(faket).any() or torch.isinf(faket).any():
                print(f"Warning: faket contains NaN/inf at epoch {i}. Skipping update.")
                continue
            
            fakeact = apply_activate(faket, self.transformer.output_info)
            
            # fakeact 안정성 체크
            if torch.isnan(fakeact).any() or torch.isinf(fakeact).any():
                print(f"Warning: fakeact contains NaN/inf at epoch {i}. Skipping update.")
                continue
                
            fake_cat = torch.cat([fakeact, c], dim=1)
            real_cat = torch.cat([real, c_perm], dim=1)
            
            # cat 데이터 안정성 체크
            if torch.isnan(fake_cat).any() or torch.isinf(fake_cat).any() or \
               torch.isnan(real_cat).any() or torch.isinf(real_cat).any():
                print(f"Warning: cat data (fake_cat/real_cat) contains NaN/inf at epoch {i}. Skipping update.")
                continue
                
            real_cat_d = self.Dtransformer.transform(real_cat)
            fake_cat_d = self.Dtransformer.transform(fake_cat)
            
            # transform 후 데이터 안정성 체크
            if torch.isnan(real_cat_d).any() or torch.isinf(real_cat_d).any():
                nan_count = torch.isnan(real_cat_d).sum().item() + torch.isinf(real_cat_d).sum().item()
                print(f"Warning: real_cat_d contains NaN/inf at epoch {i} ({nan_count} values). Replacing with zeros.")
                real_cat_d = torch.where(torch.isnan(real_cat_d) | torch.isinf(real_cat_d), 
                                         torch.zeros_like(real_cat_d), real_cat_d)
            
            if torch.isnan(fake_cat_d).any() or torch.isinf(fake_cat_d).any():
                nan_count = torch.isnan(fake_cat_d).sum().item() + torch.isinf(fake_cat_d).sum().item()
                print(f"Warning: fake_cat_d contains NaN/inf at epoch {i} ({nan_count} values). Replacing with zeros.")
                fake_cat_d = torch.where(torch.isnan(fake_cat_d) | torch.isinf(fake_cat_d), 
                                         torch.zeros_like(fake_cat_d), fake_cat_d)
                
            optimizerD.zero_grad()
            y_real,_ = discriminator(real_cat_d)
            y_fake,_ = discriminator(fake_cat_d)
            
            # Discriminator 출력값 안정성 체크 (NaN/inf 방지)
            if torch.isnan(y_real).any() or torch.isinf(y_real).any() or \
               torch.isnan(y_fake).any() or torch.isinf(y_fake).any():
                print(f"Warning: Discriminator output contains NaN/inf at epoch {i}. Skipping update.")
                continue
            
            # 출력값 클리핑 (0~1 범위로 제한, 수치 안정성 확보)
            y_real = torch.clamp(y_real, min=1e-6, max=1.0 - 1e-6)
            y_fake = torch.clamp(y_fake, min=1e-6, max=1.0 - 1e-6)
            
            loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
            
            # Loss NaN 체크 (계산 후)
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print(f"Warning: Discriminator loss is NaN/inf at epoch {i}. Skipping update.")
                continue
            
            loss_d.backward()
            
            # Discriminator에도 Gradient Clipping 추가 (강화: max_norm=0.5)
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
            
            # Generator 출력값 안정성 체크
            if torch.isnan(fake).any() or torch.isinf(fake).any():
                print(f"Warning: Generator output contains NaN/inf at epoch {i}. Skipping update.")
                continue
            
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, self.transformer.output_info)
            
            # fakeact 안정성 체크
            if torch.isnan(fakeact).any() or torch.isinf(fakeact).any():
                print(f"Warning: fakeact contains NaN/inf at epoch {i}. Skipping update.")
                continue

            fake_cat = torch.cat([fakeact, c], dim=1) 
            fake_cat = self.Dtransformer.transform(fake_cat)
                
            y_fake,info_fake = discriminator(fake_cat)
            
            # Discriminator 출력값 안정성 체크
            if torch.isnan(y_fake).any() or torch.isinf(y_fake).any() or \
               torch.isnan(info_fake).any() or torch.isinf(info_fake).any():
                print(f"Warning: Discriminator output (G) contains NaN/inf at epoch {i}. Skipping update.")
                continue
            
            cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
            
            # Cross entropy 안정성 체크
            if torch.isnan(cross_entropy) or torch.isinf(cross_entropy):
                print(f"Warning: Cross entropy is NaN/inf at epoch {i}. Skipping update.")
                continue

            _,info_real = discriminator(real_cat_d)
            
            # info_real 안정성 체크
            if torch.isnan(info_real).any() or torch.isinf(info_real).any():
                print(f"Warning: info_real contains NaN/inf at epoch {i}. Skipping update.")
                continue
            
            # y_fake 클리핑 (수치 안정성)
            y_fake = torch.clamp(y_fake, min=1e-6, max=1.0 - 1e-6)
            
            g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
            
            # Generator loss 안정성 체크
            if torch.isnan(g) or torch.isinf(g):
                print(f"Warning: Generator loss is NaN/inf at epoch {i}. Skipping update.")
                continue
            
            g.backward(retain_graph=True)
            
            # 2025.12.17 Info Loss 정규화 (차원 독립적)
            # loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
            # loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
            # loss_info = loss_mean + loss_std 
            info_fake_flat = info_fake.view(self.batch_size, -1)
            info_real_flat = info_real.view(self.batch_size, -1)
            info_dim = info_fake_flat.shape[1]
            
            # info_dim이 0이면 스킵
            if info_dim == 0:
                print(f"Warning: info_dim is 0 at epoch {i}. Skipping info loss.")
            else:
                mean_diff = torch.mean(info_fake_flat, dim=0) - torch.mean(info_real_flat, dim=0)
                
                # std 계산 시 batch_size가 1이면 unbiased=False 사용 (수치 안정성)
                std_fake = torch.std(info_fake_flat, dim=0, unbiased=(self.batch_size > 1))
                std_real = torch.std(info_real_flat, dim=0, unbiased=(self.batch_size > 1))
                std_diff = torch.abs(std_fake - std_real)  # 절댓값 사용

                # NaN 체크 및 클리핑
                mean_diff = torch.clamp(mean_diff, min=-10.0, max=10.0)
                std_diff = torch.clamp(std_diff, min=0.0, max=10.0)
                
                # NaN/inf 체크
                if torch.isnan(mean_diff).any() or torch.isinf(mean_diff).any() or \
                   torch.isnan(std_diff).any() or torch.isinf(std_diff).any():
                    print(f"Warning: Info loss components contain NaN/inf at epoch {i}. Skipping info loss.")
                else:
                    loss_mean = torch.norm(mean_diff, 1) / info_dim
                    loss_std = torch.norm(std_diff, 1) / info_dim
                    
                    # Loss 값 안정성 체크
                    if torch.isnan(loss_mean) or torch.isinf(loss_mean) or \
                       torch.isnan(loss_std) or torch.isinf(loss_std):
                        print(f"Warning: Info loss (mean/std) is NaN/inf at epoch {i}. Skipping info loss.")
                    else:
                        # 가중치 적용 (Info Loss의 영향력 조절)
                        info_weight = 0.1
                        loss_info = (loss_mean + loss_std) * info_weight
                        
                        # 최종 loss_info 안정성 체크
                        if torch.isnan(loss_info) or torch.isinf(loss_info):
                            print(f"Warning: Info loss is NaN/inf at epoch {i}. Skipping info loss.")
                        else:
                            loss_info.backward()
            
            # 2025.12.18 Gradient Clipping 강화 (max_norm=0.5로 축소하여 발산 방지)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            
            optimizerG.step()

            if (i + 1) % 10 == 0:
                # 2025.11.28
                # print(f"Step: {i}/{self.epochs} Loss: {loss_mean:.4f}")
                print(f"Step: {i+1}/{self.epochs} Loss_mean (norm): {loss_mean:.4f} Loss_std (norm): {loss_std:.4f} Loss_info: {loss_info:.4f}")

            if problem_type:
                fake = self.generator(noisez)
                
                # Generator 출력값 안정성 체크
                if torch.isnan(fake).any() or torch.isinf(fake).any():
                    print(f"Warning: Generator output (classifier) contains NaN/inf at epoch {i}. Skipping classifier update.")
                else:
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.output_info)
                    
                    # fakeact 안정성 체크
                    if torch.isnan(fakeact).any() or torch.isinf(fakeact).any():
                        print(f"Warning: fakeact (classifier) contains NaN/inf at epoch {i}. Skipping classifier update.")
                    else:
                        real_pre, real_label = classifier(real)
                        fake_pre, fake_label = classifier(fakeact)
                        
                        # Classifier 출력값 안정성 체크
                        if torch.isnan(real_pre).any() or torch.isinf(real_pre).any() or \
                           torch.isnan(fake_pre).any() or torch.isinf(fake_pre).any():
                            print(f"Warning: Classifier output contains NaN/inf at epoch {i}. Skipping classifier update.")
                        else:
                            c_loss = CrossEntropyLoss() 
                            
                            if (st_ed[1] - st_ed[0])==1:
                                c_loss= SmoothL1Loss()
                                real_label = real_label.type_as(real_pre)
                                fake_label = fake_label.type_as(fake_pre)
                                real_label = torch.reshape(real_label,real_pre.size())
                                fake_label = torch.reshape(fake_label,fake_pre.size())
                                
                            
                            elif (st_ed[1] - st_ed[0])==2:
                                c_loss = BCELoss()
                                real_label = real_label.type_as(real_pre)
                                fake_label = fake_label.type_as(fake_pre)

                            loss_cc = c_loss(real_pre, real_label)
                            loss_cg = c_loss(fake_pre, fake_label)
                            
                            # Loss 안정성 체크
                            if torch.isnan(loss_cc) or torch.isinf(loss_cc) or \
                               torch.isnan(loss_cg) or torch.isinf(loss_cg):
                                print(f"Warning: Classifier loss contains NaN/inf at epoch {i}. Skipping classifier update.")
                            else:
                                optimizerG.zero_grad()
                                loss_cg.backward()
                                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
                                optimizerG.step()

                                optimizerC.zero_grad()
                                loss_cc.backward()
                                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=0.5)
                                optimizerC.step()
                                
    @torch.no_grad()
    def sample(self, n, seed=0):

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
            # print(len(data))
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result = self.transformer.inverse_transform(data)
        
        return result[0:n]

