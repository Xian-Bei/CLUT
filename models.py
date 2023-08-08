import torch.nn as nn
import torchvision.transforms as transforms
from utils.LUT import *
from ipdb import set_trace as S
import tinycudann as tcnn
from thop import profile
import trilinear


   
        
        
class CLUTNet(nn.Module): 
    def __init__(self, nsw, dim=33, backbone='Backbone', *args, **kwargs):
        super().__init__()
        self.TrilinearInterpolation = TrilinearInterpolation()
        self.pre = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = eval(backbone)()
        last_channel = self.backbone.last_channel
        self.classifier = nn.Sequential(
                nn.Conv2d(last_channel, 128,1,1),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Conv2d(128, int(nsw[:2]),1,1),
        )
        nsw = nsw.split("+")
        num, s, w = int(nsw[0]), int(nsw[1]), int(nsw[2])
        self.CLUTs = CLUT(num, dim, s, w)

    def fuse_basis_to_one(self, img, TVMN=None):
        mid_results = self.backbone(self.pre(img))
        weights = self.classifier(mid_results)[:,:,0,0] # n, num
        D3LUT, tvmn_loss = self.CLUTs(weights, TVMN)
        return D3LUT, tvmn_loss    

    def forward(self, img, img_org, TVMN=None):
        D3LUT, tvmn_loss = self.fuse_basis_to_one(img, TVMN)
        img_res = self.TrilinearInterpolation(D3LUT, img_org)
        return {
            "fakes": img_res + img_org,
            "3DLUT": D3LUT,
            "tvmn_loss": tvmn_loss,
        }

class CLUT(nn.Module):
    def __init__(self, num, dim=33, s="-1", w="-1", *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s,self.w = s,w = eval(str(s)), eval(str(w))
        # +: compressed;  -: uncompressed
        if s == -1 and w == -1: # standard 3DLUT
            self.mode = '--'
            self.LUTs = nn.Parameter(torch.zeros(num,3,dim,dim,dim))
        elif s != -1 and w == -1:  
            self.mode = '+-'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num*3*dim*dim))
        elif s == -1 and w != -1: 
            self.mode = '-+'
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(num*3*dim, w))

        else: # full-version CLUT
            self.mode = '++'
            self.s_Layers = nn.Parameter(torch.rand(dim, s)/5-0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim*dim)/5-0.1)
            self.LUTs = nn.Parameter(torch.zeros(s*num*3,w))
        print("n=%d s=%d w=%d"%(num, s, w), self.mode)

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        if self.mode == "--":
            D3LUTs = self.LUTs
        else:
            if self.mode == "+-":
                # d,s  x  s,num*3dd  -> d,num*3dd -> d,num*3,dd -> num,3,d,dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs).reshape(dim,num*3,dim*dim).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "-+":
                # num*3d,w x w,dd -> num*3d,dd -> num,3ddd
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(num,3,self.dim,self.dim,self.dim)
            if self.mode == "++":
                # s*num*3, w  x   w, dd -> s*num*3,dd -> s,num*3*dd -> d,num*3*dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs.mm(self.w_Layers).reshape(-1,num*3*dim*dim)).reshape(dim,num*3,dim**2).permute(1,0,2).reshape(num,3,self.dim,self.dim,self.dim)
            D3LUTs = cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weights, TVMN): # n,num
        dim = self.dim
        num = self.num

        D3LUTs = self.reconstruct_luts()
        if TVMN is None:
            tvmn_loss = 0
        else:
            tvmn_loss = TVMN(D3LUTs)
        D3LUT = weights.mm(D3LUTs.reshape(num,-1)).reshape(-1,3,dim,dim,dim)
        return D3LUT, tvmn_loss

    def forward(self, weights, TVMN=None):
        lut, tvmn_loss = self.combine(weights, TVMN)
        return lut, tvmn_loss


encoding_config= {
    "n_levels": 16,
    "otype": "HashGrid",
    "n_features_per_level": 2,
    "log2_hashmap_size": 12,
    "base_resolution": 17,
    "max_resolution": 64,
    "interpolation": "Linear" 
    # "interpolation": "Smoothstep" 
}
network_config = {
    "otype": "CutlassMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 32,
    "n_hidden_layers": 1
}
'''
6 * 2^13: 6 * 8,192 = 49,152
7 * 2^13: 7 * 8,192 = 57,344
'''
class HashLUT(nn.Module): 
    def __init__(self, nt="7+13", backbone='Backbone', use_mlp='mlp', min_max='9+64', *args, **kwargs):
        super().__init__()

        self.N = encoding_config["n_levels"] = int(nt.split("+")[0])
        T = encoding_config["log2_hashmap_size"] = int(nt.split("+")[1])
        D_min = encoding_config["base_resolution"] = int(min_max.split("+")[0])
        D_max = encoding_config["max_resolution"] = int(min_max.split("+")[1])
        b = encoding_config["per_level_scale"] = np.exp(np.log(D_max/D_min)/(self.N-1))
        print(f"{self.N} tables of range:{D_min}-{D_max}, T:{T}, b:{b:.3f}")
        
        self.use_mlp = not 'no' in use_mlp
        print("mlp: ", self.use_mlp)
        if use_mlp:
            self.hashluts = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.N*3, encoding_config=encoding_config, network_config=network_config)
        else:
            encoding_config["n_features_per_level"] = 4
            self.hashluts = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        
        
        net = eval(backbone)()
        last_channel = net.last_channel
        if 'Small' in backbone:
            self.expert = nn.Sequential(
                net,
                nn.Flatten(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(last_channel, self.N),
            ) 
        else:
            self.expert = nn.Sequential(
                net,
                nn.Flatten(),
                nn.Linear(last_channel, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(last_channel, self.N),
            ) 
        print(backbone, 'backbone')


    def forward(self, img, img_org, *args, **kwargs):
        assert img.shape[1] == 3 and img_org.shape[-1] == 3
        B, H, W, _ = img_org.shape # B H W C
        mid_results = self.hashluts(img_org.reshape(-1, 3)).reshape(B, H*W, self.N, -1)[...,:3] # B, HW, N, 3            
        weights = self.expert(img).reshape(B, 1, self.N, 1) # B, 1, N, 1
        img_res = (mid_results * weights).sum(2).reshape(B, H, W, 3) # B, H, W, 3
        
        return {
            "fakes": img_res,
        }
        
        


    


# 245,024 params
class Backbone(nn.Module): 
    def __init__(self): # org both
        super().__init__()
        self.model = nn.Sequential(
            *CnnActNorm(3, 16, normalization=True), # 128**16
            *CnnActNorm(16, 32, normalization=True), # 64**32
            *CnnActNorm(32, 64, normalization=True), # 32**64
            *CnnActNorm(64, 128, normalization=True), # 16**128
            *CnnActNorm(128, 128, normalization=False), # 8**128
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        )
        self.last_channel = 128

    def forward(self, x):
        return self.model(x)

# 61,456 parameters
class SmallBackbone(nn.Module): 
    def __init__(self, last_channel=64): # org both
        super().__init__()
        self.model = nn.Sequential(
            *CnnActNorm(3, 8, normalization=True), # 128**16
            *CnnActNorm(8, 16, normalization=True), # 64**32
            *CnnActNorm(16, 32, normalization=True), # 32**64
            *CnnActNorm(32, 64, normalization=True), # 16**128
            *CnnActNorm(64, 64, normalization=False), # 8**128
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        )
        self.last_channel = 64
    
    def forward(self, x):
        return self.model(x)
    
class TVMN(nn.Module): # (n,)3,d,d,d   or   (n,)3,d
    def __init__(self, dim=33, lambda_smooth=0.0001, lambda_mn=10.0):
        super(TVMN,self).__init__()
        self.dim, self.lambda_smooth, self.lambda_mn = dim, lambda_smooth, lambda_mn
        self.relu = torch.nn.ReLU()
       
        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float)
        weight_b[..., (0, dim - 2), :, :] *= 2.0        
        self.register_buffer('weight_r', weight_r, persistent=False)
        self.register_buffer('weight_g', weight_g, persistent=False)
        self.register_buffer('weight_b', weight_b, persistent=False)

        self.register_buffer('tvmn_shape', torch.empty(3), persistent=False)


    def forward(self, LUT): 
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        if len(LUT.shape) > 3: # n,3,d,d,d  or  3,d,d,d
            dif_r = LUT[...,:-1] - LUT[...,1:]
            dif_g = LUT[...,:-1,:] - LUT[...,1:,:]
            dif_b = LUT[...,:-1,:,:] - LUT[...,1:,:,:]
            tvmn[0] =   torch.mean(dif_r**2 * self.weight_r[:,0]) + \
                        torch.mean(dif_g**2 * self.weight_g[:,0]) + \
                        torch.mean(dif_b**2 * self.weight_b[:,0])
            tvmn[1] =   torch.mean(self.relu(dif_r * self.weight_r[:,0])**2) + \
                        torch.mean(self.relu(dif_g * self.weight_g[:,0])**2) + \
                        torch.mean(self.relu(dif_b * self.weight_b[:,0])**2)
            tvmn[2] = 0
        else: # n,3,d  or  3,d
            dif = LUT[...,:-1] - LUT[...,1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif**2
            dif[...,(0,dim-2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0

        return self.lambda_smooth*(tvmn[0]+10*tvmn[2]) + self.lambda_mn*tvmn[1]

def CnnActNorm(in_filters, out_filters, kernel_size=3, sp="2_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        B = x.size(0)

        if B == 1:
            assert 1 == trilinear.forward(lut,
                                          x,
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          B)
        elif B > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut,
                                          x.permute(1,0,2,3).contiguous(),
                                          output,
                                          dim,
                                          shift,
                                          binsize,
                                          W,
                                          H,
                                          B)
            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, B])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, B = int_package
        dim, shift, W, H, B = int(dim), int(shift), int(W), int(H), int(B)
        binsize = float(float_package[0])

        if B == 1:
            assert 1 == trilinear.backward(x,
                                           x_grad,
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           B)
        elif B > 1:
            assert 1 == trilinear.backward(x.permute(1,0,2,3).contiguous(),
                                           x_grad.permute(1,0,2,3).contiguous(),
                                           lut_grad,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           B)
        return lut_grad, x_grad

# trilinear_need: imgs=nchw, lut=3ddd or 13ddd
class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        
        if lut.shape[0] > 1:
            if lut.shape[0] == x.shape[0]: # n,c,H,W
                use_res = torch.empty_like(x)
                for i in range(lut.shape[0]):
                    use_res[i:i+1] = TrilinearInterpolationFunction.apply(lut[i:i+1], x[i:i+1])[1]
            else:
                n,c,h,w = x.shape
                use_res = torch.empty(n, lut.shape[0], c, h, w).cuda()
                for i in range(lut.shape[0]):
                    use_res[:,i] = TrilinearInterpolationFunction.apply(lut[i:i+1], x)[1]
        else: # n,c,H,W
            use_res = TrilinearInterpolationFunction.apply(lut, x)[1]
        return use_res
        # return torch.clip(TrilinearInterpolationFunction.apply(lut, x)[1],0,1)


if __name__ == "__main__":
    model = Backbone()
    small_model = SmallBackbone()
    inputs = torch.ones(1,1,3,256,256)
    # flops, params = profile(model, inputs)
    flops, params = profile(small_model, inputs)
    print(flops, params)