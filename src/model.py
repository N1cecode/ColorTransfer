from src.transformer import *

class TransformerEncoder(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution,  num_heads=4, window_size=8,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU(), drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList(
            [nn.InstanceNorm2d(dim, affine=True), act_layer, 
             PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim),
             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path),
             PatchUnEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim),
             
             nn.InstanceNorm2d(dim, affine=True), act_layer, 
             PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim),
             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path),
             PatchUnEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim)
             ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    

class TransformerDecoder(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
    """

    def __init__(self, dim, input_resolution, style_dim=64, num_heads=4, window_size=8,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU(), drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        # build blocks
        self.blocks = nn.ModuleList(
            [AdaIN(style_dim=style_dim, dim=dim, act_layer=act_layer), 
             PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim),
             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path),
             PatchUnEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim),
             
             AdaIN(style_dim=style_dim, dim=dim, act_layer=act_layer),
             PatchEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim),
             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path),
             PatchUnEmbed(img_size=input_resolution, patch_size=1, in_chans=dim, embed_dim=dim)
             ])

    def forward(self, x, s):
        for block in self.blocks:
            x = block(x, s)
        return x


class Downsample(nn.Module):
    def __init__(self, scale, in_dim, out_dim, input_resolution=None):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim // (scale**2), 3, 1, 1)
        self.down = nn.PixelUnshuffle(scale)
    
    def forward(self, x):
        x = self.conv(x) + x
        x = self.down(x)
        return x


class Upsample(nn.Module):
    def __init__(self, scale, in_dim, out_dim, input_resolution=None):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, (scale**2)*out_dim, 3, 1, 1)
        self.up = nn.PixelShuffle(scale)
    
    def forward(self, x):
        x = self.conv(x) + x
        x = self.up(x)
        return x


class EncodeBLK(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, scale=None, num_heads=4, window_size=8, mlp_ratio=2):
        super().__init__()
        self.scale = scale
        self.res = input_resolution
        self.transformer = TransformerEncoder(dim=in_dim, input_resolution=[input_resolution, input_resolution], 
                                              num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio)

        if self.scale:
            self.down = Downsample(scale=scale, in_dim=in_dim, out_dim=out_dim)
        
    def forward(self, x):
        x = self.transformer(x) + x
        if self.scale:
            x = self.down(x)
        return x
    

class DecodeBLK(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, style_dim=64, scale=None, num_heads=4, window_size=8, mlp_ratio=2):
        super().__init__()
        self.scale = scale
        if self.scale:
            self.up = Upsample(scale=self.scale, in_dim=in_dim, out_dim=out_dim)

        self.res = input_resolution

        self.transformer = TransformerDecoder(dim=out_dim, input_resolution=[self.res, self.res], style_dim=style_dim,
                                              num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio)
        
    def forward(self, x, s):
        if self.scale:
            x = self.up(x)
        x = self.transformer(x, s) + x
        return x


class HighPass(nn.Module):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))
    
    
class Generator(nn.Module):
    def __init__(self, img_size=256, input_channel=3, dim=64, max_dim=512,
                mlp_ratio=2, num_heads=4, patch_size=1, window_size=8,
                style_dim=64, w_hpf=1, skip_size=[32,64,128], act_layer=nn.GELU()):
        super().__init__()
        self.img_size = img_size
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.skip_size = skip_size
        
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        
        self.from_img = nn.Conv2d(input_channel, dim, 3, 1, 1)
        self.to_img = nn.Sequential(
            nn.InstanceNorm2d(dim, affine=True),
            act_layer,
            nn.Conv2d(dim, 3, 1, 1, 0)
        )
        
        # down/up-sampling blocks
        # 固定最终瓶颈层的特征的H*W为16*16，有hpf则固定为8*8
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        
        resolution = img_size

        low_dim = dim
        high_dim = 0
        for i in range(repeat_num):
            high_dim = min(low_dim*2, max_dim)
            # stack-like
            self.encode.append(EncodeBLK(scale=2, in_dim=low_dim, out_dim=high_dim, input_resolution=resolution,
                                         num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio))

            self.decode.insert(0, DecodeBLK(scale=2, in_dim=high_dim, out_dim=low_dim, style_dim=style_dim, input_resolution=resolution,
                                             num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio))  
            
            low_dim = high_dim
            resolution = resolution // 2
        
        # bottleneck blocks
        for _ in range(2):
            self.encode.append(EncodeBLK(scale=None, in_dim=high_dim, out_dim=high_dim, input_resolution=resolution,
                                          num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio))
            self.decode.insert(0, DecodeBLK(scale=None, in_dim=high_dim, out_dim=high_dim, style_dim=style_dim, input_resolution=resolution,
                                            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio))
        
        self.hpf = HighPass(1)

    def forward(self, x, s):
        x = self.from_img(x)
        cache = {}
        for block in self.encode:
            if (x.size(2) in self.skip_size):
                cache[x.size(2)] = x
            x = block(x)

        for block in self.decode:
            x = block(x, s)
            if (x.size(2) in self.skip_size):
                x = x + self.hpf(cache[x.size(2)])
                
        return self.to_img(x)


class Discriminator(nn.Module):
    def __init__(self, img_size=256, input_channel=3, dim=64, max_dim=512, num_domains=2, 
                mlp_ratio=2, num_heads=4, patch_size=1, window_size=8, act_layer=nn.GELU()):
        super().__init__()
        self.img_size = img_size
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        
        self.blocks = nn.ModuleList()
        
        self.from_img = nn.Conv2d(input_channel, dim, 3, 1, 1)
        
        # down-sampling blocks
        # 下采样至4*4
        repeat_num = int(np.log2(img_size)) - 2
        
        resolution = img_size
        in_dim = dim
        out_dim = 0
        for _ in range(repeat_num):
            out_dim = min(in_dim*2, max_dim)
            self.blocks.append(EncodeBLK(scale=2, in_dim=in_dim, out_dim=out_dim, input_resolution=resolution,
                                         num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio))
            in_dim = out_dim
            resolution = resolution // 2
        
        self.blocks.append(act_layer)
        self.blocks.append(nn.Conv2d(out_dim, out_dim, 4, 1, 0))
        self.blocks.append(act_layer)
        self.blocks.append(nn.Conv2d(out_dim, num_domains, 1, 1, 0))

    def forward(self, x, y):
        x = self.from_img(x)
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        x = x[idx, y]
        return x


class ColorEncoder(nn.Module):
    def __init__(self, img_size=256, input_channel=3, dim=64, max_dim=512, num_domains=2, style_dim=64,
                mlp_ratio=2, num_heads=4, patch_size=1, window_size=8, act_layer=nn.GELU()):
        super().__init__()
        
        self.img_size = img_size
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        
        self.body = nn.ModuleList()
        
        self.from_img = nn.Conv2d(input_channel, dim, 3, 1, 1)
        
        # down-sampling blocks
        # 下采样至4*4
        repeat_num = int(np.log2(img_size)) - 2
        
        resolution = img_size
        in_dim = dim
        out_dim = 0
        for _ in range(repeat_num):
            out_dim = min(in_dim*2, max_dim)
            self.body.append(EncodeBLK(scale=2, in_dim=in_dim, out_dim=out_dim, input_resolution=resolution,
                                         num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio))
            in_dim = out_dim
            resolution = resolution // 2
        
        self.body.append(act_layer)
        self.body.append(nn.Conv2d(out_dim, out_dim, 4, 1, 0))
        self.body.append(act_layer)
        
        self.head = nn.ModuleList()
        for _ in range(num_domains):
            self.head += [nn.Linear(out_dim, style_dim)]
        
    def forward(self, x, y):
        x = self.from_img(x)
        for block in self.body:
            x = block(x)
        x = x.view(x.size(0), -1)
        
        out = []
        for layer in self.head:
            out += [layer(x)]
        out = torch.stack(out, dim=1)   # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y] # (batch, style_dim)
        return s


class MappingNet(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2, act_layer=nn.GELU()):
        super().__init__()
        
        self.body = nn.ModuleList()
        self.body.append(nn.Linear(latent_dim, 512))
        self.body.append(act_layer)
        
        for _ in range(3):
            self.body.append(nn.Linear(512, 512))
            self.body.append(act_layer)
            
        self.head = nn.ModuleList()
        for _ in range(num_domains):
            self.head += [nn.Sequential(nn.Linear(512, 512),
                                        act_layer,
                                        nn.Linear(512, 512),
                                        act_layer,
                                        nn.Linear(512, 512),
                                        act_layer,
                                        nn.Linear(512, style_dim))]

    def forward(self, z, y):
        for block in self.body:
            z = block(z)
        
        out = []
        for layer in self.head:
            out += [layer(z)]
        out = torch.stack(out, dim=1)   # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


import copy
from munch import Munch
# from src.wing import FAN
def build_model(args):
    generator = nn.DataParallel(Generator(img_size=args.img_size, style_dim=args.style_dim))
    mapping_network = nn.DataParallel(MappingNet(latent_dim=args.latent_dim, style_dim=args.style_dim, num_domains=args.num_domains))
    style_encoder = nn.DataParallel(ColorEncoder(img_size=args.img_size, style_dim=args.style_dim, num_domains=args.num_domains))
    discriminator = nn.DataParallel(Discriminator(img_size=args.img_size, num_domains=args.num_domains))
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
    #     fan.get_heatmap = fan.module.get_heatmap
    #     nets.fan = fan
    #     nets_ema.fan = fan

    return nets, nets_ema

if __name__ == "__main__":
    from torchinfo import summary
    
    x = torch.rand(1, 3, 256, 256)
    y = torch.tensor([1])
    z = torch.rand(1, 16)
    s = torch.rand(1, 64)
    
    G = Generator()
    summary(G, input_data=(x, s))
    
    D = Discriminator()
    D(x, y)
    summary(D, input_data=(G(x, s), y))
    
    E = ColorEncoder()
    summary(E, input_data=(x, y))
    
    M = MappingNet()
    summary(M, input_data=(z, y))