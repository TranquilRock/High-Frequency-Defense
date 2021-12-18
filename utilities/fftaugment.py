import torch
import torch.fft as fft

class lowPassFilter(object):
    '''Transformation allow only low frequency component. Var limit must be with in 0,0.5'''
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, input):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) < self.limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) < self.limit
        kernel = torch.outer(pass2, pass1).cuda()
        fft_input = fft.rfft2(input)
        return fft.irfft2(fft_input * kernel, s=input.shape[-2:])

    def __repr__(self):
        return self.__class__.__name__+'()'

class highPassFilter(object):
    '''Transformation allow only high frequency component. Var limit must be with in 0,0.5'''
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, input):
        pass1 = torch.abs(fft.rfftfreq(input.shape[-1])) > self.limit
        pass2 = torch.abs(fft.fftfreq(input.shape[-2])) > self.limit
        kernel = torch.outer(pass2, pass1)
        fft_input = fft.rfft2(input)
        return fft.irfft2(fft_input * kernel, s=input.shape[-2:])

    def __repr__(self):
        return self.__class__.__name__+'()'

class highPassNoise(object):
    '''Transformation adding random high frequency component from dataset. Var limit must be with in 0,0.5'''
    def __init__(self, limit, eps, dataset, srng):
        self.limit = limit
        assert eps <= 1.0, "eps too large"
        self.eps = eps
        self.dataset = dataset
        self.dLen = len(self.dataset)
        self.srng = srng
    def __call__(self, input):
        # sample_idx = torch.randint(self.dLen, size=(1,)).item()
        sample_idx = self.srng.gen()
        randPic = self.dataset[sample_idx][0]
        pass1 = torch.abs(fft.rfftfreq(randPic.shape[-1])) > self.limit
        pass2 = torch.abs(fft.fftfreq(randPic.shape[-2])) > self.limit
        kernel = torch.outer(pass2, pass1).cuda()
        fft_randPic = fft.rfft2(randPic).cuda()
        return (1 - self.eps)*input + self.eps * fft.irfft2(fft_randPic * kernel, s=randPic.shape[-2:])
    def __repr__(self):
        return self.__class__.__name__+'()'

class highPassNoiseTarget(object):
    '''Make its target from (1) => (main_target, subtarget)'''
    def __init__(self, dataset, srng):
        self.dataset = dataset
        self.dLen = len(self.dataset)
        self.srng = srng

    def __call__(self, target: int):
        sample_idx = self.srng.gen()
        randLabel = self.dataset[sample_idx][1] 
        return torch.tensor([target, randLabel])
    def __repr__(self):
        return self.__class__.__name__+'()'

def get_thresholds(input):
    pass1 = torch.abs(fft.rfftfreq(input.shape[-1]))
    pass2 = torch.abs(fft.fftfreq(input.shape[-2]))
    return pass1, pass2
