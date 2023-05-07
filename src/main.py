import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from model.sspcab_torch import MY_SSPCAB

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            sub_model = MY_SSPCAB(channels=3, reduction_ratio=8)
            sub_model.to('cuda')
            t = Trainer(args, loader, _model, _loss, checkpoint, sub_model)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
