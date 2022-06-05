import os
import torch
import PIL.Image
import pandas as pd
import numpy as np

from torch.autograd import Variable


labels_map = {
    0: "bacterial_leaf_blight",
    1: "bacterial_leaf_streak",
    2: "bacterial_panicle_blight",
    3: "blast",
    4: "brown_spot",
    5: "dead_heart",
    6: "downy_mildew",
    7: "hispa",
    8: "normal",
    9: "tungro",
}


class Predicter(object):
    def __init__(self, args, model, transform):
        self.__args = args
        self.__model = model
        self.__transform = transform
        self.__model.load_state_dict(torch.load(args.weights_path), strict=True)

        if args.gpus == [0]:
            gpu_status = 'Single-GPU'
            device = torch.device("cuda:0")
            self.__model.to(device)
        else:
            gpu_status = 'Multi-GPU'
            self.__model = torch.nn.DataParallel(self.__model, device_ids=args.gpus, output_device=args.gpus[0])

        print('****************************************************************************************************')
        print('Model:')
        print(self.__model)
        print('****************************************************************************************************')
        print('GPU   Status: ' + gpu_status)
        print('****************************************************************************************************')
        self.__model.eval()

    def predict_csv(self):
        df = pd.read_csv(self.__args.submission_file_path)
        for index, row in df.iterrows():
            test_file_dir = os.path.join(self.__args.dataset_dir, 'train_valid_test', 'test', 'unknown', row[0])
            img = PIL.Image.open(test_file_dir)
            input_test = self.__transform(img).unsqueeze(0)
            input_test = Variable(input_test).cuda()
            with torch.no_grad():
                output_test = self.__model.forward(input_test)
                softmax = torch.nn.Softmax(dim=1)
                output_test = softmax(output_test)
            output_test = output_test.cpu().detach().numpy()
            label_test = np.argmax(output_test)
            df.iloc[index, 1] = (labels_map[label_test.item()])
        print(df)
        df.to_csv(self.__args.submission_file_path, index=None)


def main():
    pass


if __name__ == '__main__':
    main()
