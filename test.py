import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res



def main():
    parser = argparse.ArgumentParser(description='Multimodal Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--n_cluster', type=int, default=3,
                        help='number of clusters of k-means ')
    parser.add_argument('--alpha', default=1,
                        help='fusion degree, should be a float or a list which length is n_cluster')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='weight of pairwise term in alpha-expansion')
    parser.add_argument('--max_cycles', default=None,
                        help='max_cycles of alpha-expansion')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help=' pretrained model state')

    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model(n_cluster=args.n_cluster,
                  alpha=args.alpha,
                  device=device,
                  lam=args.lam,
                  max_cycles=args.max_cycles)
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
        print(f'{args.model_state_path} loaded')
    model = model.to(device)

    c = Image.open(args.content)
    s = Image.open(args.style)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor)
    out_denorm = denorm(out, device)

    if args.output_name is None:
        c_name = os.path.splitext(os.path.basename(args.content))[0]
        s_name = os.path.splitext(os.path.basename(args.style))[0]
        args.output_name = f'{c_name}_{s_name}'

    try:
        save_image(out_denorm, f'{args.output_name}.jpg', nrow=1)

        o = Image.open(f'{args.output_name}.jpg')
        s = s.resize((i // 4 for i in c.size))
        box = (0, o.height - s.height)
        o.paste(s, box)
        o.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)
        print(f'result saved into files starting with {args.output_name}')
    except:
        pass


if __name__ == '__main__':
    main()
