from argparse import ArgumentParser

class Config:
    '''Argument Parser.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='configuration options for deep fusion using HydraFusion.')
        self.parser.add_argument('--activation', type=str, default='relu', help='Activation function to use, options: [relu, leaky_relu].')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Number of inputs in a batch. NOTE: only batch size 1 works currently.')
        self.parser.add_argument('--device', type=str, default="cuda", help='The device on which models are run, options: [cuda, cpu].')
        self.parser.add_argument('--pretrained', type=bool, default=False, help='loads pretrained ImageNet resnet18 weights into branches and stems.')
        self.parser.add_argument('--resume', type=bool, default=False, help='resume training from checkpoint.') 
        self.parser.add_argument('--fusion_type', type=str, default=None, help='type of fusion to perform across branch outputs.')
        self.parser.add_argument('--fusion_sweep', type=bool, default=False, help='Perform a sweep over the fusion tuning parameters.')
        self.parser.add_argument('--use_custom_transforms',type=bool, default=True, help='set true to use RADIATE transforms. set false to use ImageNet transforms.')
        self.parser.add_argument('--create_gate_dataset', type=bool, default=False, help="set to true to generate a gate training dataset.")
        self.parser.add_argument('--enable_radar', type=bool, default=True)
        self.parser.add_argument('--enable_camera', type=bool, default=True)
        self.parser.add_argument('--enable_lidar', type=bool, default=True)
        self.parser.add_argument('--enable_cam_fusion', type=bool, default=True)
        self.parser.add_argument('--enable_radar_lidar_fusion', type=bool, default=True)
        self.parser.add_argument('--enable_cam_lidar_fusion', type=bool, default=True)


        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        print("Configuration loaded successfully.")