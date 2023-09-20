import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-7, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.8, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.95, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--Z_dim", type=int, default=1828, help="dimension of the latent space (noise)")
    parser.add_argument("--Stru_dim", type=int, default=1826, help="dimension of molecular descriptors")
    parser.add_argument("--Time_dim", type=int, default=1,
                        help="dimension of sacrificed time point (4,8,15,29 days)")
    parser.add_argument("--Dose_dim", type=int, default=1, help="dimension of dose level (low:middle:high=1:3:10)")
    parser.add_argument("--Measurement_dim", type=int, default=38,
                        help="dimension of Hematology and Biochemistry measurements")
    parser.add_argument("--n_critic", type=int, default=5, help="number of critic iterations per generator iteration")
    parser.add_argument("--interval", type=int, default=500, help="number of intervals you want to save models")
    parser.add_argument("--lambda_gp", type=float, default=1.0, help="strength of the gradient penalty regularization term")
    parser.add_argument("--model_path", type=str, default='./models', help="path to model saving folder")

    parser.add_argument("--filename_Losses", type=str, default='Loss.txt', help="filename of losses")

    parser.add_argument("--num_generate", type=int, default=100, help="number of blood testing records you want to generate")
    opt = parser.parse_args()

    return opt
