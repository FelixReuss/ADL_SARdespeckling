from adl_sardespeckling.train import train_model
from adl_sardespeckling.train import make_prediction
from adl_sardespeckling.postprocessing import despeckle_sar_image
import argparse


if __name__ == '__main__':

    #Initializing argument parser
    parser = argparse.ArgumentParser()

    #Required argument to parse the function to call
    parser.add_argument("-function",  "--function", type=str,
                        help="String indicating the function to call: train to train a model, predict to predict data, despeckle to despeckle an image" , required=True)

    #Optional arguments for the function train_model
    parser.add_argument("--train_path", type=str,
                        help="Path to the input train data", required=False)
    parser.add_argument("--reference_path", type=str,
                        help="Path to the reference data", required=False)
    parser.add_argument("--batch_size", type=int,
                        help="Batch size to feed to the model", required=False)
    parser.add_argument("--steps_per_epochs", type=int,
                        help="Number of steps per eopch. Steps time batch size should equal number of samples", required=False)
    parser.add_argument("--patch_size", type=int,
                        help="Patch size of the train and reference data", required=False)
    parser.add_argument( "--n_channels", type=int,
                        help="Number of channels of the input data", required=False)
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs to train the network", required=False)
    parser.add_argument("--save_model", type=str,
                        help="If a path is provided, the model is saved to this path", required=False)

    # Optional arguments for the function make prediction
    parser.add_argument("--path2input", type=str,
                        help="Path to the input image", required=False)
    parser.add_argument("--path2model", type=str,
                        help="Path to the model h5 file", required=False)
    parser.add_argument("--outpath", type=str,
                        help="Ouput path for the predicted iamge", required=False)

    # Optional arguments for the function despeckle sar image
    parser.add_argument("--input_path", type=str,
                        help="Path to the input image", required=False)
    parser.add_argument("--output_path", type=str,
                        help="Ouput path for the despeckled iamge", required=False)
    parser.add_argument("--pathmodel",type=str,
                        help="Path to the model h5 file", required=False)
    parser.add_argument("--overlay", type=int,
                        help="Overlay between the image patches", required=False)

    # Parse arguments
    args = parser.parse_args()

    # If argument is train call the train_model function with the respective parameters
    if args.function == 'train':
        train_model(image_path=args.train_path, reference_path=args.reference_path, steps_per_epoch=args.steps_per_epochs, batch_size=args.batch_size, patch_size=args.patch_size, n_channels=args.n_channels, epochs=args.epochs, save_model=args.save_model)

    # If argument is predict call the make_prediction function with the respective parameters
    if args.function == 'predict':
        make_prediction(path2input=args.path2input, path2model=args.path2model, outpath=args.outpath)

    # If argument is despeckle call the despeckle_sar_image function with the respective parameters
    if args.function == 'despeckle':
        despeckle_sar_image(input_path=args.input_path, output_path=args.output_path, path2model=args.pathmodel, overlay=args.overlay)

