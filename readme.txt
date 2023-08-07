files usage:

228_training_aerial_imagery.py - python scritp to import images or numpy arrays from an 'images' folder contained in the same directory. To be used as the main training file
UNET.ipynb - notebook used to develop the training script above. Good for testing different methods and data types
mask_creation_framework.ipynb - Creates the input file for the training to take place. Takes in two files, a geotiff image and the matching shapefile with training data. Output is png image of the geotiff and a png of the shapefile or a numpy array of the shapefile
mask_creation_functions.py - supporting file with functions used in the training script
predict_on_image.py - used after the model is trained and created to infer on an image
simple_nulti_unet_model.py - UNET framework setup function. Used in training script
smooth_tiled_predictions.py - used to split up image into tiles for prediction in manageable size
