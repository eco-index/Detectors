Model:
The native forest detector is an artificial neural network which is based on a UNET framework. UNETs are well known in the remote sensing realm to have strong predictive abilities to differentiate different types of ground cover and ecosystems. The input data used to train and predict with the UNET was a combination of Planet Labs 8 band multispectral data (https://developers.planet.com/docs/apis/data/sensors/), Copernicus Sentinel-1 SAR data, and NASA Digital Elevation Map data. All data types were collected from the months of August to October of 2020 in order to have the closest matching dates to the Land Cover Database v5 timestamp 2018 training dataset. The three data sources were combined into a data cube before being fed into the UNET for training. In total, 3000 data cubes at a size of 2048x2048 pixels were used for training the model. This model was then run for 100 iterations before being determined visually that further training steps were not increasing the accuracy of predictions. 

Training Data:
The Land Cover Database was used as the training dataset as the predicted variable. The dataset first had the vegetated layers separated and then combined into five groups of classes to be used for training the model: 
Native forest ecosystems - Broadleaf Indigenous Forest, Indigenous Forest, Mānuka and/or Kānuka (used to form the Eco-index Observer native forest detections).
Grasslands - Alpine Grass/ Herbfield, Depleted Grassland, High Producing Exotic Grassland, Low Producing Grassland, Short-Rotation Cropland, Tall Tussock Grassland, Urban Parkland
Exotic forests - Deciduous Hardwood, Exotic Forest, Mixed Exotic Shrubland
Scrub - Gorse and/or Broom, Matagouri or Grey Scrub, Sub Alpine Shrubland
Other - Remaining LCDB v5 classes.
These groups of classes were then fed into the model during training to find repeating patterns using the neural network. Even though the five classes were used, only the outputs of the native forest ecosystems were validated due to time restrictions. The remaining classes were still necessary to create a balanced dataset for building the model, even though they are currently unvalidated. 

Prediction:
Predictions from the created model were then run on imagery from the most recent timeframe according to the clients request. The outputs are provided as an image fine containing colours for each class mentioned above. These images are then turned into a vector file format (shapefile) in order to compare the prediction to the ground truth in the validation step.

Validation: 
A subsample of 10% of the predictions are randomly selected for manual validation. This process creates sample squares of approximately 37km2 from the predicted area. The shapefile outputs are then transferred into either a google earth portal or an Eco-index based portal for manual validation. The choice of portal depends on the area being covered and the number of predictions. A validation area of more than 1000 predictions will go into the Eco-index portal and an area with less than 1000 predictions will go into the google earth portal. This difference is due to the ability of google earth to process large amounts of predictions, however, the google earth portal provides higher resolution imagery to validate against and is generally preferred. 

Results:
The output from the validation exercise is a report representing the accuracy of the prediction across the entire predicted area. The shapefile data is then used to calculate a variety of outputs for the clients including: total area of native forest within the given boundaries, amount of change of native forest between time points, area breakdown of smaller catchments within a larger boundary. These results are then provided in a report or as an excel formatted file. 
