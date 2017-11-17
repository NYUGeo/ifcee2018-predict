# Pile Capacity Predictor (beta)

### Click [here](http://cue3.engineering.nyu.edu:5012) for the live web application, or read below to run locally.

<br>

This online tool features a *Support Vector Regressor* to predict the axial load capacity of pile foundations given soil type, average SPT-N values, pile type and open/closed end condition, pile cross sectional area, circumference and length. The process is outlined in:

>Machairas, N. P., and Iskander, M. G. (2018). “An Investigation of Pile Design Utilizing Advanced Data Analytics.” *Proceedings of the International Foundations Congress and Equipment Expo 2018*, ADSC-The International Association of Foundation Drilling, DFI (Deep Foundations Institute), G-I (Geo-Institute of American Society of Civil Engineers), and PDCA (Pile Driving Contractors Association), March 5-10, 2018, Orlando, Florida.

#### DISCLAIMER

This tool is offered without any warranties about the accuracy of the predicted capacity. The predicted capacity is a result of approximation by scientific methodologies. The authors' sole intent is to further advance the field of Geotechnical Engineering and are not offering this online tool as a design aid. **Use to learn and experiment, do not design piles based on the numbers you get below.**

<br>

## Installation

Developed in Python 3.6. Create a virtual environment and install necessary packages from `requirements.txt`. Run with:

```
bokeh serve --show ifcee2018_predict.py
```
