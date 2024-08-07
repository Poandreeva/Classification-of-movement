# Biotechnical Assistive Coaching System for Sports Fencing
## Overview
The Biotechnical Assistive Coaching System for Sports Fencing is designed to determine the quantitative characteristics of performed movements and their variation with increasing fatigue. This system allows coaches and athletes to obtain objective quantitative information about the parameters of movements at different points in the training and competition process, which is a necessity in sports fencing. The implementation of this system enables a comprehensive analysis of fencers' movements, enhancing the efficiency of the training process and improving sports results.

## Key Features
Wearable Inertial Sensors: Utilizes wearable inertial sensors that allow for more accurate measurement of movement parameters such as acceleration and speed, without the need for expensive high-speed video capture systems.
Automated Movement Analysis: The system automates the process of evaluating movement parameters, requiring the isolation and classification of individual movements.
## Experimental Research
### First Experimental Study:

* Developed movement classification algorithms using machine learning methods, achieving high accuracy and effectiveness.
* Employed logistic regression models that demonstrated perfect classification accuracy, proving the viability of using gyroscopic data for movement classification.
### Second Experimental Study:

* Focused on the impact of progressive fatigue on movement parameters.
* Significant reductions observed with increasing fatigue: the average maximum speed of lunging movements decreased by 25.5% from 228.3 cm/s in the first set to 170.1 cm/s in the third set, and the movement amplitude decreased by 29.9% from 135.4 cm to 94.9 cm.
* Fatigue assessment was based on Heart Rate Recovery (HRR) time calculations, with intensity corresponding to 71% of the maximum possible heart rate for the age, which falls within Zone 2 of aerobic intensity.
* Recovery times post-exercise showed a cumulative effect of fatigue: 137 seconds after the first set, increasing to 162 seconds after the second, and reaching 217 seconds after the third.
## Applications
The developed biotechnical system and movement analysis techniques can be applied not only in fencing but also in other individual sports where precision and coordination of movements are crucial. Integrating such systems into the training process can enhance the quality of athlete preparation and improve sports performance.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. Follow the instructions in `main.py` to perform various stages of data analysis.

## File Description

- `data_processing.py`: Contains functions for data processing and loading.
- `feature_extraction.py`: Contains functions for feature extraction from data.
- `movement_classification.py`: Contains models and methods for movement classification.
- `heart_rate_analysis.py`: Contains methods for heart rate recovery analysis.
- `visualizations.py`: Contains functions for data visualization.
- `main.py`: The main file to run the entire project.
- `utils/`: Folder with utility functions, such as `file_io.py`, `data_split.py`, and `plot_utils.py`.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
