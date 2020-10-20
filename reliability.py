#!/usr/bin/env python
# encoding: utf-8

import csv
import pandas as pd
from kripp_juan import alpha as k_alpha 

INPUT_FILE = "data/TrackVarietyAnswers.csv"
VALUES_DOMAIN = ["List A", "List B", "I don't know"]

if __name__ == '__main__':



    reliability_data = pd.read_csv(INPUT_FILE)



    k_a = k_alpha(reliability_data.iloc[[5,3,4]], value_domain=VALUES_DOMAIN, 
                                        			level_of_measurement='nominal')


    print(k_a)


    k_a = k_alpha(reliability_data.iloc[[1,2,3]], value_domain=VALUES_DOMAIN, 
                                        			level_of_measurement='nominal')

    print(k_a)
