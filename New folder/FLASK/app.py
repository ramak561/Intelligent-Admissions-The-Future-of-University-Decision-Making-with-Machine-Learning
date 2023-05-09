#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(_name_)
# Import necessary libraries
from tensorflow.keras.models import load_model

#model = pickle.load(open('university.pkl','rb'))

