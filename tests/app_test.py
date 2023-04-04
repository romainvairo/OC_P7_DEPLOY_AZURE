# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
sys.path.insert(0, "../")
from enum import Enum
import urllib
import json
import requests
import aiounittest
from typing import Dict
from datetime import datetime
import app

class testunit(aiounittest.AsyncTestCase):
    def test_on_prediction_return_value(self):
            res = app.predict_sentiment("happy")
            assert res

    def test_on_prediction_return_empty(self):
            res = app.predict_sentiment(0)
            assert res