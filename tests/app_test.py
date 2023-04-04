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
            assert len(res)>0

    def test_clean_text_return_value_text(self):
           res = app.clean_text("i love python")
           assert res=="i love python"

    def test_clean_text_send_value_plus_return_cleantext(self):
           res = app.clean_text("i+ love+ python")
           assert res=="i love python"