# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import netCDF4
import base64
import json
import copy
import time
import os

key = "EJLuwPKKIBHvufUMqJil6ODk7_ga"
secret = "W4wY3VTFHwLZpxx1TGuV2kVNSRAa"

payload_template = {
    "datasetId": "EO:ECMWF:DAT:CAMS_SOLAR_RADIATION_TIMESERIES",
    "stringInputValues": [
        {"name": "latitude", "value": "0"},
        {"name": "longitude", "value": "0"},
        {"name": "altitude", "value": "-999"},
    ],
    "stringChoiceValues": [
        {"name": "sky_type", "value": "observed_cloud"},
        {"name": "time_step", "value": "15minute"},
        {"name": "time_reference", "value": "UT"},
        {"name": "format", "value": "netcdf"},
    ],
    "dateRangeSelectValues": [
        {"name": "date_range", "start": "2019-01-01", "end": "2019-01-10"}
    ],
}

job_id_request_url = "https://apis.wekeo.eu/databroker/0.1.0/datarequest"
job_status_url = "https://apis.wekeo.eu/databroker/0.1.0/datarequest/status/{}"
ext_uri_url = "https://apis.wekeo.eu/databroker/0.1.0/datarequest/jobs/{}/result"
download_url = "https://apis.wekeo.eu/databroker/0.1.0/datarequest/result/{job_id}?externalUri={ext_uri}"

class CopernicusWrapper:
    def __init__(
        self,
        latitude,
        longitude,
        end,
        start="2014-01-01",
        time_intervall="15minute",
        key=None,
        secret=None,
    ):
        self.lat = latitude
        self.long = longitude
        self.start = start
        self.end = end
        self.time_intervall = time_intervall
        if key==None or secret==None:
            try:
                self.key = os.environ['WEKEO_KEY']
                self.secret = os.environ['WEKEO_SECRET']
            except:
                raise RuntimeError('No credentials found. Provide with either function arguments or enviromental variables (WEKEO_KEY and WEKEO_SECRET)')

    def download_data(self, filepath):
        code = str(base64.b64encode((key + ":" + secret).encode("utf-8")), "utf-8")
        r = requests.post(
            "https://apis.wekeo.eu/token",
            headers={"Authorization": "Basic {}".format(code)},
            data={"grant_type": "client_credentials"},
        )
        access_token = r.json()["access_token"]
        headers = {"Authorization": "Bearer {}".format(access_token),
                   'Content-Type': 'application/json'}

        payload = copy.copy(payload_template)
        payload["stringInputValues"][0]["value"] = str(self.lat)
        payload["stringInputValues"][1]["value"] = str(self.long)
        payload["stringChoiceValues"][1]["value"] = self.time_intervall
        payload["dateRangeSelectValues"][0]["start"] = self.start
        payload["dateRangeSelectValues"][0]["end"] = self.end

        job_id_response = requests.post(
            job_id_request_url, headers=headers, data=json.dumps(payload)
        )
        job_id = job_id_response.json()["jobId"]
        if job_id_response.status_code == 200:
            print("Request accepted. Waiting for response.")
        for i in range(100):
            print(".", end="")
            time.sleep(6)
            job_status_response = requests.get(
                job_status_url.format(job_id), headers=headers
            )
            if job_status_response.json()["status"] == "FAILED":
                raise IOError(job_status_response.json()['message'])
            if job_status_response.json()["complete"]:
                print("Processing complete. Starting download", end="")
                ext_uri_r = requests.get(ext_uri_url.format(job_id), headers=headers)
                ext_uri = ext_uri_r.json()["content"][0]["externalUri"]

                url = download_url.format_map({"job_id": job_id, "ext_uri": ext_uri})
                r = requests.get(url, headers=headers, stream=True)
                if r.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                            print(".", end="")
                break
        else:
            raise IOError('Request time out.')

if __name__ == "__main__":
    wrapper = CopernicusWrapper(latitude=52.5, longitude=13.4, end="2015-01-10")
    wrapper.download_data(filepath="berlin_2014.nc")