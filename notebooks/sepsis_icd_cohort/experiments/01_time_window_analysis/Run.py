import os
import sys

import logging

log = logging.getLogger("EHR-ML")
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.FileHandler(filename=os.environ['EICU_EHR_PIPELINE_BASE'] + '/data/experiments/01_time_window_analysis/ehr-ml.log')
ch.setFormatter(format)
log.addHandler(ch)
import warnings
warnings.simplefilter(action='ignore', category=Warning)


sys.path.append(os.environ['EICU_EHR_PIPELINE_BASE'] + "/EHR-ML")

from ehrml.utils import DataUtils
from ehrml.utils import MlUtils


for windowEnd in range(11, 15):
    for windowStart in range(3, 4):
        try:
            log.info('Starting - windowStart:' + str(windowStart) + ' windowEnd:' + str(windowEnd))
            log.info('Fetching EHR data')
            data = DataUtils.readEicuData(
                dirPath=os.environ['EICU_EHR_PIPELINE_BASE'] + '/data/final/data_matrix.csv',
                windowStart=windowStart,
                windowEnd=windowEnd
                )
            X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y = data

            log.info('Building ML models')

            xgbEnsembleScores = MlUtils.buildEnsembleXGBoostModel(X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y)

            log.info('Saving results')

            DataUtils.saveCvScores(
                scores_dict=xgbEnsembleScores,
                dirPath=os.environ['EICU_EHR_PIPELINE_BASE'] + '/data/experiments/01_time_window_analysis',
                fileName='ws_' + str(windowStart) + '_we_' + str(windowEnd) + '.json'
                )
        except Exception as e:
            log.error('Exception!!')
            log.error(e)
