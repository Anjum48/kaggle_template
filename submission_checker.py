import datetime
import time
from datetime import timezone

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import COMP_NAME

api = KaggleApi()
api.authenticate()


result_ = api.competition_submissions(COMP_NAME)[0]
latest_ref = str(result_)
submit_time = result_.date

status = ""

while status != "complete":
    list_of_submission = api.competition_submissions(COMP_NAME)
    for result in list_of_submission:
        if str(result.ref) == latest_ref:
            break
    status = result.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time = int((now - submit_time).seconds / 60) + 1
    if status == "complete":
        print("\r", f"run-time: {elapsed_time} min, LB: {result.publicScore}")
    else:
        print("\r", f"elapsed time: {elapsed_time} min", end="")
        time.sleep(60)
