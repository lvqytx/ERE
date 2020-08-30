from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import requests
import json
import logging

executor = ThreadPoolExecutor(2)
app = Flask(__name__)
logger = logging.getLogger()


@app.route('/job/create', methods=["POST"])
def run_jobs():
    # 交由线程去执行耗时任务
    data = request.get_json()
    process_id = data["process_id"]
    filename = data["filename"]

    executor.submit(long_task, 'hello', 123)
    return 'long task running.'


def update_job(finish, process_id, percent, filename):
    payload = {"status": finish, "id": process_id, "percent": percent, "filename": filename}
    response = requests.post("http://localhost:8080/api/process", data=json.dumps(payload))
    logger.info(response)


# 耗时任务
def long_task(arg1, arg2):
    print("args: %s %s!" % (arg1, arg2))
    sleep(5)
    print("Task is done!")


if __name__ == '__main__':
    app.run()
