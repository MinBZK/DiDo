import common
from datetime import datetime

def iso_cet_date(datum: datetime):
    result = datum.strftime("%Y-%m-%d %H:%M:%S CET")

    return result


print("Current Time =", common.iso_cet_date(datetime.now()))

datetime_str = '09/19/22 13:55:26'
datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
print(iso_cet_date(datetime_object))

date_str = '2022-09-19'
date_object = datetime.strptime(date_str, '%Y-%m-%d').date()
print(iso_cet_date(date_object))  # printed in default format