from datetime import datetime, timedelta

year = '2022'
month = 9
datetime_str = f'{year}-{month:02d}-01 00:00:00'
print(datetime_str)
start_date = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

print(start_date)  # printed in default format

end_date: datetime = start_date + timedelta(seconds = -1) # datetime.datetime(year, end_mo + 1, 1)
print(end_date)
