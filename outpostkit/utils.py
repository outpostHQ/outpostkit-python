from datetime import datetime


def convert_outpost_date_str_to_date(date_string:str)->datetime:
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")
